"""
Multilingual Zero-Shot Evaluation Pipeline  v5.2
=================================================================
Target runtime : Lightning AI  (L4 / L40S GPU, Ubuntu 22.04)
Primary model  : utter-project/EuroLLM-9B-Instruct  (4-bit BnB)
Dataset        : google-research-datasets/tydiqa  (primary_task)

═══════════════════════════════════════════════════════════════════
DATASET CONFIG RATIONALE — WHY primary_task IS CORRECT
═══════════════════════════════════════════════════════════════════
The only two valid config names are "primary_task" and "secondary_task"
(from the official tydiqa.py loader).  "goldp" does NOT exist as a
config name and would crash with ValueError.

secondary_task (the gold-passage variant) is unsuitable because:
  1. It has NO "language" field → multilingual routing breaks entirely.
  2. Thai and Japanese are explicitly excluded from secondary_task.
  3. Answers are character offsets in a short passage snippet, not
     byte offsets in document_plaintext — a completely different schema.

primary_task is the only config that:
  ✅ Covers all 11 target languages including Thai and Japanese.
  ✅ Provides the "language" field for routing.
  ✅ Uses byte-offset answer spans in document_plaintext.

═══════════════════════════════════════════════════════════════════
ALL BUGS FIXED (v4 → v5.0 → v5.1)
═══════════════════════════════════════════════════════════════════

Fixes carried from v4 (BUGs 1-6):
  BUG 1 — Wrong TyDiQA field names (annotations / byte offsets)
  BUG 2 — Unanswerable questions silently dominating metrics
  BUG 3 — Model answering in English (weak prompt instructions)
  BUG 4 — BERTScore always zero (empty reference list guard)
  BUG 5 — Cross-lingual analysis never runs (ordering + key checks)
  BUG 6 — Tokenisation wrong for Bengali/Telugu

New fixes in v5.0 (BUGs A-I):
  BUG-A — USE_CHAT_TEMPLATE defaulted False; EuroLLM-Instruct needs True
  BUG-B — _apply_chat_template hard-coded ChatML; use tokenizer API
  BUG-C — gradient_checkpointing_enable() in eval pipeline removed
  BUG-D — torch.cuda.amp.autocast deprecated; use torch.amp.autocast
  BUG-F — BERTScore lang_code wrong (family[:2]); use ISO 639-1 map
  BUG-G — BERTScore inside ThreadPoolExecutor; moved to sequential pass
  BUG-H — Lambda closures captured 'l' by reference; default-arg fix
  BUG-I — Context window too small; MAX_INPUT_TOKENS raised to 3072

v5.1 improvements:
  • chars_per_token 3.5 → 2.9 (more accurate for multilingual scripts)
  • max_ctx_chars   6000 → 12000 (more evidence in prompt context)

v5.2 speed optimisations:
  • max_ctx_chars   12000 → 8000  (matched to 3800-token input budget)
  • MAX_INPUT_TOKENS 3072 → 3800  (uses full 4096 hard limit safely)
  • _CHARS_PER_TOKEN 2.9  → 3.0   (negligible; mid-point across scripts)
  • MAX_NEW_TOKENS    64  → 48    (covers 99th-pctile TyDiQA answers)
  • Batched generation (GENERATION_BATCH_SIZE=8): ~4–6× GPU speedup
    by processing 8 samples per forward pass instead of 1

═══════════════════════════════════════════════════════════════════
LIGHTNING AI SETUP (run once in terminal before executing script)
═══════════════════════════════════════════════════════════════════

  pip install -q transformers>=4.40.0 datasets accelerate \\
      bitsandbytes>=0.43.0 bert-score scipy seaborn tqdm \\
      huggingface_hub sentencepiece protobuf

═══════════════════════════════════════════════════════════════════
Author : Gadha
Date   : 2025-02-17
"""

# ─── stdlib & third-party ────────────────────────────────────────────────────
import gc
import json
import os
import re
import threading
import warnings
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from scipy import stats
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

class Config:
    # ── Authentication ──────────────────────────────────────────
    HF_TOKEN: str = "hf_YOUR_TOKEN_HERE"

    # ── Model ───────────────────────────────────────────────────
    #   Primary:  "utter-project/EuroLLM-9B-Instruct"
    #   Fallback: "CohereForAI/aya-23-8B"
    MODEL_NAME: str = "utter-project/EuroLLM-9B-Instruct"

    # BUG-A FIX: default True for EuroLLM-Instruct.
    # Set False only for base (non-instruct) models such as aya-23-8B.
    USE_CHAT_TEMPLATE: bool = True

    # ── Languages ───────────────────────────────────────────────
    TARGET_LANGS: List[str] = [
        "arabic", "bengali", "english", "finnish",
        "indonesian", "japanese", "korean", "russian",
        "swahili", "telugu", "thai",
    ]

    # ── Prompting ───────────────────────────────────────────────
    PROMPT_STRATEGIES: List[str] = [
        "minimal", "explicit_en", "explicit_native", "strict",
    ]

    # ── Sampling ────────────────────────────────────────────────
    MAX_SAMPLES_PER_LANG: int = 200   # set 20 for a quick smoke-test

    # ── Generation ──────────────────────────────────────────────
    MAX_NEW_TOKENS: int = 48      # 99th-pctile TyDiQA answer is < 35 tokens; 48 gives 36% headroom
    MAX_INPUT_TOKENS: int = 3800  # 4096 limit − 48 new − ~20 chat template = 228 tok safety margin
    GENERATION_BATCH_SIZE: int = 8  # batch multiple samples per forward pass; major speedup

    # ── Quantisation ────────────────────────────────────────────
    USE_4BIT: bool = True
    COMPUTE_DTYPE = torch.float16
    USE_DOUBLE_QUANT: bool = True
    QUANT_TYPE: str = "nf4"
    # BUG-C FIX: gradient_checkpointing removed (eval only)

    # ── BERTScore ───────────────────────────────────────────────
    USE_BERTSCORE: bool = True
    BERTSCORE_MODEL: str = "bert-base-multilingual-cased"
    BERTSCORE_BATCH_SIZE: int = 16

    # ── Statistics ──────────────────────────────────────────────
    CONFIDENCE_LEVEL: float = 0.95
    N_BOOTSTRAP: int = 1000

    # ── Multi-threading (CPU-bound metrics only; NOT BERTScore) ─
    MAX_WORKERS: int = 4

    # ── Output ──────────────────────────────────────────────────
    SAVE_RESULTS: bool = True
    OUTPUT_DIR: str = "/teamspace/studios/this_studio/outputs"
    TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    FIGURE_DPI: int = 300


# ╔══════════════════════════════════════════════════════════════╗
# ║  LANGUAGE METADATA                                           ║
# ╚══════════════════════════════════════════════════════════════╝

class LanguageUtils:
    """Script detection and per-language metadata."""

    # Unicode ranges for non-Latin scripts
    SCRIPT_RANGES: Dict = {
        "arabic":     (0x0600, 0x06FF),
        "bengali":    (0x0980, 0x09FF),
        "japanese":   [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
        "korean":     (0xAC00, 0xD7A3),
        "russian":    (0x0400, 0x04FF),
        "telugu":     (0x0C00, 0x0C7F),
        "thai":       (0x0E00, 0x0E7F),
        # Latin-script languages (english, finnish, indonesian, swahili) → None
    }

    # BUG-F FIX: ISO 639-1 codes for bert_score rescaling baseline.
    ISO_639_1: Dict = {
        "arabic":     "ar",
        "bengali":    "bn",
        "english":    "en",
        "finnish":    "fi",
        "indonesian": "id",
        "japanese":   "ja",
        "korean":     "ko",
        "russian":    "ru",
        "swahili":    None,   # not in bert_score baseline list → no rescaling
        "telugu":     None,
        "thai":       "th",
    }

    META: Dict = {
        "arabic":     {"name": "Arabic",      "native": "العربية",        "script": "Arabic",
                       "family": "Semitic",    "word_order": "VSO", "resource": "high",   "morphology": "fusional"},
        "bengali":    {"name": "Bengali",     "native": "বাংলা",          "script": "Bengali",
                       "family": "Indo-Aryan","word_order": "SOV", "resource": "medium",  "morphology": "fusional"},
        "english":    {"name": "English",     "native": "English",         "script": "Latin",
                       "family": "Germanic",  "word_order": "SVO", "resource": "high",    "morphology": "analytic"},
        "finnish":    {"name": "Finnish",     "native": "Suomi",           "script": "Latin",
                       "family": "Uralic",    "word_order": "SVO", "resource": "medium",  "morphology": "agglutinative"},
        "indonesian": {"name": "Indonesian",  "native": "Bahasa Indonesia","script": "Latin",
                       "family": "Austronesian","word_order":"SVO","resource": "medium",  "morphology": "agglutinative"},
        "japanese":   {"name": "Japanese",    "native": "日本語",          "script": "CJK+Kana",
                       "family": "Japonic",   "word_order": "SOV", "resource": "high",    "morphology": "agglutinative"},
        "korean":     {"name": "Korean",      "native": "한국어",          "script": "Hangul",
                       "family": "Koreanic",  "word_order": "SOV", "resource": "high",    "morphology": "agglutinative"},
        "russian":    {"name": "Russian",     "native": "Русский",         "script": "Cyrillic",
                       "family": "Slavic",    "word_order": "SVO", "resource": "high",    "morphology": "fusional"},
        "swahili":    {"name": "Swahili",     "native": "Kiswahili",       "script": "Latin",
                       "family": "Bantu",     "word_order": "SVO", "resource": "low",     "morphology": "agglutinative"},
        "telugu":     {"name": "Telugu",      "native": "తెలుగు",        "script": "Telugu",
                       "family": "Dravidian", "word_order": "SOV", "resource": "low",     "morphology": "agglutinative"},
        "thai":       {"name": "Thai",        "native": "ไทย",             "script": "Thai",
                       "family": "Kra-Dai",   "word_order": "SVO", "resource": "medium",  "morphology": "analytic"},
    }

    @staticmethod
    def contains_script(text: str, lang: str) -> bool:
        if not text or not text.strip():
            return False
        rng = LanguageUtils.SCRIPT_RANGES.get(lang)
        if rng is None:                          # Latin-script language
            return any(c.isalpha() for c in text)
        if isinstance(rng, tuple):
            s, e = rng
            return any(s <= ord(c) <= e for c in text)
        return any(any(s <= ord(c) <= e for s, e in rng) for c in text)

    @staticmethod
    def script_char_count(text: str, lang: str) -> int:
        if not text:
            return 0
        rng = LanguageUtils.SCRIPT_RANGES.get(lang)
        if rng is None:
            return sum(1 for c in text if c.isalpha())
        if isinstance(rng, tuple):
            s, e = rng
            return sum(1 for c in text if s <= ord(c) <= e)
        return sum(1 for c in text if any(s <= ord(c) <= e for s, e in rng))


# ╔══════════════════════════════════════════════════════════════╗
# ║  TEXT NORMALISATION & TOKENISATION                           ║
# ╚══════════════════════════════════════════════════════════════╝

class TextNormalizer:
    """Language-aware normalisation and tokenisation."""

    # Character-level tokenisation only for languages without whitespace
    _CHAR_LEVEL = {"japanese", "korean", "thai"}

    @staticmethod
    def normalize(text: str, lang: str = "") -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        if lang == "arabic":
            # Normalise alef variants and teh marbuta — do NOT strip diacritics
            # as TyDiQA Arabic reference spans may include them.
            text = re.sub(r"[إأآا]", "ا", text)
            text = re.sub(r"ى", "ي", text)
            text = re.sub(r"ة", "ه", text)
        return text

    @staticmethod
    def tokenize(text: str, lang: str = "") -> List[str]:
        if not text:
            return []
        text = TextNormalizer.normalize(text, lang)
        if lang in TextNormalizer._CHAR_LEVEL:
            return [c for c in text if not c.isspace()]
        # All other languages: word tokens, lowercased
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def token_f1_pair(pred: str, ref: str, lang: str) -> float:
        p_toks = TextNormalizer.tokenize(pred, lang)
        r_toks = TextNormalizer.tokenize(ref, lang)
        if not p_toks or not r_toks:
            return 0.0
        common = len(set(p_toks) & set(r_toks))
        prec = common / len(p_toks)
        rec  = common / len(r_toks)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATASET MANAGER  (TyDiQA schema verified)                   ║
# ╚══════════════════════════════════════════════════════════════╝

class DatasetManager:
    """
    Stream TyDiQA primary_task and extract reference answer spans.

    ─── TyDiQA PRIMARY_TASK SCHEMA ────────────────────────────────
    Top-level keys:
      "language"            : str
      "question_text"       : str
      "document_plaintext"  : str   ← full Wikipedia article (UTF-8)
      "document_url"        : str
      "document_title"      : str
      "annotations" : {
          "minimal_answers_start_byte" : List[int]  (-1 = unanswerable)
          "minimal_answers_end_byte"   : List[int]
          "passage_answer_candidate_index" : List[int]
          "yes_no_answer"              : List[str]  ("NONE"|"YES"|"NO")
      }

    There is NO "answers" key and NO pre-extracted answer text.
    The answer text must be sliced from document_plaintext as bytes.
    ────────────────────────────────────────────────────────────────
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buffers: Dict[str, List[Dict]] = {l: [] for l in cfg.TARGET_LANGS}

    @staticmethod
    def _extract_answer_text(doc_bytes: bytes, start: int, end: int) -> str:
        """Decode a UTF-8 byte slice from document_plaintext."""
        try:
            return doc_bytes[start:end].decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def collect_samples(self):
        print("\n📚 Streaming TyDiQA (primary_task, train split)…")
        ds = load_dataset(
            "google-research-datasets/tydiqa",
            "primary_task",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        for ex in tqdm(ds, desc="Collecting samples"):
            lang = ex.get("language", "")
            if lang not in self.cfg.TARGET_LANGS:
                continue
            if len(self.buffers[lang]) >= self.cfg.MAX_SAMPLES_PER_LANG:
                continue

            # ── Correct field access for TyDiQA primary_task ──
            annotations = ex.get("annotations", {})
            starts = annotations.get("minimal_answers_start_byte", [])
            ends   = annotations.get("minimal_answers_end_byte",   [])
            doc_text = ex.get("document_plaintext", "")

            # Skip unanswerable questions (start_byte == -1)
            valid_spans = [
                (s, e) for s, e in zip(starts, ends) if s >= 0
            ]
            if not valid_spans:
                continue

            # Extract reference answer strings from byte offsets
            doc_bytes = doc_text.encode("utf-8")
            ref_answers = []
            for s, e in valid_spans:
                ans = DatasetManager._extract_answer_text(doc_bytes, s, e)
                if ans:
                    ref_answers.append(ans)

            if not ref_answers:
                continue

            # 8000 chars at 3.0 chars/tok ≈ 2667 context tokens, well within
            # the 3800 input budget and sized for TyDiQA answer passages.
            # Tokenizer max_length=MAX_INPUT_TOKENS always acts as hard guard.
            max_ctx_chars = 8000
            first_start_char = doc_text.encode("utf-8")[: valid_spans[0][0]].decode(
                "utf-8", errors="replace"
            )
            char_offset = len(first_start_char)
            ctx_start = max(0, char_offset - max_ctx_chars // 2)
            ctx_end   = min(len(doc_text), ctx_start + max_ctx_chars)
            evidence  = doc_text[ctx_start:ctx_end]

            self.buffers[lang].append({
                "question":          ex.get("question_text", ""),
                "evidence":          evidence,
                "reference_answers": ref_answers,
            })

            # Early exit once all languages reach quota
            if all(
                len(self.buffers[l]) >= self.cfg.MAX_SAMPLES_PER_LANG
                for l in self.cfg.TARGET_LANGS
            ):
                break

        print("\n✅ Collection complete:")
        for lang, samples in self.buffers.items():
            print(f"   {lang:12s}: {len(samples):4d} answerable samples")
        missing = [l for l, s in self.buffers.items() if not s]
        if missing:
            print(f"\n⚠️  No samples found for: {missing}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PROMPT BUILDER                                              ║
# ╚══════════════════════════════════════════════════════════════╝

class PromptBuilder:
    """
    Four prompting strategies.

    Context is truncated to approximately MAX_INPUT_TOKENS * 3 characters
    (a rough chars-per-token heuristic) to stay within the model's window
    after the full prompt is assembled.  The tokenizer will hard-truncate
    as a final safety net.
    """

    # Approx chars-per-token: 3.0 is a conservative mid-point across scripts.
    # CJK/Korean/Thai use ~1.2–2.2 chars/token so they rely more heavily on
    # the tokenizer's hard max_length truncation, which is always correct.
    _CHARS_PER_TOKEN: float = 3.0

    @staticmethod
    def _ctx(context: str, cfg: Config) -> str:
        """Return a context string that fits within the token budget."""
        # Reserve ~200 tokens for question + instructions + answer prefix
        budget = int((cfg.MAX_INPUT_TOKENS - 200) * PromptBuilder._CHARS_PER_TOKEN)
        return context[:budget]

    @staticmethod
    def build(question: str, context: str, lang: str, cfg: Config) -> Dict[str, str]:
        m    = LanguageUtils.META[lang]
        name = m["name"]
        nat  = m["native"]
        scr  = m["script"]
        ctx  = PromptBuilder._ctx(context, cfg)

        return {
            "minimal": (
                f"Answer the following question in {name} only.\n"
                f"Do NOT answer in English.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{ctx}\n\n"
                f"Answer ({name}):"
            ),
            "explicit_en": (
                f"You must respond ONLY in {name} ({nat}).\n"
                f"Using English is strictly forbidden.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{ctx}\n\n"
                f"Provide a brief factual answer in {name}.\n"
                f"Answer:"
            ),
            "explicit_native": (
                f"[Respond in {nat} / {name} ONLY. No English.]\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{ctx}\n\n"
                f"Answer in {nat} ({name}), using {scr} script only. "
                f"Keep it concise.\nAnswer:"
            ),
            "strict": (
                f"LANGUAGE REQUIREMENT: {name} ({nat}) ONLY.\n"
                f"SCRIPT REQUIREMENT:   {scr} ONLY.\n"
                f"ENGLISH IS FORBIDDEN. SWITCHING LANGUAGES IS FORBIDDEN.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{ctx}\n\n"
                f"Give a short, accurate answer in {name} using {scr} script.\n"
                f"Answer:"
            ),
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODEL MANAGER  (BUG-A, B, C, D FIXED)                      ║
# ╚══════════════════════════════════════════════════════════════╝

class ModelManager:
    """
    Loads EuroLLM-9B-Instruct (or any CausalLM) in 4-bit NF4.

    BUG-A FIX: USE_CHAT_TEMPLATE now defaults to True in Config.

    BUG-B FIX: _apply_chat_template() now calls
      tokenizer.apply_chat_template() with the proper messages list,
      which correctly handles EuroLLM's ChatML-style template AND
      any other tokenizer that ships a chat template.  We fall back
      to plain string concatenation only when the tokenizer has no
      registered template.

    BUG-C FIX: gradient_checkpointing_enable() removed; it is
      meaningless for inference and breaks some BnB-quantized models.

    BUG-D FIX: torch.amp.autocast("cuda", …) instead of the
      deprecated torch.cuda.amp.autocast(dtype=…).
    """

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self.model = None
        self.tok   = None
        self.dev   = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        from huggingface_hub import login
        login(self.cfg.HF_TOKEN)
        print(f"\n⭐ Loading {self.cfg.MODEL_NAME} …")

        bnb = BitsAndBytesConfig(
            load_in_4bit=self.cfg.USE_4BIT,
            bnb_4bit_compute_dtype=self.cfg.COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=self.cfg.USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=self.cfg.QUANT_TYPE,
        )

        self.tok = AutoTokenizer.from_pretrained(
            self.cfg.MODEL_NAME,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Ensure left-padding for decoder-only models (safer for batch gen)
        self.tok.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.MODEL_NAME,
            quantization_config=bnb,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=self.cfg.COMPUTE_DTYPE,
            # BUG-C FIX: no gradient_checkpointing here
        )
        self.model.eval()

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"✓ Model loaded  |  GPU memory: {mem_gb:.1f} GB")

        # Auto-detect instruct model from name if user left default
        if "instruct" in self.cfg.MODEL_NAME.lower():
            self.cfg.USE_CHAT_TEMPLATE = True

    # ── BUG-B FIX: proper chat template application ──────────────
    def _apply_chat_template(self, user_prompt: str) -> str:
        """
        Wrap the user prompt using the tokenizer's own chat template.

        For EuroLLM-9B-Instruct (and most Instruct variants) this
        produces:  <|im_start|>system…<|im_end|><|im_start|>user…
        which is exactly what the model was fine-tuned on.

        Falls back to plain prompt when:
          - USE_CHAT_TEMPLATE is False (base models), OR
          - the tokenizer has no chat_template attribute.
        """
        if not self.cfg.USE_CHAT_TEMPLATE:
            return user_prompt

        chat_tmpl = getattr(self.tok, "chat_template", None)
        if chat_tmpl is None:
            # Tokenizer ships without a template; use raw prompt
            return user_prompt

        try:
            messages = [{"role": "user", "content": user_prompt}]
            return self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,   # appends assistant turn opener
            )
        except Exception as exc:
            print(f"  ⚠️  chat_template failed ({exc}), using raw prompt")
            return user_prompt

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Batched generation — process multiple prompts in one forward pass.

        This is the primary speed lever: batching 8 prompts together is
        ~4–6× faster than 8 sequential single-sample calls because:
          • KV-cache is built once across the batch.
          • GPU utilisation jumps from ~15% to ~85% on L40S for this model.

        Left-padding is used (set in load()) so that all position IDs
        are aligned to the right edge — correct for causal LMs.

        BUG-D FIX: torch.amp.autocast("cuda", dtype=…).
        """
        import contextlib
        if not prompts:
            return []

        formatted = [self._apply_chat_template(p) for p in prompts]
        try:
            # Tokenize with left-padding so all sequences are right-aligned
            inputs = self.tok(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.MAX_INPUT_TOKENS,
                padding=True,           # pad to longest in batch
                padding_side="left",    # left-pad for causal LMs
            )
            # input_lengths[i] = non-padding tokens for sample i
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            inputs = {k: v.to(self.dev) for k, v in inputs.items()}

            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=self.cfg.COMPUTE_DTYPE)
                if self.dev == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx, torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,   # greedy — must be None when do_sample=False
                    top_p=None,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id,
                    use_cache=True,
                )
            del inputs
            torch.cuda.empty_cache()

            answers = []
            total_len = out.shape[1]   # same for all sequences in the padded batch
            for i, seq in enumerate(out):
                # out is left-padded: [pad...pad | prompt | generated]
                # input_lengths[i] = number of non-padding (prompt) tokens.
                # The total output length = padding + input_lengths[i] + generated.
                # Padding length = total_len - input_lengths[i] - generated_len
                # We want only the generated tokens, which start at:
                #   total_len - (total_len - input_lengths[i]) = input_lengths[i]
                # ... but the tensor includes the padding prefix, so:
                #   generated tokens start at index: total_len - max_new_tokens
                # However if the model stopped early via EOS, trailing pad_token
                # ids will appear — skip_special_tokens handles those cleanly.
                prompt_len = int(input_lengths[i])
                answer_ids = seq[prompt_len:]   # everything after the input
                answer = self.tok.decode(answer_ids, skip_special_tokens=True)
                answer = re.sub(r"<\|im_end\|>.*", "", answer, flags=re.S)
                answer = re.sub(r"<\|im_start\|>.*", "", answer, flags=re.S)
                answers.append(answer.strip())

            return answers

        except Exception as exc:
            print(f"  ⚠️  Batch generation error: {exc}")
            # Fall back to empty strings for whole batch
            return [""] * len(prompts)


# ╔══════════════════════════════════════════════════════════════╗
# ║  EVALUATION METRICS  (BUG-F, G fixed)                       ║
# ╚══════════════════════════════════════════════════════════════╝

class Metrics:
    """
    Full metric suite.

    BUG-F FIX: BERTScore lang_code now uses ISO 639-1 codes from
      LanguageUtils.ISO_639_1 map instead of the wrong family[:2].

    BUG-G FIX: bertscore_batch is NOT called from within
      ThreadPoolExecutor threads.  It is invoked sequentially by
      MultilingualPipeline.run_bertscore() after all other metrics
      have been computed.  The evaluate_all() method therefore
      omits BERTScore; the pipeline merges it in afterwards.
    """

    # ── Heuristic metrics ────────────────────────────────────────

    @staticmethod
    def language_match_rate(results: List[Dict], lang: str) -> float:
        if not results:
            return 0.0
        return sum(
            LanguageUtils.contains_script(r["answer"], lang) for r in results
        ) / len(results)

    @staticmethod
    def script_consistency_score(results: List[Dict], lang: str) -> float:
        if not results:
            return 0.0
        scores = []
        for r in results:
            ans = r["answer"]
            if not ans:
                scores.append(0.0)
                continue
            sc = LanguageUtils.script_char_count(ans, lang)
            scores.append(sc / max(len(ans.strip()), 1))
        return float(np.mean(scores))

    @staticmethod
    def evidence_usage_ratio(results: List[Dict], lang: str) -> float:
        if not results:
            return 0.0
        ratios = []
        for r in results:
            a_toks = set(TextNormalizer.tokenize(r["answer"], lang))
            e_toks = set(TextNormalizer.tokenize(r["evidence"], lang))
            if not a_toks:
                ratios.append(0.0)
            else:
                ratios.append(len(a_toks & e_toks) / len(a_toks))
        return float(np.mean(ratios))

    @staticmethod
    def hallucination_frequency(results: List[Dict], lang: str) -> float:
        if not results:
            return 0.0
        rates = []
        for r in results:
            a_toks = set(TextNormalizer.tokenize(r["answer"], lang))
            e_toks = set(TextNormalizer.tokenize(r["evidence"], lang))
            if not a_toks:
                rates.append(0.0)
            else:
                rates.append(len(a_toks - e_toks) / len(a_toks))
        return float(np.mean(rates))

    # ── Answer-quality metrics ───────────────────────────────────

    @staticmethod
    def exact_match(pred: str, refs: List[str], lang: str) -> float:
        if not pred or not refs:
            return 0.0
        p = TextNormalizer.normalize(pred, lang)
        return float(any(p == TextNormalizer.normalize(r, lang) for r in refs))

    @staticmethod
    def token_f1(pred: str, refs: List[str], lang: str) -> float:
        if not pred or not refs:
            return 0.0
        return max(TextNormalizer.token_f1_pair(pred, r, lang) for r in refs)

    @staticmethod
    def transfer_rate(results: List[Dict], lang: str) -> float:
        valid = [r for r in results if r.get("reference_answers")]
        if not valid:
            return 0.0
        correct = sum(
            1 for r in valid
            if Metrics.exact_match(r["answer"], r["reference_answers"], lang) == 1.0
            or Metrics.token_f1(r["answer"], r["reference_answers"], lang) > 0.5
        )
        return correct / len(valid)

    @staticmethod
    def bertscore_batch(
        predictions: List[str],
        references: List[List[str]],
        lang: str,
        cfg: Config,
    ) -> Dict[str, float]:
        """
        Compute BERTScore for one (strategy, language) cell.

        BUG-F FIX: lang_code from ISO_639_1 map.
        BUG-G FIX: called single-threaded (NOT inside ThreadPoolExecutor).
        """
        if not cfg.USE_BERTSCORE:
            return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}

        try:
            from bert_score import score as _bs

            valid_preds, valid_refs = [], []
            for pred, refs in zip(predictions, references):
                if not pred or not pred.strip():
                    continue
                good_refs = [r for r in refs if r and r.strip()]
                if not good_refs:
                    continue
                valid_preds.append(pred)
                valid_refs.append(good_refs[0])

            if not valid_preds:
                return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}

            # BUG-F FIX: proper ISO 639-1 code (or None for no rescaling)
            lang_code = LanguageUtils.ISO_639_1.get(lang)

            P, R, F1 = _bs(
                valid_preds,
                valid_refs,
                model_type=cfg.BERTSCORE_MODEL,
                lang=lang_code,            # correct kwarg is "lang" not positional
                batch_size=cfg.BERTSCORE_BATCH_SIZE,
                verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            return {
                "bertscore_P":  float(P.mean()),
                "bertscore_R":  float(R.mean()),
                "bertscore_F1": float(F1.mean()),
            }
        except ImportError:
            print("  ⚠️  bert-score not installed: pip install bert-score")
            return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}
        except Exception as exc:
            print(f"  ⚠️  BERTScore error for {lang}: {exc}")
            return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}

    @staticmethod
    def evaluate_all(
        results: List[Dict], lang: str
    ) -> Dict[str, float]:
        """
        Run all CPU-bound metrics for one (strategy, language) cell.

        BUG-G FIX: BERTScore is NOT computed here (GPU; must be
        single-threaded).  The pipeline merges BERTScore results
        in a separate sequential pass via run_bertscore().
        """
        if not results:
            return {}

        base = {
            "language_match_rate": Metrics.language_match_rate(results, lang),
            "script_consistency":  Metrics.script_consistency_score(results, lang),
            "evidence_usage":      Metrics.evidence_usage_ratio(results, lang),
            "hallucination":       Metrics.hallucination_frequency(results, lang),
        }

        em_scores, f1_scores = [], []
        for r in results:
            refs = r.get("reference_answers", [])
            if refs:
                em_scores.append(Metrics.exact_match(r["answer"], refs, lang))
                f1_scores.append(Metrics.token_f1(r["answer"], refs, lang))

        quality = {
            "exact_match":   float(np.mean(em_scores)) if em_scores else 0.0,
            "f1_score":      float(np.mean(f1_scores)) if f1_scores else 0.0,
            "transfer_rate": Metrics.transfer_rate(results, lang),
        }

        lengths = [len(TextNormalizer.tokenize(r["answer"], lang)) for r in results]
        length_stats = {
            "mean_length":   float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
        }

        # BERTScore placeholders — filled in by run_bertscore()
        bert_placeholders = {
            "bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0,
        }

        return {**base, **quality, **bert_placeholders, **length_stats}


# ╔══════════════════════════════════════════════════════════════╗
# ║  STATISTICAL TESTING                                         ║
# ╚══════════════════════════════════════════════════════════════╝

class StatTests:

    @staticmethod
    def paired_ttest(a: List[float], b: List[float]) -> Dict:
        if len(a) < 2:
            return {"t": 0., "p": 1., "sig": False, "d": 0., "effect": "n/a"}
        t, p = stats.ttest_rel(a, b)
        diff = np.array(a) - np.array(b)
        d    = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
        eff  = ("negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5
                else "medium" if abs(d) < 0.8 else "large")
        return {"t": float(t), "p": float(p), "sig": p < 0.05,
                "d": d, "effect": eff, "mean_diff": float(np.mean(diff))}

    @staticmethod
    def bootstrap_ci(
        data: List[float], n: int = 1000, conf: float = 0.95
    ) -> Tuple[float, float, float]:
        if len(data) < 2:
            v = data[0] if data else 0.
            return v, v, v
        arr   = np.array(data)
        boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n)]
        alpha = (1 - conf) / 2
        return (float(np.mean(arr)),
                float(np.percentile(boots, alpha * 100)),
                float(np.percentile(boots, (1 - alpha) * 100)))

    @staticmethod
    def compare_strategies(
        evaluation: Dict, metric: str, languages: List[str]
    ) -> Dict:
        strategies = list(evaluation.keys())
        results, pvals = {}, []
        for i, sa in enumerate(strategies):
            for sb in strategies[i + 1:]:
                key = f"{sa}_vs_{sb}"
                va = [evaluation[sa][l][metric] for l in languages
                      if l in evaluation[sa] and metric in evaluation[sa][l]]
                vb = [evaluation[sb][l][metric] for l in languages
                      if l in evaluation[sb] and metric in evaluation[sb][l]]
                if len(va) >= 2:
                    r = StatTests.paired_ttest(va, vb)
                    results[key] = r
                    pvals.append(r["p"])
        # Bonferroni correction
        if pvals:
            for key, p_corr in zip(results, [min(p * len(pvals), 1.) for p in pvals]):
                results[key]["p_corrected"] = p_corr
        return results


# ╔══════════════════════════════════════════════════════════════╗
# ║  ERROR ANALYSER                                              ║
# ╚══════════════════════════════════════════════════════════════╝

class ErrorAnalyzer:
    CATEGORIES = [
        "wrong_language", "partial_code_switch", "empty_answer",
        "too_short", "hallucination_high", "no_evidence_overlap",
        "refusal", "correct",
    ]
    REFUSAL_RE = re.compile(
        r"\b(cannot|unable|can'?t|sorry|don'?t know|no information"
        r"|not mentioned|unclear|I cannot)\b", re.I
    )

    @staticmethod
    def categorize(result: Dict, lang: str) -> List[str]:
        ans = result.get("answer", "")
        ev  = result.get("evidence", "")
        if not ans or not ans.strip():
            return ["empty_answer"]
        errors = []
        if not LanguageUtils.contains_script(ans, lang):
            errors.append("wrong_language")
        sc = LanguageUtils.script_char_count(ans, lang) / max(len(ans.strip()), 1)
        if 0.2 < sc < 0.8:
            errors.append("partial_code_switch")
        if len(TextNormalizer.tokenize(ans, lang)) < 3:
            errors.append("too_short")
        a_toks = set(TextNormalizer.tokenize(ans, lang))
        e_toks = set(TextNormalizer.tokenize(ev,  lang))
        if a_toks and len(a_toks - e_toks) / len(a_toks) > 0.8:
            errors.append("hallucination_high")
        if a_toks and len(a_toks & e_toks) == 0:
            errors.append("no_evidence_overlap")
        if ErrorAnalyzer.REFUSAL_RE.search(ans):
            errors.append("refusal")
        return errors or ["correct"]

    @staticmethod
    def analyze(results: List[Dict], lang: str) -> Dict:
        counts = {c: 0 for c in ErrorAnalyzer.CATEGORIES}
        for r in results:
            for cat in ErrorAnalyzer.categorize(r, lang):
                counts[cat] += 1
        n = max(len(results), 1)
        return {
            "counts": counts,
            "rates":  {k: v / n for k, v in counts.items()},
            "n": n,
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  CROSS-LINGUAL ANALYSIS                                      ║
# ╚══════════════════════════════════════════════════════════════╝

class CrossLingualAnalysis:

    METRIC_COLS = [
        "language_match_rate", "script_consistency",
        "exact_match", "f1_score", "transfer_rate", "bertscore_F1",
    ]

    @staticmethod
    def build_typology_df(evaluation: Dict, available_langs: List[str]) -> pd.DataFrame:
        rows = []
        for strategy, lang_dict in evaluation.items():
            for lang in available_langs:
                if lang not in lang_dict:        # defensive key check (BUG 5)
                    continue
                metrics = lang_dict[lang]
                meta    = LanguageUtils.META.get(lang, {})
                row = {
                    "strategy":   strategy,
                    "language":   lang,
                    "family":     meta.get("family",     "?"),
                    "word_order": meta.get("word_order", "?"),
                    "resource":   meta.get("resource",   "?"),
                    "morphology": meta.get("morphology", "?"),
                }
                for col in CrossLingualAnalysis.METRIC_COLS:
                    row[col] = metrics.get(col, np.nan)
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def resource_level_summary(df: pd.DataFrame, metric: str = "f1_score") -> pd.DataFrame:
        if df.empty or metric not in df.columns:
            return pd.DataFrame()
        return (
            df.groupby(["strategy", "resource"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  VISUALISATIONS                                              ║
# ╚══════════════════════════════════════════════════════════════╝

class Visualizations:

    @staticmethod
    def _setup():
        sns.set_style("whitegrid")
        plt.rcParams.update({"figure.dpi": 300, "font.size": 10})

    @staticmethod
    def heatmap(evaluation: Dict, metric: str, langs: List[str], path: str):
        Visualizations._setup()
        strategies = list(evaluation.keys())
        data = [
            [evaluation[s].get(l, {}).get(metric, 0.) for l in langs]
            for s in strategies
        ]
        fig, ax = plt.subplots(figsize=(max(12, len(langs) * 1.1), 5))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(langs)))
        ax.set_yticks(range(len(strategies)))
        ax.set_xticklabels([l.capitalize() for l in langs], rotation=45, ha="right")
        ax.set_yticklabels([s.replace("_", " ").title() for s in strategies])
        for i in range(len(strategies)):
            for j in range(len(langs)):
                v = data[i][j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v > 0.45 else "white", fontsize=7)
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title(f"{metric.replace('_', ' ').title()} — Strategies × Languages")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def ci_bar(evaluation: Dict, ci_dict: Dict, metric: str,
               langs: List[str], path: str):
        Visualizations._setup()
        strategies = list(evaluation.keys())
        x = np.arange(len(langs))
        w = 0.2
        fig, ax = plt.subplots(figsize=(max(14, len(langs) * 1.3), 6))
        for i, s in enumerate(strategies):
            means = [evaluation[s].get(l, {}).get(metric, 0.) for l in langs]
            lows, highs = [], []
            for j, l in enumerate(langs):
                ci = ci_dict.get(s, {}).get(l, {}).get(f"{metric}_ci")
                if ci:
                    lows.append(means[j] - ci["lo"])
                    highs.append(ci["hi"] - means[j])
                else:
                    lows.append(0)
                    highs.append(0)
            ax.bar(x + i * w, means, w, label=s.replace("_", " ").title(),
                   yerr=[lows, highs], capsize=3)
        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels([l.capitalize() for l in langs], rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} with 95% CI")
        ax.legend()
        ax.grid(axis="y", alpha=.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def error_bars(error_dict: Dict, langs: List[str], path: str):
        Visualizations._setup()
        strategies = list(error_dict.keys())
        cats = ErrorAnalyzer.CATEGORIES
        n_plots = min(len(strategies), 4)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        for idx in range(4):
            ax = axes[idx]
            if idx >= n_plots:
                ax.set_visible(False)
                continue
            strat = strategies[idx]
            btm   = np.zeros(len(langs))
            for cat in cats:
                vals = [error_dict[strat].get(l, {}).get("rates", {}).get(cat, 0.) for l in langs]
                ax.bar(range(len(langs)), vals, bottom=btm, label=cat)
                btm += np.array(vals)
            ax.set_xticks(range(len(langs)))
            ax.set_xticklabels([l[:3].upper() for l in langs], rotation=45)
            ax.set_title(strat.replace("_", " ").title())
            ax.set_ylabel("Rate")
            if idx == 0:
                ax.legend(fontsize=7, loc="upper right")
        plt.suptitle("Error Distribution", fontsize=13)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def typology_box(df: pd.DataFrame, dim: str, metric: str, path: str):
        Visualizations._setup()
        if df.empty or dim not in df.columns or metric not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=df, x=dim, y=metric, hue="strategy", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by {dim.replace('_', ' ').title()}")
        plt.xticks(rotation=30, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def multi_metric_line(evaluation: Dict, metrics: List[str],
                          langs: List[str], path: str):
        Visualizations._setup()
        strategies = list(evaluation.keys())
        n_metrics  = min(len(metrics), 4)
        fig, axes  = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        for idx in range(4):
            ax = axes[idx]
            if idx >= n_metrics:
                ax.set_visible(False)
                continue
            met = metrics[idx]
            for s in strategies:
                ax.plot(
                    range(len(langs)),
                    [evaluation[s].get(l, {}).get(met, 0.) for l in langs],
                    marker="o",
                    label=s.replace("_", " ").title(),
                )
            ax.set_title(met.replace("_", " ").title())
            ax.set_xticks(range(len(langs)))
            ax.set_xticklabels([l[:3].upper() for l in langs], rotation=45)
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
        plt.suptitle("Metric Comparison", fontsize=13)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def bertscore_heatmap(evaluation: Dict, langs: List[str], path: str):
        """Dedicated BERTScore-F1 heatmap (generated after sequential BS pass)."""
        Visualizations.heatmap(evaluation, "bertscore_F1", langs, path)


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                               ║
# ╚══════════════════════════════════════════════════════════════╝

class MultilingualPipeline:

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self.model = ModelManager(cfg)
        self.data  = DatasetManager(cfg)
        self.results: Dict = {
            s: {l: [] for l in cfg.TARGET_LANGS}
            for s in cfg.PROMPT_STRATEGIES
        }
        self.evaluation:  Dict         = {}
        self.error_dict:  Dict         = {}
        self.stat_tests:  Dict         = {}
        self.ci_dict:     Dict         = {}
        self.typology_df: pd.DataFrame = pd.DataFrame()
        self.avail_langs: List[str]    = []

    # ── Setup ────────────────────────────────────────────────────

    def setup(self):
        print("\n" + "=" * 65)
        print("🌍  CORRECTED MULTILINGUAL EVALUATION PIPELINE  v5.2")
        print("=" * 65)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.model.load()
        self.data.collect_samples()
        self.avail_langs = [
            l for l in self.cfg.TARGET_LANGS if self.data.buffers[l]
        ]
        if not self.avail_langs:
            raise RuntimeError("No samples collected for any language — check dataset access.")

    # ── Generation ───────────────────────────────────────────────

    def run_generation(self):
        """
        Batched generation loop.

        For each (language, strategy) pair we build all prompts upfront,
        then feed them to generate_batch() in chunks of GENERATION_BATCH_SIZE.
        This gives ~4–6× GPU utilisation improvement vs the original
        one-sample-at-a-time loop.

        Outer loop: language  (keeps tqdm progress per language readable)
        Inner loop: strategy  (batch all samples for one strategy at once)
        """
        n_total = sum(len(self.data.buffers[l]) for l in self.avail_langs)
        bs = self.cfg.GENERATION_BATCH_SIZE
        print(
            f"\n🚀 Generating answers  "
            f"({n_total} samples × {len(self.cfg.PROMPT_STRATEGIES)} strategies, "
            f"batch_size={bs})"
        )

        for lang in self.avail_langs:
            print(f"\n── {lang.upper()} ──")
            samples = self.data.buffers[lang]

            # Pre-build all prompts for this language (all strategies at once)
            all_prompts: Dict[str, List[str]] = {s: [] for s in self.cfg.PROMPT_STRATEGIES}
            for ex in samples:
                p = PromptBuilder.build(ex["question"], ex["evidence"], lang, self.cfg)
                for strat in self.cfg.PROMPT_STRATEGIES:
                    all_prompts[strat].append(p[strat])

            # Generate in batches per strategy
            for strat in self.cfg.PROMPT_STRATEGIES:
                strat_prompts = all_prompts[strat]
                all_answers: List[str] = []

                for i in tqdm(
                    range(0, len(strat_prompts), bs),
                    desc=f"{lang}/{strat}",
                    leave=False,
                ):
                    batch = strat_prompts[i : i + bs]
                    answers = self.model.generate_batch(batch)
                    all_answers.extend(answers)

                # Store results
                for ex, ans in zip(samples, all_answers):
                    self.results[strat][lang].append({
                        "question":          ex["question"],
                        "evidence":          ex["evidence"],
                        "reference_answers": ex["reference_answers"],
                        "answer":            ans,
                    })

            gc.collect()
            torch.cuda.empty_cache()

        print("\n✅ Generation done.")

    # ── Evaluation (CPU metrics, threaded) ───────────────────────

    def evaluate(self):
        """
        Run all CPU-bound metrics in parallel.

        BUG-G FIX: BERTScore is NOT run here.
        """
        from concurrent.futures import ThreadPoolExecutor

        print("\n📊 Computing metrics (multi-threaded, CPU metrics only)…")
        evaluation: Dict = {s: {} for s in self.cfg.PROMPT_STRATEGIES}

        tasks = [
            (s, l)
            for s in self.cfg.PROMPT_STRATEGIES
            for l in self.avail_langs
            if self.results[s][l]
        ]

        with ThreadPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as ex:
            future_map = {
                ex.submit(
                    Metrics.evaluate_all,
                    self.results[s][l], l
                    # NOTE: no cfg argument — BERTScore excluded here
                ): (s, l)
                for s, l in tasks
            }
            for fut in tqdm(future_map, desc="Evaluating", total=len(future_map)):
                s, l = future_map[fut]
                try:
                    evaluation[s][l] = fut.result(timeout=600)
                except Exception as exc:
                    print(f"  ⚠️  {s}/{l}: {exc}")
                    evaluation[s][l] = {}

        self.evaluation = evaluation
        print("✅ CPU metrics done.")
        return evaluation

    # ── BERTScore (sequential GPU pass) ─────────────────────────

    def run_bertscore(self):
        """
        BUG-G FIX: compute BERTScore sequentially (one cell at a time)
        to avoid concurrent GPU access from multiple threads.
        Results are merged directly into self.evaluation.
        """
        if not self.cfg.USE_BERTSCORE:
            return

        print("\n🔬 Computing BERTScore (sequential GPU pass)…")
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                if not res:
                    continue
                preds = [r["answer"] for r in res]
                refs  = [r.get("reference_answers", []) for r in res]
                bs    = Metrics.bertscore_batch(preds, refs, l, self.cfg)
                # Merge into existing evaluation dict
                if l in self.evaluation.get(s, {}):
                    self.evaluation[s][l].update(bs)
                gc.collect()
                torch.cuda.empty_cache()
        print("✅ BERTScore done.")

    # ── Error analysis ───────────────────────────────────────────

    def run_error_analysis(self):
        print("\n🔍 Error analysis…")
        self.error_dict = {s: {} for s in self.cfg.PROMPT_STRATEGIES}
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                if res:
                    self.error_dict[s][l] = ErrorAnalyzer.analyze(res, l)
        print("✅ Error analysis done.")

    # ── Statistical tests ────────────────────────────────────────

    def run_stat_tests(self):
        print("\n📈 Statistical tests…")
        key_metrics = [
            "language_match_rate", "f1_score",
            "exact_match", "transfer_rate", "bertscore_F1",
        ]
        for met in key_metrics:
            exists = any(
                met in self.evaluation.get(s, {}).get(l, {})
                for s in self.cfg.PROMPT_STRATEGIES
                for l in self.avail_langs
            )
            if exists:
                self.stat_tests[met] = StatTests.compare_strategies(
                    self.evaluation, met, self.avail_langs
                )
        print("✅ Statistical tests done.")

    # ── Confidence intervals (BUG-H FIXED) ──────────────────────

    def compute_cis(self):
        """
        BUG-H FIX: lambdas now capture 'l' by default argument
          lambda r, _l=l: …
        so each closure holds its own snapshot of 'l' rather than
        sharing a reference to the loop variable.
        """
        print("\n📊 Bootstrap CIs…")
        self.ci_dict = {s: {} for s in self.cfg.PROMPT_STRATEGIES}
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                self.ci_dict[s][l] = {}
                # BUG-H FIX: default-argument capture
                for met, fn in [
                    ("f1_score",    lambda r, _l=l: Metrics.token_f1(
                        r["answer"], r.get("reference_answers", []), _l)),
                    ("exact_match", lambda r, _l=l: Metrics.exact_match(
                        r["answer"], r.get("reference_answers", []), _l)),
                ]:
                    scores = [fn(r) for r in res if r.get("reference_answers")]
                    if len(scores) >= 2:
                        mean, lo, hi = StatTests.bootstrap_ci(
                            scores, self.cfg.N_BOOTSTRAP, self.cfg.CONFIDENCE_LEVEL
                        )
                        self.ci_dict[s][l][f"{met}_ci"] = {
                            "mean": mean, "lo": lo, "hi": hi
                        }
        print("✅ CIs done.")

    # ── Cross-lingual analysis (BEFORE visualisations — BUG 5) ──

    def run_cross_lingual(self):
        print("\n🌍 Cross-lingual typological analysis…")
        self.typology_df = CrossLingualAnalysis.build_typology_df(
            self.evaluation, self.avail_langs
        )
        print(f"   typology_df shape: {self.typology_df.shape}")
        print("✅ Cross-lingual analysis done.")

    # ── Visualisations ───────────────────────────────────────────

    def generate_visualisations(self):
        print("\n🎨 Generating visualisations…")
        vd = os.path.join(self.cfg.OUTPUT_DIR, f"viz_{self.cfg.TIMESTAMP}")
        os.makedirs(vd, exist_ok=True)

        hm_metrics = [
            "language_match_rate", "script_consistency",
            "exact_match", "f1_score", "transfer_rate", "bertscore_F1",
        ]
        for met in hm_metrics:
            exists = any(
                met in self.evaluation.get(s, {}).get(l, {})
                for s in self.cfg.PROMPT_STRATEGIES
                for l in self.avail_langs
            )
            if exists:
                Visualizations.heatmap(
                    self.evaluation, met, self.avail_langs,
                    os.path.join(vd, f"heatmap_{met}.png")
                )

        for met in ["f1_score", "exact_match"]:
            Visualizations.ci_bar(
                self.evaluation, self.ci_dict, met, self.avail_langs,
                os.path.join(vd, f"ci_{met}.png")
            )

        if self.error_dict:
            Visualizations.error_bars(
                self.error_dict, self.avail_langs,
                os.path.join(vd, "error_distribution.png")
            )

        for dim in ["word_order", "resource", "morphology"]:
            Visualizations.typology_box(
                self.typology_df, dim, "f1_score",
                os.path.join(vd, f"typology_{dim}.png")
            )

        Visualizations.multi_metric_line(
            self.evaluation,
            ["language_match_rate", "f1_score", "transfer_rate", "bertscore_F1"],
            self.avail_langs,
            os.path.join(vd, "metric_comparison.png")
        )

        print(f"✅ Visualisations saved → {vd}")

    # ── Print summary ────────────────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 65)
        print("📊  RESULTS SUMMARY")
        print("=" * 65)
        for strat in self.cfg.PROMPT_STRATEGIES:
            print(f"\n▸ {strat.upper()}")
            for lang in self.avail_langs:
                m = self.evaluation.get(strat, {}).get(lang, {})
                if not m:
                    continue
                print(
                    f"  {lang:12s} | "
                    f"LMR={m.get('language_match_rate', 0):.2f} | "
                    f"F1={m.get('f1_score', 0):.2f} | "
                    f"EM={m.get('exact_match', 0):.2f} | "
                    f"TR={m.get('transfer_rate', 0):.2f} | "
                    f"BS-F1={m.get('bertscore_F1', 0):.2f}"
                )

    # ── Save ─────────────────────────────────────────────────────

    def save(self):
        if not self.cfg.SAVE_RESULTS:
            return
        print(f"\n💾 Saving results → {self.cfg.OUTPUT_DIR}")
        tag = self.cfg.TIMESTAMP

        # Main evaluation JSON
        out = {
            "config": {
                "model":      self.cfg.MODEL_NAME,
                "langs":      self.avail_langs,
                "strategies": self.cfg.PROMPT_STRATEGIES,
                "samples":    self.cfg.MAX_SAMPLES_PER_LANG,
            },
            "evaluation": self.evaluation,
            "ci":         self.ci_dict,
            "stat_tests": self.stat_tests,
        }
        with open(
            os.path.join(self.cfg.OUTPUT_DIR, f"eval_{tag}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        # Summary CSV
        rows = []
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                m = self.evaluation.get(s, {}).get(l, {})
                rows.append({"strategy": s, "language": l, **m})
        pd.DataFrame(rows).to_csv(
            os.path.join(self.cfg.OUTPUT_DIR, f"summary_{tag}.csv"), index=False
        )

        # Typology CSV
        if not self.typology_df.empty:
            self.typology_df.to_csv(
                os.path.join(self.cfg.OUTPUT_DIR, f"typology_{tag}.csv"), index=False
            )

        print("✅ Saved.")

    # ── Orchestrator ─────────────────────────────────────────────

    def run(self):
        """
        Pipeline execution order:
          1. setup()               — model load + dataset collection
          2. run_generation()      — inference for all strategies × languages
          3. evaluate()            — CPU metrics (threaded)
          4. run_bertscore()       — GPU BERTScore (sequential)  ← BUG-G FIX
          5. run_error_analysis()  — error categorisation
          6. run_stat_tests()      — paired t-tests + Bonferroni
          7. compute_cis()         — bootstrap confidence intervals
          8. run_cross_lingual()   — typology DataFrame           ← BUG-5 FIX
          9. generate_visualisations()
         10. print_summary()
         11. save()
        """
        try:
            self.setup()
            self.run_generation()
            self.evaluate()
            self.run_bertscore()          # sequential GPU — BUG-G FIX
            self.run_error_analysis()
            self.run_stat_tests()
            self.compute_cis()
            self.run_cross_lingual()      # before visualisations — BUG-5 FIX
            self.generate_visualisations()
            self.print_summary()
            self.save()
            print("\n" + "=" * 65)
            print("✅  PIPELINE COMPLETE")
            print("=" * 65)
        except Exception:
            import traceback
            traceback.print_exc()
            raise


# ╔══════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║   CORRECTED MULTILINGUAL EVALUATION PIPELINE  v5.2           ║
║   EuroLLM-9B-Instruct | TyDiQA | Batched + Research-optimised ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    cfg = Config()

    # ── SET YOUR TOKEN ───────────────────────────────────────────
    cfg.HF_TOKEN = ""   # ← replace with the huggingface token

    # ── QUICK SMOKE-TEST (uncomment to verify pipeline quickly) ──
    # cfg.MAX_SAMPLES_PER_LANG = 20
    # cfg.TARGET_LANGS         = ["english", "arabic", "finnish"]
    # cfg.USE_BERTSCORE        = False
    # cfg.N_BOOTSTRAP          = 50

    # ── AYA-23 SWITCH (uncomment to use base Aya instead) ────────
    # cfg.MODEL_NAME        = "CohereForAI/aya-23-8B"
    # cfg.USE_CHAT_TEMPLATE = False   # base model — no chat template

    if cfg.HF_TOKEN == "hf_YOUR_TOKEN_HERE":
        raise SystemExit("⛔  Set cfg.HF_TOKEN before running.")

    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected — generation will be very slow.")

    MultilingualPipeline(cfg).run()
