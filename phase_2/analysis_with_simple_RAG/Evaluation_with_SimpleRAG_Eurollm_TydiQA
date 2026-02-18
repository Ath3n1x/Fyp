# =============================================================================
# RAG-ONLY Multilingual Evaluation Pipeline v5.7 — Hybrid Chunk Retrieval
# =============================================================================

import gc
import json
import os
import re
import warnings
import contextlib
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")
# Author: Swathi
# =============================================================================
# CONFIG
# =============================================================================
class Config:
    HF_TOKEN: str = " "  #<-- ENTER YOUR HUGGINGFACE TOKEN
    MODEL_NAME: str = "utter-project/EuroLLM-9B-Instruct"
    USE_CHAT_TEMPLATE: bool = True

    RETRIEVER_NAME: str = "BAAI/bge-m3"
    TOP_K: int = 5
    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 100

    TARGET_LANGS: List[str] = [
        "arabic", "bengali", "english", "finnish", "indonesian",
        "japanese", "korean", "russian", "swahili", "telugu", "thai"
    ]
    MAX_SAMPLES_PER_LANG: int = 100   # Change to 800 for full 8800 samples

    PROMPT_STRATEGIES: List[str] = ["minimal", "explicit_en", "explicit_native", "strict"]

    MAX_NEW_TOKENS: int = 48
    MAX_INPUT_TOKENS: int = 3800
    GENERATION_BATCH_SIZE: int = 8
    USE_4BIT: bool = True
    COMPUTE_DTYPE = torch.float16
    USE_DOUBLE_QUANT: bool = True
    QUANT_TYPE: str = "nf4"

    USE_BERTSCORE: bool = True
    BERTSCORE_MODEL: str = "bert-base-multilingual-cased"
    BERTSCORE_BATCH_SIZE: int = 16

    CONFIDENCE_LEVEL: float = 0.95
    N_BOOTSTRAP: int = 1000
    MAX_WORKERS: int = 4
    SAVE_RESULTS: bool = True
    OUTPUT_DIR: str = "/teamspace/studios/this_studio/outputs"
    TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# LANGUAGE UTILS
# =============================================================================
class LanguageUtils:
    SCRIPT_RANGES: Dict = {
        "arabic": (0x0600, 0x06FF),
        "bengali": (0x0980, 0x09FF),
        "japanese": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
        "korean": (0xAC00, 0xD7A3),
        "russian": (0x0400, 0x04FF),
        "telugu": (0x0C00, 0x0C7F),
        "thai": (0x0E00, 0x0E7F),
    }

    ISO_639_1: Dict = {
        "arabic": "ar", "bengali": "bn", "english": "en", "finnish": "fi",
        "indonesian": "id", "japanese": "ja", "korean": "ko", "russian": "ru",
        "swahili": None, "telugu": None, "thai": "th"
    }

    META: Dict = {
        "arabic": {"name": "Arabic", "native": "العربية", "script": "Arabic", "family": "Semitic", "word_order": "VSO", "resource": "high", "morphology": "fusional"},
        "bengali": {"name": "Bengali", "native": "বাংলা", "script": "Bengali", "family": "Indo-Aryan", "word_order": "SOV", "resource": "medium", "morphology": "fusional"},
        "english": {"name": "English", "native": "English", "script": "Latin", "family": "Germanic", "word_order": "SVO", "resource": "high", "morphology": "analytic"},
        "finnish": {"name": "Finnish", "native": "Suomi", "script": "Latin", "family": "Uralic", "word_order": "SVO", "resource": "medium", "morphology": "agglutinative"},
        "indonesian": {"name": "Indonesian", "native": "Bahasa Indonesia", "script": "Latin", "family": "Austronesian", "word_order": "SVO", "resource": "medium", "morphology": "agglutinative"},
        "japanese": {"name": "Japanese", "native": "日本語", "script": "CJK+Kana", "family": "Japonic", "word_order": "SOV", "resource": "high", "morphology": "agglutinative"},
        "korean": {"name": "Korean", "native": "한국어", "script": "Hangul", "family": "Koreanic", "word_order": "SOV", "resource": "high", "morphology": "agglutinative"},
        "russian": {"name": "Russian", "native": "Русский", "script": "Cyrillic", "family": "Slavic", "word_order": "SVO", "resource": "high", "morphology": "fusional"},
        "swahili": {"name": "Swahili", "native": "Kiswahili", "script": "Latin", "family": "Bantu", "word_order": "SVO", "resource": "low", "morphology": "agglutinative"},
        "telugu": {"name": "Telugu", "native": "తెలుగు", "script": "Telugu", "family": "Dravidian", "word_order": "SOV", "resource": "low", "morphology": "agglutinative"},
        "thai": {"name": "Thai", "native": "ไทย", "script": "Thai", "family": "Kra-Dai", "word_order": "SVO", "resource": "medium", "morphology": "analytic"},
    }

    @staticmethod
    def contains_script(text: str, lang: str) -> bool:
        if not text or not text.strip(): return False
        rng = LanguageUtils.SCRIPT_RANGES.get(lang)
        if rng is None: return any(c.isalpha() for c in text)
        if isinstance(rng, tuple):
            s, e = rng
            return any(s <= ord(c) <= e for c in text)
        return any(any(s <= ord(c) <= e for s, e in rng) for c in text)

    @staticmethod
    def script_char_count(text: str, lang: str) -> int:
        if not text: return 0
        rng = LanguageUtils.SCRIPT_RANGES.get(lang)
        if rng is None: return sum(1 for c in text if c.isalpha())
        if isinstance(rng, tuple):
            s, e = rng
            return sum(1 for c in text if s <= ord(c) <= e)
        return sum(1 for c in text if any(s <= ord(c) <= e for s, e in rng))


# =============================================================================
# TEXT NORMALIZER
# =============================================================================
class TextNormalizer:
    _CHAR_LEVEL = {"japanese", "korean", "thai"}

    @staticmethod
    def normalize(text: str, lang: str = "") -> str:
        if not text: return ""
        text = re.sub(r"\s+", " ", text).strip()
        if lang == "arabic":
            text = re.sub(r"[إأآا]", "ا", text)
            text = re.sub(r"ى", "ي", text)
            text = re.sub(r"ة", "ه", text)
        return text

    @staticmethod
    def tokenize(text: str, lang: str = "") -> List[str]:
        if not text: return []
        text = TextNormalizer.normalize(text, lang)
        if lang in TextNormalizer._CHAR_LEVEL:
            return [c for c in text if not c.isspace()]
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def token_f1_pair(pred: str, ref: str, lang: str) -> float:
        p_toks = TextNormalizer.tokenize(pred, lang)
        r_toks = TextNormalizer.tokenize(ref, lang)
        if not p_toks or not r_toks: return 0.0
        common = len(set(p_toks) & set(r_toks))
        prec = common / len(p_toks)
        rec = common / len(r_toks)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# =============================================================================
# DATASET MANAGER — Chunking + Parent URL Tracking
# =============================================================================
class DatasetManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buffers: Dict[str, List[Dict]] = {l: [] for l in cfg.TARGET_LANGS}
        self.corpus_chunks: Dict[str, List[Tuple[str, str]]] = {l: [] for l in cfg.TARGET_LANGS}  # (chunk, parent_url)

    @staticmethod
    def _extract_answer_text(doc_bytes: bytes, start: int, end: int) -> str:
        try:
            return doc_bytes[start:end].decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int = 768, overlap: int = 100) -> List[str]:
        if not text: return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def collect_samples(self):
        print("\n📚 Streaming TyDiQA primary_task + Chunking documents...")
        ds = load_dataset("google-research-datasets/tydiqa", "primary_task",
                          split="train", streaming=True, trust_remote_code=True)

        for ex in tqdm(ds, desc="Collecting"):
            lang = ex.get("language", "")
            if lang not in self.cfg.TARGET_LANGS: continue

            url = ex.get("document_url", "")
            doc_text = ex.get("document_plaintext", "")

            if url and doc_text:
                chunks = self._split_into_chunks(doc_text, self.cfg.CHUNK_SIZE, self.cfg.CHUNK_OVERLAP)
                for chunk in chunks:
                    if not any(c[1] == url and c[0] == chunk for c in self.corpus_chunks[lang]):
                        self.corpus_chunks[lang].append((chunk, url))

            if len(self.buffers[lang]) >= self.cfg.MAX_SAMPLES_PER_LANG:
                continue

            annotations = ex.get("annotations", {})
            starts = annotations.get("minimal_answers_start_byte", [])
            ends = annotations.get("minimal_answers_end_byte", [])
            valid_spans = [(s, e) for s, e in zip(starts, ends) if s >= 0]
            if not valid_spans: continue

            doc_bytes = doc_text.encode("utf-8")
            ref_answers = [self._extract_answer_text(doc_bytes, s, e)
                           for s, e in valid_spans if self._extract_answer_text(doc_bytes, s, e)]
            if not ref_answers: continue

            max_ctx_chars = 8000
            first_start_char = doc_text.encode("utf-8")[:valid_spans[0][0]].decode("utf-8", errors="replace")
            char_offset = len(first_start_char)
            ctx_start = max(0, char_offset - max_ctx_chars // 2)
            ctx_end = min(len(doc_text), ctx_start + max_ctx_chars)
            evidence = doc_text[ctx_start:ctx_end]

            self.buffers[lang].append({
                "question": ex.get("question_text", ""),
                "evidence": evidence,
                "reference_answers": ref_answers,
                "gold_url": url,
                "retrieved_urls": [],
                "hallucination": 0.0,
                "evidence_usage": 0.0
            })

            if all(len(self.buffers[l]) >= self.cfg.MAX_SAMPLES_PER_LANG for l in self.cfg.TARGET_LANGS):
                break

        print("\n✅ Collection + Chunking complete:")
        for lang, samples in self.buffers.items():
            print(f"  {lang:12s}: {len(samples):4d} samples | {len(self.corpus_chunks[lang]):5d} chunks")


# =============================================================================
# PROMPT BUILDER
# =============================================================================
class PromptBuilder:
    _CHARS_PER_TOKEN: float = 3.0

    @staticmethod
    def _ctx(context: str, cfg: Config) -> str:
        budget = int((cfg.MAX_INPUT_TOKENS - 200) * PromptBuilder._CHARS_PER_TOKEN)
        return context[:budget]

    @staticmethod
    def build(question: str, context: str, lang: str, cfg: Config) -> Dict[str, str]:
        m = LanguageUtils.META[lang]
        name, nat, scr = m["name"], m["native"], m["script"]
        ctx = PromptBuilder._ctx(context, cfg)
        return {
            "minimal": f"Answer the following question in {name} only.\nDo NOT answer in English.\n\nQuestion: {question}\n\nContext:\n{ctx}\n\nAnswer ({name}):",
            "explicit_en": f"You must respond ONLY in {name} ({nat}).\nUsing English is strictly forbidden.\n\nQuestion: {question}\n\nContext:\n{ctx}\n\nProvide a brief factual answer in {name}.\nAnswer:",
            "explicit_native": f"[Respond in {nat} / {name} ONLY. No English.]\n\nQuestion: {question}\n\nContext:\n{ctx}\n\nAnswer in {nat} ({name}), using {scr} script only. Keep it concise.\nAnswer:",
            "strict": f"LANGUAGE REQUIREMENT: {name} ({nat}) ONLY.\nSCRIPT REQUIREMENT: {scr} ONLY.\nENGLISH IS FORBIDDEN.\n\nQuestion: {question}\n\nContext:\n{ctx}\n\nGive a short, accurate answer in {name} using {scr} script.\nAnswer:"
        }


# =============================================================================
# MODEL MANAGER
# =============================================================================
class ModelManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.tok = None
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        from huggingface_hub import login
        login(self.cfg.HF_TOKEN)
        print(f"\n⭐ Loading {self.cfg.MODEL_NAME} (4-bit)…")

        bnb = BitsAndBytesConfig(
            load_in_4bit=self.cfg.USE_4BIT,
            bnb_4bit_compute_dtype=self.cfg.COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=self.cfg.USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=self.cfg.QUANT_TYPE,
        )

        self.tok = AutoTokenizer.from_pretrained(self.cfg.MODEL_NAME, trust_remote_code=True, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.MODEL_NAME,
            quantization_config=bnb,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=self.cfg.COMPUTE_DTYPE,
        )
        self.model.eval()
        print(f"✓ Model loaded | GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def _apply_chat_template(self, user_prompt: str) -> str:
        if not self.cfg.USE_CHAT_TEMPLATE or not hasattr(self.tok, "chat_template") or self.tok.chat_template is None:
            return user_prompt
        try:
            messages = [{"role": "user", "content": user_prompt}]
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_prompt

    def generate_batch(self, prompts: List[str]) -> List[str]:
        if not prompts: return []
        formatted = [self._apply_chat_template(p) for p in prompts]
        try:
            inputs = self.tok(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.MAX_INPUT_TOKENS,
                padding=True,
                padding_side="left",
            )
            max_prompt_len = inputs.input_ids.shape[1]
            inputs = {k: v.to(self.dev) for k, v in inputs.items()}

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=self.cfg.COMPUTE_DTYPE)
                if self.dev == "cuda" else contextlib.nullcontext()
            )

            with torch.no_grad(), autocast_ctx:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id,
                    use_cache=True,
                )

            answers = []
            for seq in out:
                answer_ids = seq[max_prompt_len:]
                answer = self.tok.decode(answer_ids, skip_special_tokens=True)
                answer = re.sub(r"<\|im_end\|>.*", "", answer, flags=re.S)
                answer = re.sub(r"<\|im_start\|>.*", "", answer, flags=re.S)
                answers.append(answer.strip())
            return answers
        except Exception as exc:
            print(f" ⚠️ Batch error: {exc}")
            return [""] * len(prompts)


# =============================================================================
# METRICS
# =============================================================================
class Metrics:
    @staticmethod
    def language_match_rate(results: List[Dict], lang: str) -> float:
        if not results: return 0.0
        return sum(LanguageUtils.contains_script(r["answer"], lang) for r in results) / len(results)

    @staticmethod
    def script_consistency_score(results: List[Dict], lang: str) -> float:
        if not results: return 0.0
        scores = [LanguageUtils.script_char_count(r["answer"], lang) / max(len(r["answer"].strip()), 1) if r["answer"] else 0.0 for r in results]
        return float(np.mean(scores))

    @staticmethod
    def evidence_usage_ratio(results: List[Dict], lang: str) -> float:
        if not results: return 0.0
        ratios = []
        for r in results:
            a_toks = set(TextNormalizer.tokenize(r["answer"], lang))
            e_toks = set(TextNormalizer.tokenize(r["evidence"], lang))
            ratios.append(len(a_toks & e_toks) / len(a_toks) if a_toks else 0.0)
        return float(np.mean(ratios))

    @staticmethod
    def hallucination_frequency(results: List[Dict], lang: str) -> float:
        if not results: return 0.0
        rates = []
        for r in results:
            a_toks = set(TextNormalizer.tokenize(r["answer"], lang))
            e_toks = set(TextNormalizer.tokenize(r["evidence"], lang))
            rates.append(len(a_toks - e_toks) / len(a_toks) if a_toks else 0.0)
        return float(np.mean(rates))

    @staticmethod
    def exact_match(pred: str, refs: List[str], lang: str) -> float:
        if not pred or not refs: return 0.0
        p = TextNormalizer.normalize(pred, lang)
        return float(any(p == TextNormalizer.normalize(r, lang) for r in refs))

    @staticmethod
    def token_f1(pred: str, refs: List[str], lang: str) -> float:
        if not pred or not refs: return 0.0
        return max(TextNormalizer.token_f1_pair(pred, r, lang) for r in refs)

    @staticmethod
    def transfer_rate(results: List[Dict], lang: str) -> float:
        valid = [r for r in results if r.get("reference_answers")]
        if not valid: return 0.0
        correct = sum(1 for r in valid if Metrics.exact_match(r["answer"], r["reference_answers"], lang) == 1.0 or Metrics.token_f1(r["answer"], r["reference_answers"], lang) > 0.5)
        return correct / len(valid)

    @staticmethod
    def bertscore_batch(predictions: List[str], references: List[List[str]], lang: str, cfg: Config) -> Dict[str, float]:
        if not cfg.USE_BERTSCORE: return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}
        try:
            from bert_score import score as _bs
            valid_preds, valid_refs = [], []
            for pred, refs in zip(predictions, references):
                if pred and pred.strip():
                    good_refs = [r for r in refs if r and r.strip()]
                    if good_refs:
                        valid_preds.append(pred)
                        valid_refs.append(good_refs[0])
            if not valid_preds: return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}
            lang_code = LanguageUtils.ISO_639_1.get(lang)
            P, R, F1 = _bs(valid_preds, valid_refs, model_type=cfg.BERTSCORE_MODEL, lang=lang_code,
                           batch_size=cfg.BERTSCORE_BATCH_SIZE, verbose=False,
                           device="cuda" if torch.cuda.is_available() else "cpu")
            return {"bertscore_P": float(P.mean()), "bertscore_R": float(R.mean()), "bertscore_F1": float(F1.mean())}
        except Exception as exc:
            print(f" ⚠️ BERTScore error: {exc}")
            return {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}

    @staticmethod
    def semantic_overlap_batch(predictions: List[str], evidences: List[str], lang: str, cfg: Config) -> float:
        if not cfg.USE_BERTSCORE: return 0.0
        try:
            from bert_score import score as _bs
            valid_preds, valid_evs = [], []
            for p, e in zip(predictions, evidences):
                if p and p.strip() and e and e.strip():
                    valid_preds.append(p)
                    valid_evs.append(e[:1500])
            if not valid_preds: return 0.0
            lang_code = LanguageUtils.ISO_639_1.get(lang)
            _, _, F1 = _bs(valid_preds, valid_evs, model_type=cfg.BERTSCORE_MODEL, lang=lang_code,
                           batch_size=cfg.BERTSCORE_BATCH_SIZE, verbose=False,
                           device="cuda" if torch.cuda.is_available() else "cpu")
            return float(F1.mean())
        except Exception as exc:
            print(f" ⚠️ Semantic overlap error: {exc}")
            return 0.0

    @staticmethod
    def evaluate_all(results: List[Dict], lang: str) -> Dict[str, float]:
        if not results: return {}
        base = {
            "language_match_rate": Metrics.language_match_rate(results, lang),
            "script_consistency": Metrics.script_consistency_score(results, lang),
            "evidence_usage": Metrics.evidence_usage_ratio(results, lang),
            "hallucination": Metrics.hallucination_frequency(results, lang),
        }
        em_scores = [Metrics.exact_match(r["answer"], r.get("reference_answers", []), lang) for r in results if r.get("reference_answers")]
        f1_scores = [Metrics.token_f1(r["answer"], r.get("reference_answers", []), lang) for r in results if r.get("reference_answers")]
        quality = {
            "exact_match": float(np.mean(em_scores)) if em_scores else 0.0,
            "f1_score": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "transfer_rate": Metrics.transfer_rate(results, lang),
        }
        lengths = [len(TextNormalizer.tokenize(r["answer"], lang)) for r in results]
        length_stats = {"mean_length": float(np.mean(lengths)), "median_length": float(np.median(lengths))}
        bert_placeholders = {"bertscore_P": 0.0, "bertscore_R": 0.0, "bertscore_F1": 0.0}
        return {**base, **quality, **bert_placeholders, **length_stats, "conflict_rate": 0.0, "semantic_overlap": 0.0}


# =============================================================================
# STAT TESTS, ERROR ANALYZER, CROSS-LINGUAL
# =============================================================================
class StatTests:
    @staticmethod
    def paired_ttest(a: List[float], b: List[float]) -> Dict:
        if len(a) < 2: return {"t": 0., "p": 1., "sig": False, "d": 0., "effect": "n/a"}
        t, p = stats.ttest_rel(a, b)
        diff = np.array(a) - np.array(b)
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
        eff = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        return {"t": float(t), "p": float(p), "sig": p < 0.05, "d": d, "effect": eff}

    @staticmethod
    def bootstrap_ci(data: List[float], n: int = 1000, conf: float = 0.95) -> Tuple[float, float, float]:
        if len(data) < 2:
            v = data[0] if data else 0.
            return v, v, v
        arr = np.array(data)
        boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n)]
        alpha = (1 - conf) / 2
        return float(np.mean(arr)), float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))

    @staticmethod
    def compare_strategies(evaluation: Dict, metric: str, languages: List[str]) -> Dict:
        strategies = list(evaluation.keys())
        results = {}
        pvals = []
        for i, sa in enumerate(strategies):
            for sb in strategies[i + 1:]:
                key = f"{sa}_vs_{sb}"
                va = [evaluation[sa][l][metric] for l in languages if l in evaluation.get(sa, {}) and metric in evaluation[sa][l]]
                vb = [evaluation[sb][l][metric] for l in languages if l in evaluation.get(sb, {}) and metric in evaluation[sb][l]]
                if len(va) >= 2:
                    r = StatTests.paired_ttest(va, vb)
                    results[key] = r
                    pvals.append(r["p"])
        if pvals:
            for key, p in zip(results, [min(p * len(pvals), 1.) for p in pvals]):
                results[key]["p_corrected"] = p
        return results


class ErrorAnalyzer:
    CATEGORIES = ["wrong_language", "partial_code_switch", "empty_answer", "too_short", "hallucination_high", "no_evidence_overlap", "refusal", "correct"]
    REFUSAL_RE = re.compile(r"\b(cannot|unable|can'?t|sorry|don'?t know|no information|not mentioned|unclear|I cannot)\b", re.I)

    @staticmethod
    def categorize(result: Dict, lang: str) -> List[str]:
        ans = result.get("answer", "")
        if not ans or not ans.strip(): return ["empty_answer"]
        errors = []
        if not LanguageUtils.contains_script(ans, lang): errors.append("wrong_language")
        sc = LanguageUtils.script_char_count(ans, lang) / max(len(ans.strip()), 1)
        if 0.2 < sc < 0.8: errors.append("partial_code_switch")
        if len(TextNormalizer.tokenize(ans, lang)) < 3: errors.append("too_short")
        a_toks = set(TextNormalizer.tokenize(ans, lang))
        e_toks = set(TextNormalizer.tokenize(result.get("evidence", ""), lang))
        if a_toks and len(a_toks - e_toks) / len(a_toks) > 0.8: errors.append("hallucination_high")
        if a_toks and len(a_toks & e_toks) == 0: errors.append("no_evidence_overlap")
        if ErrorAnalyzer.REFUSAL_RE.search(ans): errors.append("refusal")
        return errors or ["correct"]

    @staticmethod
    def analyze(results: List[Dict], lang: str) -> Dict:
        counts = {c: 0 for c in ErrorAnalyzer.CATEGORIES}
        for r in results:
            for cat in ErrorAnalyzer.categorize(r, lang):
                counts[cat] += 1
        n = max(len(results), 1)
        return {"counts": counts, "rates": {k: v / n for k, v in counts.items()}, "n": n}


class CrossLingualAnalysis:
    METRIC_COLS = ["language_match_rate", "script_consistency", "exact_match", "f1_score", "transfer_rate", "bertscore_F1", "conflict_rate", "semantic_overlap"]

    @staticmethod
    def build_typology_df(evaluation: Dict, available_langs: List[str]) -> pd.DataFrame:
        rows = []
        for strategy, lang_dict in evaluation.items():
            for lang in available_langs:
                if lang not in lang_dict: continue
                metrics = lang_dict[lang]
                meta = LanguageUtils.META.get(lang, {})
                row = {
                    "strategy": strategy, "language": lang,
                    "family": meta.get("family", "?"), "word_order": meta.get("word_order", "?"),
                    "resource": meta.get("resource", "?"), "morphology": meta.get("morphology", "?"),
                }
                for col in CrossLingualAnalysis.METRIC_COLS:
                    row[col] = metrics.get(col, np.nan)
                rows.append(row)
        return pd.DataFrame(rows)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class MultilingualPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = ModelManager(cfg)
        self.data = DatasetManager(cfg)
        self.results: Dict = {s: {l: [] for l in cfg.TARGET_LANGS} for s in cfg.PROMPT_STRATEGIES}
        self.evaluation: Dict = {}
        self.error_dict: Dict = {}
        self.stat_tests: Dict = {}
        self.ci_dict: Dict = {}
        self.typology_df: pd.DataFrame = pd.DataFrame()
        self.retrieval_stats: Dict = {}
        self.avail_langs: List[str] = []
        self.retrieval_indices: Dict = {}
        self.retriever_models: Dict = {}

    def setup(self):
        print("\n" + "=" * 90)
        print("🌍 HYBRID CHUNK RAG PIPELINE v5.7 — Best Practice")
        print("   Chunk-level retrieval | Document-level P/R/MRR")
        print("=" * 90)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.model.load()
        self.data.collect_samples()
        self.avail_langs = [l for l in self.cfg.TARGET_LANGS if self.data.buffers[l]]
        self.build_retrieval_indices()
        self.retrieve_contexts()
        self.compute_retrieval_metrics()

    def build_retrieval_indices(self):
        print("\n🔍 Building BGE-M3 indices on document chunks...")
        import transformers
        if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
            transformers.utils.import_utils.is_torch_fx_available = lambda: False

        from FlagEmbedding import BGEM3FlagModel
        for lang in self.avail_langs:
            chunk_list = [chunk for chunk, _ in self.data.corpus_chunks[lang]]
            if not chunk_list: continue
            print(f"  {lang}: {len(chunk_list)} chunks")
            model = BGEM3FlagModel(self.cfg.RETRIEVER_NAME, use_fp16=True)
            embeddings = model.encode(chunk_list, batch_size=8, max_length=8192)['dense_vecs']
            self.retriever_models[lang] = model
            self.retrieval_indices[lang] = {
                "chunks": chunk_list,
                "embeddings": embeddings,
                "parent_urls": [url for _, url in self.data.corpus_chunks[lang]]
            }
        print("✅ Chunk indices ready.")

    def retrieve_contexts(self):
        print("\n🔎 Retrieving top-k chunks...")
        for lang in self.avail_langs:
            if lang not in self.retrieval_indices: continue
            idx = self.retrieval_indices[lang]
            samples = self.data.buffers[lang]
            for ex in tqdm(samples, desc=f"Retrieve {lang}", leave=False):
                q_emb = self.retriever_models[lang].encode([ex["question"]])['dense_vecs']
                sims = q_emb @ idx["embeddings"].T
                top_k_idx = np.argsort(sims[0])[::-1][:self.cfg.TOP_K]
                retrieved_chunks = [idx["chunks"][i] for i in top_k_idx]
                retrieved_urls = [idx["parent_urls"][i] for i in top_k_idx]
                ex["evidence"] = "\n\n--- Retrieved Chunk ---\n\n".join(retrieved_chunks)
                ex["retrieved_urls"] = list(dict.fromkeys(retrieved_urls))
        print("✅ Chunk retrieval completed.")

    def compute_retrieval_metrics(self):
        print("\n📈 Computing Document-level Precision@5, Recall@5, MRR...")
        self.retrieval_stats = {}
        for lang in self.avail_langs:
            samples = self.data.buffers[lang]
            p5 = r5 = mrr = 0.0
            n = 0
            for ex in samples:
                gold = ex.get("gold_url", "")
                retrieved = ex.get("retrieved_urls", [])
                if not gold or not retrieved: continue
                n += 1
                if gold in retrieved:
                    rank = retrieved.index(gold) + 1
                    r5 += 1
                    p5 += 1.0 / self.cfg.TOP_K
                    mrr += 1.0 / rank
            if n > 0:
                self.retrieval_stats[lang] = {
                    "precision@5": p5 / n,
                    "recall@5": r5 / n,
                    "mrr": mrr / n,
                    "n": n
                }
                print(f"  {lang:12s} → P@5 = {p5/n:.3f} | R@5 = {r5/n:.3f} | MRR = {mrr/n:.3f}  ({n} queries)")
        print("✅ Document-level retrieval metrics computed.")

    def run_generation(self):
        n_total = sum(len(self.data.buffers[l]) for l in self.avail_langs)
        bs = self.cfg.GENERATION_BATCH_SIZE
        print(f"\n🚀 Generating answers ({n_total} samples × {len(self.cfg.PROMPT_STRATEGIES)} strategies)...")
        for lang in self.avail_langs:
            print(f"\n── {lang.upper()} ──")
            samples = self.data.buffers[lang]
            all_prompts: Dict[str, List[str]] = {s: [] for s in self.cfg.PROMPT_STRATEGIES}
            for ex in samples:
                p = PromptBuilder.build(ex["question"], ex["evidence"], lang, self.cfg)
                for strat in self.cfg.PROMPT_STRATEGIES:
                    all_prompts[strat].append(p[strat])

            for strat in self.cfg.PROMPT_STRATEGIES:
                strat_prompts = all_prompts[strat]
                all_answers: List[str] = []
                for i in tqdm(range(0, len(strat_prompts), bs), desc=f"{lang}/{strat}", leave=False):
                    batch = strat_prompts[i:i + bs]
                    answers = self.model.generate_batch(batch)
                    all_answers.extend(answers)

                for ex, ans in zip(samples, all_answers):
                    a_toks = set(TextNormalizer.tokenize(ans, lang))
                    e_toks = set(TextNormalizer.tokenize(ex["evidence"], lang))
                    halluc = len(a_toks - e_toks) / len(a_toks) if a_toks else 0.0
                    usage = len(a_toks & e_toks) / len(a_toks) if a_toks else 0.0

                    result = {
                        "question": ex["question"],
                        "evidence": ex["evidence"],
                        "reference_answers": ex["reference_answers"],
                        "answer": ans,
                        "hallucination": halluc,
                        "evidence_usage": usage
                    }
                    self.results[strat][lang].append(result)
            gc.collect()
            torch.cuda.empty_cache()

    def evaluate(self):
        from concurrent.futures import ThreadPoolExecutor
        print("\n📊 Computing CPU metrics...")
        evaluation: Dict = {s: {} for s in self.cfg.PROMPT_STRATEGIES}
        tasks = [(s, l) for s in self.cfg.PROMPT_STRATEGIES for l in self.avail_langs if self.results[s][l]]
        with ThreadPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as ex:
            future_map = {ex.submit(Metrics.evaluate_all, self.results[s][l], l): (s, l) for s, l in tasks}
            for fut in tqdm(future_map, desc="Evaluating"):
                s, l = future_map[fut]
                evaluation[s][l] = fut.result()
        self.evaluation = evaluation

    def run_bertscore(self):
        if not self.cfg.USE_BERTSCORE: return
        print("\n🔬 Computing BERTScore...")
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                if not res: continue
                preds = [r["answer"] for r in res]
                refs = [r.get("reference_answers", []) for r in res]
                bs = Metrics.bertscore_batch(preds, refs, l, self.cfg)
                if l in self.evaluation.get(s, {}):
                    self.evaluation[s][l].update(bs)
                gc.collect()
                torch.cuda.empty_cache()

    def compute_additional_metrics(self):
        print("\n📊 Computing Conflict Rate & Semantic Overlap...")
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                if not res or l not in self.evaluation.get(s, {}): continue
                conflicts = sum(1 for r in res if r.get("hallucination", 0) > 0.5 and r.get("evidence_usage", 0) < 0.3)
                self.evaluation[s][l]["conflict_rate"] = conflicts / len(res)
                preds = [r["answer"] for r in res]
                evidences = [r["evidence"] for r in res]
                sem = Metrics.semantic_overlap_batch(preds, evidences, l, self.cfg)
                self.evaluation[s][l]["semantic_overlap"] = sem

    def run_error_analysis(self):
        print("\n🔍 Error analysis…")
        self.error_dict = {s: {} for s in self.cfg.PROMPT_STRATEGIES}
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                if res:
                    self.error_dict[s][l] = ErrorAnalyzer.analyze(res, l)

    def run_stat_tests(self):
        print("\n📈 Statistical tests…")
        key_metrics = ["language_match_rate", "f1_score", "exact_match", "transfer_rate", "bertscore_F1", "conflict_rate", "semantic_overlap"]
        for met in key_metrics:
            if any(met in self.evaluation.get(s, {}).get(l, {}) for s in self.cfg.PROMPT_STRATEGIES for l in self.avail_langs):
                self.stat_tests[met] = StatTests.compare_strategies(self.evaluation, met, self.avail_langs)

    def compute_cis(self):
        print("\n📊 Bootstrap CIs…")
        self.ci_dict = {s: {} for s in self.cfg.PROMPT_STRATEGIES}
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                res = self.results[s].get(l, [])
                self.ci_dict[s][l] = {}
                for met, fn in [
                    ("f1_score", lambda r, _l=l: Metrics.token_f1(r["answer"], r.get("reference_answers", []), _l)),
                    ("exact_match", lambda r, _l=l: Metrics.exact_match(r["answer"], r.get("reference_answers", []), _l)),
                ]:
                    scores = [fn(r) for r in res if r.get("reference_answers")]
                    if len(scores) >= 2:
                        mean, lo, hi = StatTests.bootstrap_ci(scores, self.cfg.N_BOOTSTRAP, self.cfg.CONFIDENCE_LEVEL)
                        self.ci_dict[s][l][f"{met}_ci"] = {"mean": mean, "lo": lo, "hi": hi}

    def run_cross_lingual(self):
        print("\n🌍 Cross-lingual typology analysis…")
        self.typology_df = CrossLingualAnalysis.build_typology_df(self.evaluation, self.avail_langs)
        print(f" typology_df shape: {self.typology_df.shape}")

    def print_summary(self):
        print("\n" + "=" * 90)
        print("📊 RESULTS SUMMARY (Hybrid Chunk RAG with BGE-M3)")
        print("=" * 90)
        for strat in self.cfg.PROMPT_STRATEGIES:
            print(f"\n▸ {strat.upper()}")
            for lang in self.avail_langs:
                m = self.evaluation.get(strat, {}).get(lang, {})
                if m:
                    print(f" {lang:12s} | LMR={m.get('language_match_rate',0):.2f} | F1={m.get('f1_score',0):.2f} | "
                          f"EM={m.get('exact_match',0):.2f} | Conflict={m.get('conflict_rate',0):.2f} | "
                          f"SemOverlap={m.get('semantic_overlap',0):.2f}")

    def save(self):
        if not self.cfg.SAVE_RESULTS: return
        tag = self.cfg.TIMESTAMP

        out = {
            "config": {"model": self.cfg.MODEL_NAME, "langs": self.avail_langs, "strategies": self.cfg.PROMPT_STRATEGIES,
                       "samples": self.cfg.MAX_SAMPLES_PER_LANG, "retriever": self.cfg.RETRIEVER_NAME, "chunk_size": self.cfg.CHUNK_SIZE},
            "evaluation": self.evaluation,
            "ci": self.ci_dict,
            "stat_tests": self.stat_tests,
            "retrieval_stats": self.retrieval_stats,
        }

        # === FIX FOR NUMPY BOOL_ NOT JSON SERIALIZABLE ===
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        out = convert_numpy(out)
        # =================================================

        with open(os.path.join(self.cfg.OUTPUT_DIR, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        rows = []
        for s in self.cfg.PROMPT_STRATEGIES:
            for l in self.avail_langs:
                m = self.evaluation.get(s, {}).get(l, {})
                rows.append({"strategy": s, "language": l, **m})
        pd.DataFrame(rows).to_csv(os.path.join(self.cfg.OUTPUT_DIR, f"summary_{tag}.csv"), index=False)

        if not self.typology_df.empty:
            self.typology_df.to_csv(os.path.join(self.cfg.OUTPUT_DIR, f"typology_{tag}.csv"), index=False)

        print(f"✅ Results saved to {self.cfg.OUTPUT_DIR}")

    def run(self):
        try:
            self.setup()
            self.run_generation()
            self.evaluate()
            self.run_bertscore()
            self.compute_additional_metrics()
            self.run_error_analysis()
            self.run_stat_tests()
            self.compute_cis()
            self.run_cross_lingual()
            self.print_summary()
            self.save()
            print("\n" + "=" * 90)
            print("✅ HYBRID CHUNK RAG PIPELINE COMPLETE")
            print("=" * 90)
        except Exception:
            import traceback
            traceback.print_exc()
            raise


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    cfg = Config()
    if not torch.cuda.is_available():
        print("⚠️ No CUDA detected — generation will be slow.")
    MultilingualPipeline(cfg).run()
