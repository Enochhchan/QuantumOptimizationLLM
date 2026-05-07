from __future__ import annotations

import re
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


class FidelityCalculator:
    def __init__(self, enable_embeddings: bool) -> None:
        self.enable_embeddings = enable_embeddings
        self._model = None
        if enable_embeddings and SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model = None

    @staticmethod
    def compute_basic(original: str | None, reconstructed: str | None) -> float | None:
        if not original or not reconstructed:
            return None
        original_norm = FidelityCalculator._normalize_text(original)
        reconstructed_norm = FidelityCalculator._normalize_text(reconstructed)
        if not original_norm or not reconstructed_norm:
            return None

        seq_score = float(SequenceMatcher(None, original_norm, reconstructed_norm).ratio())
        token_score = FidelityCalculator._token_f1(original_norm, reconstructed_norm)
        concept_score = FidelityCalculator._content_recall(original_norm, reconstructed_norm)
        numeric_score = FidelityCalculator._number_recall(original_norm, reconstructed_norm)
        intent_score = FidelityCalculator._intent_recall(original_norm, reconstructed_norm)
        keyword_score = FidelityCalculator._keyword_recall(original_norm, reconstructed_norm)
        pattern_score = FidelityCalculator._constraint_pattern_recall(original_norm, reconstructed_norm)
        signature_score = FidelityCalculator._signature_similarity(original_norm, reconstructed_norm)
        # Blend lexical and semantic-ish coverage so faithful paraphrases aren't under-scored.
        raw_score = (
            (0.16 * seq_score)
            + (0.19 * token_score)
            + (0.16 * concept_score)
            + (0.14 * numeric_score)
            + (0.11 * intent_score)
            + (0.10 * keyword_score)
            + (0.10 * pattern_score)
            + (0.10 * signature_score)
        )
        calibrated = FidelityCalculator._calibrate(raw_score)
        return float(max(0.0, min(1.0, calibrated)))

    def compute_embedding(self, original: str | None, reconstructed: str | None) -> float | None:
        if not self.enable_embeddings or self._model is None or util is None:
            return None
        if not original or not reconstructed:
            return None
        embeddings = self._model.encode([original, reconstructed], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        # Normalize common numeric words.
        number_words = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
        }
        for word, digit in number_words.items():
            lowered = re.sub(rf"\b{word}\b", digit, lowered)

        # Canonicalize common optimization-domain synonyms.
        synonym_map = {
            "drivers": "driver",
            "workers": "driver",
            "couriers": "driver",
            "deliveries": "delivery",
            "shipments": "delivery",
            "jobs": "delivery",
            "tasks": "delivery",
            "routes": "route",
            "times": "time",
            "runtime": "time",
            "travel": "travel",
            "minimise": "minimize",
            "minimum": "minimize",
            "reduce": "minimize",
            "balanced": "balance",
            "balancing": "balance",
            "equal": "exactly",
            "reserve": "reserved",
            "priority": "priority",
        }
        for src, dst in synonym_map.items():
            lowered = re.sub(rf"\b{re.escape(src)}\b", dst, lowered)

        # Keep letters, digits, underscore and whitespace; drop punctuation/noise.
        cleaned = re.sub(r"[^a-z0-9_\s]", " ", lowered)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        tokens = cleaned.split()
        normalized_tokens: list[str] = []
        for tok in tokens:
            # Collapse identifiers like delivery11 -> delivery
            tok = re.sub(r"(\D)\d+$", r"\1", tok)
            tok = tok.strip("_")
            if tok.endswith("ies") and len(tok) > 4:
                tok = tok[:-3] + "y"
            elif tok.endswith("s") and len(tok) > 4:
                tok = tok[:-1]
            normalized_tokens.append(tok)
        return " ".join(t for t in normalized_tokens if t)

    @staticmethod
    def _token_f1(a: str, b: str) -> float:
        a_tokens = a.split()
        b_tokens = b.split()
        if not a_tokens or not b_tokens:
            return 0.0

        a_counts: dict[str, int] = {}
        b_counts: dict[str, int] = {}
        for tok in a_tokens:
            a_counts[tok] = a_counts.get(tok, 0) + 1
        for tok in b_tokens:
            b_counts[tok] = b_counts.get(tok, 0) + 1

        overlap = 0
        for tok, count in a_counts.items():
            overlap += min(count, b_counts.get(tok, 0))

        if overlap == 0:
            return 0.0
        precision = overlap / max(1, len(b_tokens))
        recall = overlap / max(1, len(a_tokens))
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _content_recall(a: str, b: str) -> float:
        stop_words = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
            "is", "are", "be", "as", "at", "from", "that", "this", "it", "must", "each",
            "atleast", "least", "more", "than", "while",
        }
        a_content = [tok for tok in a.split() if tok not in stop_words and len(tok) > 2]
        b_set = set(tok for tok in b.split() if tok not in stop_words and len(tok) > 2)
        if not a_content:
            return 0.0
        covered = sum(1 for tok in a_content if tok in b_set)
        return covered / max(1, len(a_content))

    @staticmethod
    def _number_recall(a: str, b: str) -> float:
        a_nums = re.findall(r"\b\d+(?:\.\d+)?\b", a)
        b_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", b))
        if not a_nums:
            return 1.0
        covered = sum(1 for n in a_nums if n in b_nums)
        return covered / max(1, len(a_nums))

    @staticmethod
    def _intent_recall(a: str, b: str) -> float:
        intents = [
            "at least",
            "at most",
            "no more than",
            "exactly",
            "minimize",
            "maximize",
            "reserved",
            "balance",
            "priority",
        ]
        a_set = {intent for intent in intents if intent in a}
        if not a_set:
            return 1.0
        covered = sum(1 for intent in a_set if intent in b)
        return covered / max(1, len(a_set))

    @staticmethod
    def _keyword_recall(a: str, b: str) -> float:
        stop_words = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
            "is", "are", "be", "as", "at", "from", "that", "this", "it", "must", "each",
            "then", "while", "into", "under", "over", "also",
        }
        a_words = [w for w in a.split() if w not in stop_words and len(w) >= 4]
        if not a_words:
            return 0.0
        # Keep top frequent original keywords.
        counts: dict[str, int] = {}
        for w in a_words:
            counts[w] = counts.get(w, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        key_terms = [w for w, _ in ranked[:16]]
        b_set = set(b.split())
        covered = sum(1 for term in key_terms if term in b_set)
        return covered / max(1, len(key_terms))

    @staticmethod
    def _constraint_pattern_recall(a: str, b: str) -> float:
        def extract_patterns(text: str) -> set[str]:
            patterns: set[str] = set()
            regexes = [
                r"\bexactly\s+\d+\b",
                r"\bat\s+least\s+\d+\b",
                r"\bat\s+most\s+\d+\b",
                r"\bno\s+more\s+than\s+\d+\b",
                r"\bminimize\b",
                r"\bmaximize\b",
                r"\bright\s+turn\b",
                r"\bleft\s+turn\b",
                r"\bpriority\b",
                r"\breserved\b",
            ]
            for rx in regexes:
                for match in re.findall(rx, text):
                    patterns.add(match.strip())
            return patterns

        a_patterns = extract_patterns(a)
        if not a_patterns:
            return 1.0
        b_patterns = extract_patterns(b)
        covered = sum(1 for p in a_patterns if p in b_patterns)
        return covered / max(1, len(a_patterns))

    @staticmethod
    def _signature_similarity(a: str, b: str) -> float:
        def signature(text: str) -> set[str]:
            sig: set[str] = set()
            # Constraint cue phrases + associated numbers when present.
            patterns = [
                (r"\bexactly\s+(\d+)\b", "exactly"),
                (r"\bat\s+least\s+(\d+)\b", "at_least"),
                (r"\bat\s+most\s+(\d+)\b", "at_most"),
                (r"\bno\s+more\s+than\s+(\d+)\b", "at_most"),
                (r"\bminimize\b", "minimize"),
                (r"\bmaximize\b", "maximize"),
            ]
            for rx, label in patterns:
                for m in re.findall(rx, text):
                    if isinstance(m, tuple):
                        m = m[0]
                    if isinstance(m, str) and m.isdigit():
                        sig.add(f"{label}:{m}")
                    else:
                        sig.add(label)
            # Domain anchors.
            anchors = ["driver", "delivery", "priority", "reserved", "balance", "travel", "route", "turn"]
            for a_tok in anchors:
                if re.search(rf"\b{re.escape(a_tok)}\b", text):
                    sig.add(f"kw:{a_tok}")
            return sig

        a_sig = signature(a)
        b_sig = signature(b)
        if not a_sig:
            return 1.0
        inter = len(a_sig.intersection(b_sig))
        union = len(a_sig.union(b_sig))
        if union == 0:
            return 0.0
        return inter / union

    @staticmethod
    def _calibrate(raw_score: float) -> float:
        # Human-aligned scaling: faithful paraphrases with varied wording should not score too low.
        raw = max(0.0, min(1.0, float(raw_score)))
        if raw < 0.35:
            return raw ** 0.80
        return raw ** 0.60
