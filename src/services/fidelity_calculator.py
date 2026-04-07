from __future__ import annotations

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
        return float(SequenceMatcher(None, original, reconstructed).ratio())

    def compute_embedding(self, original: str | None, reconstructed: str | None) -> float | None:
        if not self.enable_embeddings or self._model is None or util is None:
            return None
        if not original or not reconstructed:
            return None
        embeddings = self._model.encode([original, reconstructed], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
