import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

@dataclass
class HashedTextEmbedder:
    dim: int
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.RandomState(self.seed)
        self._sign = rng.choice([-1.0, 1.0], size=(self.dim,)).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros((self.dim,), dtype=np.float32)
        toks = tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            h = (hash(tok) ^ self.seed) % self.dim
            vec[h] += self._sign[h]
        n = float(np.linalg.norm(vec))
        if n > 1e-8:
            vec /= n
        if not np.isfinite(vec).all():
            vec = np.zeros((self.dim,), dtype=np.float32)
        return vec


class SentenceTransformerEmbedder:
    """
    Semantic embedder using all-MiniLM-L6-v2 from sentence-transformers.
    Replaces HashedTextEmbedder for S2 conflict detection in the consistency
    graph — catches paraphrased fact overwrites that hash-based cosine misses.
    Lazy-loaded on first use so import cost is zero when not used.
    """
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None
        self.dim = 384  # all-MiniLM-L6-v2 output dimension

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, text: str) -> np.ndarray:
        self._load()
        text = text.strip()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vec, dtype=np.float32)


# Singleton — shared across all consistency graph instances
_sentence_embedder: Optional[SentenceTransformerEmbedder] = None

def get_sentence_embedder() -> SentenceTransformerEmbedder:
    global _sentence_embedder
    if _sentence_embedder is None:
        _sentence_embedder = SentenceTransformerEmbedder()
    return _sentence_embedder