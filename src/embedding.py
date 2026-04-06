import re
from dataclasses import dataclass
from typing import List

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