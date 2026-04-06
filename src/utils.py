import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_s() -> float:
    return time.time()

class TimeGuard:
    def __init__(self, budget_s: float, stop_ratio: float = 0.8):
        self.budget_s = float(budget_s)
        self.stop_ratio = float(stop_ratio)
        self.t0 = now_s()

    @property
    def elapsed_s(self) -> float:
        return now_s() - self.t0

    @property
    def soft_limit_s(self) -> float:
        return self.budget_s * self.stop_ratio

    def should_stop(self) -> bool:
        return self.elapsed_s >= self.soft_limit_s

def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=0))

def safe_cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if not math.isfinite(an) or not math.isfinite(bn) or an < eps or bn < eps:
        return 0.0
    v = float(np.dot(a, b) / (an * bn))
    if not math.isfinite(v):
        return 0.0
    return max(-1.0, min(1.0, v))

def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

@dataclass
class EpisodeRegime:
    noise_level: str  # {low, high}
    attacker_adaptivity: str  # {non_adaptive, adaptive}

@dataclass
class EpisodeOutcome:
    attacked: bool
    regime: EpisodeRegime
    benign_success: bool
    policy_violation: bool
    hijack_attempted: bool
    poison_survived: bool
    abstention_steps: int
    total_steps: int
    confirmations_requested: int
    trusted_retrieval_capture_rate: float
    trust_escalation_events: int

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["regime"] = asdict(self.regime)
        return d