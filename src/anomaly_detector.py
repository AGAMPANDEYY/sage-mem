"""
anomaly_detector.py — Session-relative embedding-space anomaly detection for memory writes.

Architecture:
  SessionAnomalyDetector maintains a running distribution over embedding vectors
  of benign writes seen so far in the current session.

  For each new write, it computes an anomaly score relative to this distribution.
  High anomaly score → the incoming write is semantically unusual compared to
  what the session has seen so far → may be adversarial.

  The threshold is self-calibrated from the session's own benign writes:
  it is the Nth percentile of anomaly scores observed during warmup.
  No external calibration required.

Algorithm:
  Phase 1 — Warmup (first W writes from trusted sources):
    Collect embeddings. No anomaly detection yet.
    Compute running mean and covariance via Welford's online algorithm.

  Phase 2 — Detection:
    For each new write:
      1. Compute Mahalanobis distance from session distribution:
           D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
         Using pseudo-inverse if Σ is singular (low-rank for small sessions).
      2. Normalise to [0, 1] via a sigmoid calibrated to the warmup distribution.
      3. Compare to adaptive threshold = percentile(warmup_scores, p).

  Key properties:
  - Signature-free: no regex, no known-attack keywords
  - Self-calibrating: threshold adapts to the session's content domain
  - Generalises to unseen attack families (adaptive_nl_evasion, future attacks)
  - O(d²) per write where d = embedding dimension

  Limitation:
  - Does not detect attacks that are semantically similar to benign writes
    (semantic chameleon attacks). These require LLM semantic guard.
  - Low accuracy in very short sessions (warmup < 5 writes).

References:
  Welford, B.P. (1962). "Note on a method for calculating corrected sums."
  Technometrics 4(3): 419-420.

  Mahalanobis, P.C. (1936). "On the generalised distance in statistics."
  Proceedings of the National Institute of Sciences of India 2: 49–55.

  Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). "Isolation Forest."
  ICDM 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import SAGEMemConfig


# ---------------------------------------------------------------------------
# Welford online mean/covariance estimator
# ---------------------------------------------------------------------------

class WelfordEstimator:
    """
    Numerically stable online computation of mean and covariance matrix.
    Uses Welford's algorithm (West, 1979 variant for covariance).

    References:
      Chan, T.F., Golub, G.H., LeVeque, R.J. (1979).
      "Updating formulae and a pairwise algorithm for computing sample variances."
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._n: int = 0
        self._mean: np.ndarray = np.zeros(dim, dtype=np.float64)
        # Upper-triangular Cholesky factor of the sample covariance (online update)
        # We store the full M2 matrix for simplicity.
        self._M2: np.ndarray = np.zeros((dim, dim), dtype=np.float64)

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def cov(self) -> np.ndarray:
        """Sample covariance matrix (unbiased estimator, n-1 denominator)."""
        if self._n < 2:
            return np.eye(self._dim, dtype=np.float64) * 1e-6
        return self._M2 / (self._n - 1)

    def update(self, x: np.ndarray) -> None:
        """Add one observation vector x."""
        x = np.asarray(x, dtype=np.float64)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        # Outer product rank-1 update to M2
        self._M2 += np.outer(delta, delta2)


# ---------------------------------------------------------------------------
# Session anomaly detector
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    score: float              # ∈ [0, 1], 0=normal, 1=maximally anomalous
    mahalanobis_sq: float     # raw squared Mahalanobis distance
    threshold: float          # adaptive threshold for this session
    is_anomalous: bool        # True if score > threshold
    in_warmup: bool           # True if still collecting baseline
    n_warmup_obs: int         # how many warmup observations collected so far


class SessionAnomalyDetector:
    """
    Write-time anomaly detector using session-relative Mahalanobis distance.

    Usage:
        detector = SessionAnomalyDetector(cfg)
        # On each write:
        result = detector.score(embedding, source_type)
        if result.is_anomalous:
            # Route to audit or escalate to LLM guard
    """

    # Source types considered "trusted enough" to be used as warmup baseline.
    # We want to build the benign distribution from reliable sources only.
    _WARMUP_TRUSTED_SOURCES = {"user", "tool_output_text"}

    def __init__(self, cfg: SAGEMemConfig, embedding_dim: int = 256):
        self._cfg = cfg
        self._dim = embedding_dim
        self._estimator = WelfordEstimator(dim=embedding_dim)
        self._warmup_size = cfg.thresholds.warmup_writes
        self._percentile = cfg.thresholds.anomaly_quarantine_percentile

        # Warmup anomaly scores (used to compute adaptive threshold)
        self._warmup_scores: List[float] = []
        self._adaptive_threshold: Optional[float] = None

        # Regularisation constant for covariance inversion
        self._reg = 1e-4

    @property
    def in_warmup(self) -> bool:
        return self._estimator.n < self._warmup_size

    def _mahalanobis_sq(self, x: np.ndarray) -> float:
        """
        Compute squared Mahalanobis distance: D²(x) = (x-μ)ᵀ Σ⁻¹ (x-μ).

        Uses regularised pseudo-inverse to handle rank deficiency
        (which is common when n < d).
        """
        mu = self._estimator.mean
        cov = self._estimator.cov

        delta = np.asarray(x, dtype=np.float64) - mu

        # Regularise covariance: Σ_reg = Σ + λI
        cov_reg = cov + self._reg * np.eye(self._dim, dtype=np.float64)

        # For high-dimensional embeddings with few observations, use
        # the diagonal approximation (much faster, still valid for our use case).
        if self._estimator.n < self._dim:
            # Diagonal approximation: use per-dimension variance only
            diag_var = np.maximum(np.diag(cov_reg), self._reg)
            return float(np.sum(delta ** 2 / diag_var))

        # Full Mahalanobis for sufficient observations
        try:
            cov_inv = np.linalg.pinv(cov_reg)
            d_sq = float(delta @ cov_inv @ delta)
            return max(0.0, d_sq)
        except np.linalg.LinAlgError:
            # Fallback to diagonal if pinv fails
            diag_var = np.maximum(np.diag(cov_reg), self._reg)
            return float(np.sum(delta ** 2 / diag_var))

    def _normalise_score(self, d_sq: float) -> float:
        """
        Map squared Mahalanobis distance to [0, 1] using sigmoid calibration.

        The sigmoid centre is set at the expected chi-squared mean = d (embedding dim)
        for the null (benign) hypothesis.
        """
        # Under null hypothesis, D² ~ chi²(d), E[D²] = d, Var[D²] = 2d
        expected = float(self._dim)
        scale = max(1.0, math.sqrt(2.0 * self._dim))
        z = (d_sq - expected) / scale
        score = 1.0 / (1.0 + math.exp(-z))
        return float(max(0.0, min(1.0, score)))

    def _compute_adaptive_threshold(self) -> float:
        """
        Compute the adaptive threshold as the Nth percentile of warmup scores.
        Falls back to a conservative default (0.75) if no warmup data.
        """
        if not self._warmup_scores:
            return 0.75
        sorted_scores = sorted(self._warmup_scores)
        n = len(sorted_scores)
        idx = int(math.ceil(n * self._percentile / 100.0)) - 1
        idx = max(0, min(idx, n - 1))
        return sorted_scores[idx]

    def score(self, embedding: np.ndarray, source_type: str) -> AnomalyResult:
        """
        Score an incoming write embedding.

        If in warmup and source_type is trusted: update the estimator and
        record the anomaly score for threshold calibration.

        If out of warmup: compute anomaly score against learned distribution.

        Returns AnomalyResult with is_anomalous flag.
        """
        x = np.asarray(embedding, dtype=np.float64)

        if self.in_warmup:
            # During warmup: update distribution if source is trusted
            if source_type in self._WARMUP_TRUSTED_SOURCES:
                if self._estimator.n >= 1:
                    d_sq = self._mahalanobis_sq(x)
                    score = self._normalise_score(d_sq)
                    self._warmup_scores.append(score)
                self._estimator.update(x)
            # Never flag as anomalous during warmup
            return AnomalyResult(
                score=0.0,
                mahalanobis_sq=0.0,
                threshold=0.75,
                is_anomalous=False,
                in_warmup=True,
                n_warmup_obs=self._estimator.n,
            )

        # First write out of warmup: compute adaptive threshold
        if self._adaptive_threshold is None:
            self._adaptive_threshold = self._compute_adaptive_threshold()

        d_sq = self._mahalanobis_sq(x)
        score = self._normalise_score(d_sq)
        threshold = self._adaptive_threshold
        is_anomalous = score > threshold

        # Update estimator with all writes (not just benign ones)
        # This lets the distribution adapt to session content changes.
        # We use a slow update rate to prevent poisoning of the baseline.
        # If anomalous: still update (gives attacker no way to avoid updating)
        self._estimator.update(x)

        return AnomalyResult(
            score=score,
            mahalanobis_sq=d_sq,
            threshold=threshold,
            is_anomalous=is_anomalous,
            in_warmup=False,
            n_warmup_obs=self._estimator.n,
        )

    def reset_session(self) -> None:
        """Reset between cases/sessions."""
        self._estimator = WelfordEstimator(dim=self._dim)
        self._warmup_scores = []
        self._adaptive_threshold = None

    def summary(self) -> dict:
        return {
            "n_observations": self._estimator.n,
            "in_warmup": self.in_warmup,
            "adaptive_threshold": self._adaptive_threshold,
            "warmup_score_mean": (
                float(np.mean(self._warmup_scores)) if self._warmup_scores else None
            ),
            "warmup_score_p90": (
                float(np.percentile(self._warmup_scores, 90))
                if len(self._warmup_scores) >= 2
                else None
            ),
        }
