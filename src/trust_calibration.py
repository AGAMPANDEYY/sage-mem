"""
trust_calibration.py — Bayesian per-channel trust estimation and conformal threshold calibration.

Three components:

1. BayesianChannelTrust
   Beta-Bernoulli reputation model per channel_id.
   Prior is initialized from config (calibrated or intuitive).
   Posterior updated online from gold-answer correctness signal.
   Trust = posterior mean. Uncertainty = posterior variance.

2. SourceCredibilityCalibrator
   Offline: given clean-split eval results, computes empirical
   P(item correct | source_type) for each source type.
   Writes updated trust values to a config JSON for the next run.

3. ConformalTrustThreshold
   Given a calibration set of (trust_score, is_attack) pairs from a held-out run,
   computes the trust threshold θ such that:
       P(attack_survived | attack_injected) ≤ alpha   (marginal coverage)
   using split-conformal prediction (Angelopoulos & Bates, 2022).

Theory:
   The Beta-Bernoulli model is the conjugate prior for Bernoulli observations.
   After n_correct successes and n_incorrect failures:
       posterior = Beta(alpha_0 + n_correct, beta_0 + n_incorrect)
       E[trust] = (alpha_0 + n_correct) / (alpha_0 + beta_0 + n)
   Credible interval (95%):
       scipy.stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

References:
   - Angelopoulos, A. & Bates, S. (2022). "A gentle introduction to conformal prediction
     and distribution-free uncertainty quantification." arXiv:2107.07511.
   - DeGroot, M. H. (1970). "Optimal Statistical Decisions." McGraw-Hill.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SAGEMemConfig, BayesianPriorEntry


# ---------------------------------------------------------------------------
# Bayesian per-channel trust model
# ---------------------------------------------------------------------------

@dataclass
class ChannelTrustState:
    """Mutable posterior state for a single channel_id."""
    alpha: float   # pseudo-successes (correct retrievals)
    beta: float    # pseudo-failures  (incorrect retrievals)
    n_updates: int = 0

    @property
    def mean(self) -> float:
        """Posterior mean = E[p | data]."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.5
        return self.alpha / total

    @property
    def variance(self) -> float:
        """Posterior variance of the Beta distribution."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.25
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def lower_credible(self) -> float:
        """2.5th percentile of posterior — conservative lower bound on trust."""
        # Analytic approximation: mean - 2*std
        std = math.sqrt(max(0.0, self.variance))
        return max(0.0, self.mean - 2.0 * std)

    @property
    def upper_credible(self) -> float:
        """97.5th percentile of posterior — optimistic upper bound on trust."""
        std = math.sqrt(max(0.0, self.variance))
        return min(1.0, self.mean + 2.0 * std)

    def update(self, was_correct: bool) -> None:
        """
        Bayesian update: one Bernoulli observation.
        Correct retrieval → increment alpha (success).
        Incorrect retrieval → increment beta (failure).
        """
        if was_correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.n_updates += 1

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "n_updates": self.n_updates,
            "posterior_mean": round(self.mean, 4),
            "posterior_variance": round(self.variance, 6),
        }


class BayesianChannelTrust:
    """
    Per-channel Beta-Bernoulli trust estimator.

    Each channel_id (a string like "ch_5_ocr" or "attacker_launder") gets an
    independent Beta(alpha, beta) posterior over its reliability.

    Priors are initialized from SAGEMemConfig.bayesian_prior, which maps
    source_type → BayesianPriorEntry. A channel inherits the prior of its
    source_type at first observation.

    Usage:
        trust = BayesianChannelTrust(cfg)
        trust.observe(channel_id="ch_5", source_type="tool_output_text",
                      was_correct=True)
        p = trust.get_trust(channel_id="ch_5", source_type="tool_output_text")
    """

    def __init__(self, cfg: SAGEMemConfig):
        self._cfg = cfg
        # channel_id → ChannelTrustState
        self._states: Dict[str, ChannelTrustState] = {}
        # source_type → aggregated state (for unknown channel_ids)
        self._source_type_states: Dict[str, ChannelTrustState] = {}

    def _init_channel(self, channel_id: str, source_type: str) -> ChannelTrustState:
        """Initialise a new channel state from the source_type prior."""
        prior: BayesianPriorEntry = self._cfg.bayesian_prior.get(source_type)
        state = ChannelTrustState(alpha=prior.alpha, beta=prior.beta)
        self._states[channel_id] = state
        return state

    def _init_source_type(self, source_type: str) -> ChannelTrustState:
        prior: BayesianPriorEntry = self._cfg.bayesian_prior.get(source_type)
        state = ChannelTrustState(alpha=prior.alpha, beta=prior.beta)
        self._source_type_states[source_type] = state
        return state

    def get_trust(self, channel_id: str, source_type: str) -> float:
        """
        Returns posterior mean trust for this channel.
        Falls back to source_type posterior if channel is unseen.
        Falls back to config source_cred prior if source_type is also unseen.
        """
        if channel_id in self._states:
            return self._states[channel_id].mean
        if source_type in self._source_type_states:
            return self._source_type_states[source_type].mean
        # Pure prior (no observations yet)
        return self._cfg.source_cred.get(source_type)

    def get_uncertainty(self, channel_id: str, source_type: str) -> float:
        """Posterior variance — higher means less data, less certain."""
        if channel_id in self._states:
            return self._states[channel_id].variance
        if source_type in self._source_type_states:
            return self._source_type_states[source_type].variance
        prior = self._cfg.bayesian_prior.get(source_type)
        return ChannelTrustState(prior.alpha, prior.beta).variance

    def observe(self, channel_id: str, source_type: str, was_correct: bool) -> None:
        """
        Update the posterior for a channel given a correctness observation.
        Called when a retrieved item's gold-answer support is determined.

        Updates both the per-channel state AND the source_type aggregate.
        """
        if channel_id not in self._states:
            self._init_channel(channel_id, source_type)
        self._states[channel_id].update(was_correct)

        if source_type not in self._source_type_states:
            self._init_source_type(source_type)
        self._source_type_states[source_type].update(was_correct)

    def reactive_threshold(self, quarantine_count: int) -> float:
        """
        Adaptive write-trust threshold that tightens after attacks are detected.

        θ_t = min(1, θ_0 × exp(λ × quarantine_count))

        This implements reactive trust degradation: the more attacks detected
        in the current session, the more conservative the write gate becomes.
        """
        theta_0 = self._cfg.thresholds.write_trust_threshold
        lam = self._cfg.thresholds.reactive_tightening_lambda
        theta_cap = self._cfg.thresholds.reactive_tightening_cap
        theta_t = theta_0 * math.exp(lam * quarantine_count)
        return min(theta_cap, theta_t)

    def state_summary(self) -> dict:
        return {
            "channels": {cid: s.to_dict() for cid, s in self._states.items()},
            "source_types": {st: s.to_dict() for st, s in self._source_type_states.items()},
        }


# ---------------------------------------------------------------------------
# Offline calibration: learn source credibility from clean-split eval results
# ---------------------------------------------------------------------------

class SourceCredibilityCalibrator:
    """
    Offline calibration of source_cred values from clean-split evaluation results.

    Given a results JSON file produced by run_eval.py or eval_buried_payload.py,
    computes for each source_type:

        empirical_cred[source_type] = P(item from source_type ∈ top-k correct retrievals)

    This requires per-retrieved-item source_type logging, which must be enabled
    in the eval harness (see mma_bench_suite.py: log_retrieved_source_types=True).

    Usage:
        cal = SourceCredibilityCalibrator()
        cal.add_results(results_path)
        cal.write_config(output_path)
    """

    def __init__(self):
        self._source_type_correct: Dict[str, int] = defaultdict(int)
        self._source_type_total: Dict[str, int] = defaultdict(int)

    def add_results(self, results_path: Path) -> None:
        """
        Parse a results JSON file and accumulate per-source-type correctness counts.
        Expected format: results["raw_per_item"] = list of
            {"source_type": str, "contributed_to_correct": bool}
        """
        with results_path.open() as f:
            data = json.load(f)

        per_item = data.get("raw_per_item", [])
        if not per_item:
            raise ValueError(
                f"No 'raw_per_item' key in {results_path}. "
                "Re-run eval with log_retrieved_source_types=True."
            )

        for entry in per_item:
            st = str(entry.get("source_type", "_unknown"))
            correct = bool(entry.get("contributed_to_correct", False))
            self._source_type_total[st] += 1
            if correct:
                self._source_type_correct[st] += 1

    def empirical_cred(self, min_count: int = 10) -> Dict[str, float]:
        """
        Returns {source_type: P(correct)} for types with >= min_count observations.
        Types with too few observations retain their prior from the config.
        """
        result: Dict[str, float] = {}
        for st, total in self._source_type_total.items():
            if total >= min_count:
                result[st] = self._source_type_correct[st] / total
        return result

    def write_config(
        self,
        output_path: Path,
        base_config_path: Optional[Path] = None,
        min_count: int = 10,
    ) -> None:
        """
        Write updated trust config JSON with empirical source_cred values.
        Source types with < min_count observations keep their prior values.
        """
        if base_config_path is None:
            from config import _DEFAULT_CONFIG_PATH
            base_config_path = _DEFAULT_CONFIG_PATH

        with base_config_path.open() as f:
            raw = json.load(f)

        empirical = self.empirical_cred(min_count=min_count)
        updated_cred = dict(raw["source_cred"])

        for st, cred in empirical.items():
            if st in updated_cred:
                updated_cred[st] = round(cred, 4)

        raw["source_cred"] = updated_cred
        raw["_calibrated"] = True
        raw["_calibration_source"] = "empirical_from_clean_split"

        # Update Bayesian priors to match empirical credibility.
        # Use a "virtual sample" of N=20 to set Beta(alpha, beta) so that mean = cred.
        N_virtual = 20
        for st, cred in empirical.items():
            if st in raw["bayesian_prior"]:
                alpha_new = round(cred * N_virtual, 2)
                beta_new = round((1.0 - cred) * N_virtual, 2)
                raw["bayesian_prior"][st] = {"alpha": alpha_new, "beta": beta_new}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(raw, f, indent=2)

        print(f"[calibrator] Wrote calibrated config to {output_path}")
        print(f"[calibrator] Updated {len(empirical)} source types: {list(empirical)}")


# ---------------------------------------------------------------------------
# Conformal trust threshold calibration
# ---------------------------------------------------------------------------

class ConformalTrustThreshold:
    """
    Split-conformal prediction for write-trust threshold calibration.

    Given a calibration set of (trust_score, is_attack) pairs from a
    held-out run, computes the threshold θ such that:

        P(attack_survived | attack_injected) ≤ alpha   [marginal coverage]

    using the Angelopoulos & Bates (2022) split-conformal procedure.

    Nonconformity score: s(x) = trust_score of attack item that survived.
    (Higher trust = higher nonconformity = harder to block.)

    Procedure:
        1. Collect s_i = trust_score of each attack item in calibration set.
        2. Sort ascending: s_(1) ≤ s_(2) ≤ ... ≤ s_(n).
        3. θ = s_(ceil((n+1)(1-alpha))) — the quantile threshold.
        4. Set write_trust_threshold = θ.

    This guarantees: for a new attack item drawn from the same distribution,
    P(trust_score ≥ θ) ≤ alpha, i.e., the attack survives with probability ≤ alpha.

    Note: This assumes attacks and benign items are drawn i.i.d. from their
    respective distributions — a standard conformal assumption.
    """

    def __init__(self, alpha: float, coverage: float):
        """
        alpha: target false negative rate (P(attack_survived) ≤ alpha).
        coverage: desired coverage = 1 - alpha. Must equal 1-alpha.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if abs(coverage - (1 - alpha)) > 1e-6:
            raise ValueError(f"coverage={coverage} must equal 1-alpha={1-alpha}")
        self.alpha = alpha
        self.coverage = coverage
        self._calibration_scores: List[float] = []

    def add_attack_trust_scores(self, scores: List[float]) -> None:
        """
        Add trust scores of attack items that were not quarantined (survived write gate).
        These are the nonconformity scores for the conformal procedure.
        """
        for s in scores:
            if math.isfinite(s) and 0.0 <= s <= 1.0:
                self._calibration_scores.append(float(s))

    def compute_threshold(self) -> float:
        """
        Returns the conformal trust threshold θ.
        Set write_trust_threshold = θ to achieve coverage guarantee.

        Returns 0.0 if calibration set is empty (no attacks observed).
        """
        if not self._calibration_scores:
            return 0.0

        n = len(self._calibration_scores)
        sorted_scores = sorted(self._calibration_scores)

        # Split-conformal quantile: ceil((n+1)(1-alpha)) / n
        q_index = math.ceil((n + 1) * (1.0 - self.alpha)) - 1
        q_index = max(0, min(q_index, n - 1))
        theta = sorted_scores[q_index]

        return float(theta)

    def coverage_at_threshold(self, theta: float) -> float:
        """
        Empirical coverage at a given threshold.
        = P(trust_score < theta) over calibration scores.
        """
        if not self._calibration_scores:
            return 1.0
        blocked = sum(1 for s in self._calibration_scores if s < theta)
        return blocked / len(self._calibration_scores)

    def summary(self) -> dict:
        if not self._calibration_scores:
            return {"n_calibration": 0, "threshold": 0.0, "empirical_coverage": 1.0}
        theta = self.compute_threshold()
        return {
            "n_calibration": len(self._calibration_scores),
            "alpha": self.alpha,
            "threshold": round(theta, 4),
            "empirical_coverage": round(self.coverage_at_threshold(theta), 4),
            "score_mean": round(float(np.mean(self._calibration_scores)), 4),
            "score_p50": round(float(np.median(self._calibration_scores)), 4),
            "score_p95": round(float(np.percentile(self._calibration_scores, 95)), 4),
        }
