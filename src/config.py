"""
config.py — Unified configuration system for SAGE-Mem.

All hyperparameters are declared here. Nothing is hardcoded elsewhere.
Values are loaded from a JSON config file and validated at load time.

Calibration workflow:
  1. Run `scripts/calibrate_trust.py` on clean-split LoCoMo eval results.
  2. It writes an updated `configs/calibrated_trust_config.json`.
  3. Pass that path to SAGEMemConfig.from_file() instead of the default.

No hyperparameter should ever appear as a raw literal in memory.py,
mma_bench_suite.py, guard_llm.py, or any evaluation script.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default_trust_config.json"


# ---------------------------------------------------------------------------
# Sub-configs (typed, validated)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCredConfig:
    """Per-source-type credibility scores (calibrated or prior)."""
    table: Dict[str, float]
    default: float

    def get(self, source_type: str) -> float:
        return self.table.get(source_type, self.default)

    def __post_init__(self):
        for k, v in self.table.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"source_cred[{k}]={v} must be in [0, 1]")
        if not (0.0 <= self.default <= 1.0):
            raise ValueError(f"default source_cred={self.default} must be in [0, 1]")


@dataclass(frozen=True)
class BayesianPriorEntry:
    alpha: float
    beta: float

    def __post_init__(self):
        if self.alpha <= 0 or self.beta < 0:
            raise ValueError(f"Beta prior requires alpha>0, beta>=0, got ({self.alpha}, {self.beta})")

    @property
    def mean(self) -> float:
        if self.beta == 0.0:
            return 1.0
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        if self.beta == 0.0:
            return 0.0
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n * n * (n + 1))


@dataclass(frozen=True)
class BayesianPriorConfig:
    priors: Dict[str, BayesianPriorEntry]
    default: BayesianPriorEntry

    def get(self, source_type: str) -> BayesianPriorEntry:
        return self.priors.get(source_type, self.default)


@dataclass(frozen=True)
class ThresholdConfig:
    # Write-time trust gate
    write_trust_threshold: float
    risk_quarantine_threshold: float

    # Provenance ledger
    chain_decay: float
    independence_bonus: float

    # Calibration targets
    alpha_fpr: float                    # max tolerated false-positive rate at write-time
    conformal_coverage: float           # desired coverage for conformal threshold

    # Anomaly detector
    anomaly_quarantine_percentile: float  # nth percentile of benign scores → threshold
    warmup_writes: int                    # writes before anomaly detection activates

    # Consistency graph
    consistency_contradiction_cosine: float  # cosine sim below this → contradiction
    consistency_confirm_cosine: float        # cosine sim above this → confirmation
    correction_plausibility_low: float
    correction_plausibility_high: float
    correction_frequency_max: int       # flag if user submits > this many corrections/session

    # Ensemble guard
    ensemble_escalation_delta: float    # |risk_1 - risk_2| above this → escalate
    ensemble_agreement_threshold: float # risk_1 + risk_2 above this → quarantine

    # Constructor guard
    avg_trust_high_integrity: float     # lineage avg trust above this → high integrity

    # Consolidation
    consolidation_summary_trust: float  # trust assigned to a clean guarded summary

    # Reactive trust tightening
    reactive_tightening_lambda: float   # θ_t = θ_0 * exp(-λ * quarantine_count)
    reactive_tightening_cap: float      # max threshold under reactive tightening

    # Channel reputation tracking
    channel_reputation_decay: float
    channel_reputation_recovery: float
    channel_reputation_min: float

    # Evidence → belief promotion / planning access
    belief_promotion_min_parent_trust: float
    belief_promotion_min_support: int
    belief_promotion_min_independent_support: int
    belief_promotion_visual_requires_nonvisual_support: bool
    conflict_discount_strength: float
    noisy_evidence_penalty: float
    visual_evidence_base_quality: float
    unsupported_visual_penalty: float
    planning_evidence_quality_floor: float

    def __post_init__(self):
        _assert_unit("write_trust_threshold", self.write_trust_threshold)
        _assert_unit("risk_quarantine_threshold", self.risk_quarantine_threshold)
        _assert_unit("chain_decay", self.chain_decay)
        _assert_unit("conformal_coverage", self.conformal_coverage)
        _assert_unit("alpha_fpr", self.alpha_fpr)
        _assert_unit("correction_plausibility_low", self.correction_plausibility_low)
        _assert_unit("correction_plausibility_high", self.correction_plausibility_high)
        _assert_unit("avg_trust_high_integrity", self.avg_trust_high_integrity)
        _assert_unit("consolidation_summary_trust", self.consolidation_summary_trust)
        _assert_unit("reactive_tightening_cap", self.reactive_tightening_cap)
        _assert_unit("channel_reputation_decay", self.channel_reputation_decay)
        _assert_unit("channel_reputation_min", self.channel_reputation_min)
        _assert_unit("belief_promotion_min_parent_trust", self.belief_promotion_min_parent_trust)
        _assert_unit("conflict_discount_strength", self.conflict_discount_strength)
        _assert_unit("noisy_evidence_penalty", self.noisy_evidence_penalty)
        _assert_unit("visual_evidence_base_quality", self.visual_evidence_base_quality)
        _assert_unit("unsupported_visual_penalty", self.unsupported_visual_penalty)
        _assert_unit("planning_evidence_quality_floor", self.planning_evidence_quality_floor)
        if not math.isfinite(self.channel_reputation_recovery) or self.channel_reputation_recovery < 1.0:
            raise ValueError("channel_reputation_recovery must be a finite float >= 1.0")
        if self.belief_promotion_min_support < 1:
            raise ValueError("belief_promotion_min_support must be >= 1")
        if self.belief_promotion_min_independent_support < 1:
            raise ValueError("belief_promotion_min_independent_support must be >= 1")
        if self.correction_plausibility_low >= self.correction_plausibility_high:
            raise ValueError("correction_plausibility_low must be < high")


@dataclass(frozen=True)
class ConsolidationConfig:
    period_K: int
    keep_last_M_raw: int
    procedural_classifier_threshold: float

    def __post_init__(self):
        if self.period_K < 1:
            raise ValueError("period_K must be >= 1")
        if self.keep_last_M_raw < 1:
            raise ValueError("keep_last_M_raw must be >= 1")
        _assert_unit("procedural_classifier_threshold", self.procedural_classifier_threshold)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    mma_w_source: float
    mma_w_decay: float
    mma_w_consensus: float
    mma_decay_half_life_steps: int

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if abs(self.mma_w_source + self.mma_w_decay + self.mma_w_consensus - 1.0) > 1e-6:
            raise ValueError("MMA weights must sum to 1.0")
        if self.mma_decay_half_life_steps < 1:
            raise ValueError("mma_decay_half_life_steps must be >= 1")


@dataclass(frozen=True)
class GuardConfig:
    tier1_model: str
    tier2_model: str
    tier1_model_oai: str
    tier2_model_oai: str
    cache_size: int
    max_text_chars: int

    def __post_init__(self):
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        if self.max_text_chars < 1:
            raise ValueError("max_text_chars must be >= 1")


@dataclass(frozen=True)
class InjectionConfig:
    """Controls attack injection position mode. No fixed positions."""
    position_mode: str  # "random" | "early" | "late" | "post_consolidation"
    early_fraction: float
    late_fraction: float
    post_consolidation_k: int

    def __post_init__(self):
        valid_modes = {"random", "early", "late", "post_consolidation"}
        if self.position_mode not in valid_modes:
            raise ValueError(f"position_mode must be one of {valid_modes}")
        _assert_unit("early_fraction", self.early_fraction)
        _assert_unit("late_fraction", self.late_fraction)
        if self.early_fraction >= self.late_fraction:
            raise ValueError("early_fraction must be < late_fraction")
        if self.post_consolidation_k < 1:
            raise ValueError("post_consolidation_k must be >= 1")


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SAGEMemConfig:
    """
    Root configuration object. Immutable after construction.
    Load via SAGEMemConfig.from_file(path) — never construct manually in eval scripts.
    """
    source_cred: SourceCredConfig
    bayesian_prior: BayesianPriorConfig
    thresholds: ThresholdConfig
    consolidation: ConsolidationConfig
    retrieval: RetrievalConfig
    guard: GuardConfig
    injection: InjectionConfig
    calibrated: bool
    calibration_source: str

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "SAGEMemConfig":
        """Load config from a JSON file. Validates all fields."""
        p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with p.open() as f:
            raw = json.load(f)

        src_raw = raw["source_cred"]
        source_cred = SourceCredConfig(
            table={k: float(v) for k, v in src_raw.items() if not k.startswith("_")},
            default=float(src_raw.get("_default", 0.35)),
        )

        bp_raw = raw["bayesian_prior"]
        bp_default_raw = bp_raw.get("_default", {"alpha": 3.5, "beta": 6.5})
        priors = {
            k: BayesianPriorEntry(alpha=float(v["alpha"]), beta=float(v["beta"]))
            for k, v in bp_raw.items()
            if not k.startswith("_")
        }
        bayesian_prior = BayesianPriorConfig(
            priors=priors,
            default=BayesianPriorEntry(
                alpha=float(bp_default_raw["alpha"]),
                beta=float(bp_default_raw["beta"]),
            ),
        )

        t = raw["thresholds"]
        thresholds = ThresholdConfig(
            write_trust_threshold=float(t["write_trust_threshold"]),
            risk_quarantine_threshold=float(t["risk_quarantine_threshold"]),
            chain_decay=float(t["chain_decay"]),
            independence_bonus=float(t["independence_bonus"]),
            alpha_fpr=float(t["alpha_fpr"]),
            conformal_coverage=float(t["conformal_coverage"]),
            anomaly_quarantine_percentile=float(t["anomaly_quarantine_percentile"]),
            warmup_writes=int(t["warmup_writes"]),
            consistency_contradiction_cosine=float(t["consistency_contradiction_cosine"]),
            consistency_confirm_cosine=float(t["consistency_confirm_cosine"]),
            correction_plausibility_low=float(t["correction_plausibility_low"]),
            correction_plausibility_high=float(t["correction_plausibility_high"]),
            correction_frequency_max=int(t["correction_frequency_max"]),
            ensemble_escalation_delta=float(t["ensemble_escalation_delta"]),
            ensemble_agreement_threshold=float(t["ensemble_agreement_threshold"]),
            avg_trust_high_integrity=float(t["avg_trust_high_integrity"]),
            reactive_tightening_lambda=float(t["reactive_tightening_lambda"]),
            reactive_tightening_cap=float(t.get("reactive_tightening_cap", 0.45)),
            channel_reputation_decay=float(t.get("channel_reputation_decay", 0.70)),
            channel_reputation_recovery=float(t.get("channel_reputation_recovery", 1.05)),
            channel_reputation_min=float(t.get("channel_reputation_min", 0.10)),
            belief_promotion_min_parent_trust=float(t.get("belief_promotion_min_parent_trust", 0.60)),
            belief_promotion_min_support=int(t.get("belief_promotion_min_support", 2)),
            belief_promotion_min_independent_support=int(t.get("belief_promotion_min_independent_support", 2)),
            belief_promotion_visual_requires_nonvisual_support=bool(
                t.get("belief_promotion_visual_requires_nonvisual_support", True)
            ),
            conflict_discount_strength=float(t.get("conflict_discount_strength", 0.45)),
            noisy_evidence_penalty=float(t.get("noisy_evidence_penalty", 0.65)),
            visual_evidence_base_quality=float(t.get("visual_evidence_base_quality", 0.90)),
            unsupported_visual_penalty=float(t.get("unsupported_visual_penalty", 0.70)),
            planning_evidence_quality_floor=float(t.get("planning_evidence_quality_floor", 0.35)),
            consolidation_summary_trust=float(t.get("consolidation_summary_trust", 0.70)),
        )

        c = raw["consolidation"]
        consolidation = ConsolidationConfig(
            period_K=int(c["period_K"]),
            keep_last_M_raw=int(c["keep_last_M_raw"]),
            procedural_classifier_threshold=float(c["procedural_classifier_threshold"]),
        )

        r = raw["retrieval"]
        retrieval = RetrievalConfig(
            top_k=int(r["top_k"]),
            mma_w_source=float(r["mma_w_source"]),
            mma_w_decay=float(r["mma_w_decay"]),
            mma_w_consensus=float(r["mma_w_consensus"]),
            mma_decay_half_life_steps=int(r["mma_decay_half_life_steps"]),
        )

        g = raw["guard"]
        guard = GuardConfig(
            tier1_model=str(g["tier1_model"]),
            tier2_model=str(g["tier2_model"]),
            tier1_model_oai=str(g["tier1_model_oai"]),
            tier2_model_oai=str(g["tier2_model_oai"]),
            cache_size=int(g["cache_size"]),
            max_text_chars=int(g["max_text_chars"]),
        )

        inj = raw["injection"]
        injection = InjectionConfig(
            position_mode=str(inj["position_mode"]),
            early_fraction=float(inj["early_fraction"]),
            late_fraction=float(inj["late_fraction"]),
            post_consolidation_k=int(inj["post_consolidation_k"]),
        )

        return cls(
            source_cred=source_cred,
            bayesian_prior=bayesian_prior,
            thresholds=thresholds,
            consolidation=consolidation,
            retrieval=retrieval,
            guard=guard,
            injection=injection,
            calibrated=bool(raw.get("_calibrated", False)),
            calibration_source=str(raw.get("_calibration_source", "unknown")),
        )

    def to_dict(self) -> dict:
        """Serialise back to the JSON dict format for logging."""
        return {
            "source_cred": {**self.source_cred.table, "_default": self.source_cred.default},
            "thresholds": {
                "write_trust_threshold": self.thresholds.write_trust_threshold,
                "risk_quarantine_threshold": self.thresholds.risk_quarantine_threshold,
                "chain_decay": self.thresholds.chain_decay,
                "alpha_fpr": self.thresholds.alpha_fpr,
                "conformal_coverage": self.thresholds.conformal_coverage,
                "anomaly_quarantine_percentile": self.thresholds.anomaly_quarantine_percentile,
                "reactive_tightening_lambda": self.thresholds.reactive_tightening_lambda,
                "reactive_tightening_cap": self.thresholds.reactive_tightening_cap,
                "channel_reputation_decay": self.thresholds.channel_reputation_decay,
                "channel_reputation_recovery": self.thresholds.channel_reputation_recovery,
                "channel_reputation_min": self.thresholds.channel_reputation_min,
                "belief_promotion_min_parent_trust": self.thresholds.belief_promotion_min_parent_trust,
                "belief_promotion_min_support": self.thresholds.belief_promotion_min_support,
                "belief_promotion_min_independent_support": self.thresholds.belief_promotion_min_independent_support,
                "belief_promotion_visual_requires_nonvisual_support": self.thresholds.belief_promotion_visual_requires_nonvisual_support,
                "conflict_discount_strength": self.thresholds.conflict_discount_strength,
                "noisy_evidence_penalty": self.thresholds.noisy_evidence_penalty,
                "visual_evidence_base_quality": self.thresholds.visual_evidence_base_quality,
                "unsupported_visual_penalty": self.thresholds.unsupported_visual_penalty,
                "planning_evidence_quality_floor": self.thresholds.planning_evidence_quality_floor,
            },
            "calibrated": self.calibrated,
            "calibration_source": self.calibration_source,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_unit(name: str, value: float) -> None:
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name}={value} must be a finite float in [0, 1]")
