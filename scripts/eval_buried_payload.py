"""
eval_buried_payload.py — Evaluate WARP against the buried_payload attack.

The buried_payload attack embeds an ACTION_DIRECTIVE inside an existing
legitimate observation (same channel_id, same source_type, same trust score).
A document-level write gate cannot distinguish the malicious paragraph from
the benign ones without reading the full content.

Two defenses are compared:
  1. H1 without ASU (document-level): content detection quarantines the WHOLE
     document → ASR=0 but benign content from the same document is lost.
  2. H1 with ASU (segment-level): content detection quarantines ONLY the
     malicious paragraph → ASR=0 AND benign segments are preserved.

MMA (retrieve-time only) has no write-time content detection → ASR=1.0.

This experiment isolates the sub-document semantic mismatch problem and
validates Atomic Semantic Unit segmentation as a surgical defense.

Hypothesis:
  H_ASU: For buried_payload attacks, H1+ASU achieves equal or lower ASR than
         H1 (document-level) while achieving higher or equal BCU-clean,
         constituting a Pareto improvement within the consolidation architecture.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

from mma_bench_suite import (
    aggregate_eval_metrics,
    load_mma_bench_cases,
    load_mm_browsecomp_cases,
    run_mma_bench_eval,
    run_mm_browsecomp_eval,
)
from procedural import train_procedural_detector
from utils import set_all_seeds

load_dotenv()

SEEDS = [0, 1, 2]
QA_PER_CASE = 20
# Only the buried_payload attack — isolates this specific threat
ATTACKS_BURIED_ONLY = ["buried_payload"]
# All attacks including buried — shows combined exposure
ATTACKS_ALL = [
    "semantic_mimicry", "constructor_launder", "label_gaming",
    "ocr_injection", "vision_caption_injection", "buried_payload",
]
OUT = Path("results/buried_payload_eval.json")
OUT_TABLE = Path("results/buried_payload_table.txt")

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"Device: {device} | Seeds: {SEEDS} | QA/case: {QA_PER_CASE}")

set_all_seeds(0)
hp_base = {
    "embed_dim": 256, "top_k": 8, "consolidation_period_K": 4, "keep_last_M_raw": 4,
    "ocr_noise_prob_low": 0.05, "ocr_noise_prob_high": 0.25,
    "multimodal_turn_rate": 0.20, "vision_caption_mode": "synthetic",
    "trust_accept_threshold": 0.55, "abstain_on_low_trust": False,
    "mma_w_source": 0.5, "mma_w_decay": 0.2, "mma_w_consensus": 0.3,
    "mma_decay_half_life_steps": 50, "short_context_keep_last_k": 8,
    "procedural_classifier_threshold": 0.60, "quarantine_on_fail": True,
    "chain_decay": 0.85, "independence_bonus": 1.2, "max_chain_len": 5,
    "h2_write_trust_threshold": 0.25, "h3_write_trust_threshold": 0.20,
    "tool_attestation_required": True,
    "high_risk_requires_corroboration": 2, "require_user_confirmation": True,
    "max_confirmation_rate": 0.25, "clean_confirmation_token_probability": 0.10,
    "procclf_dim": 256, "procclf_lr": 1e-4, "procclf_weight_decay": 0.01,
    "procclf_epochs": 3, "procclf_batch_size": 32, "procclf_max_grad_norm": 1.0,
    "label_stream_n": 5000,
    "enable_multimodal_locomo": True, "enable_cross_topic_split": False,
    "mem0_api_key": None, "mem0_infer": True,
    "enable_asu_segmentation": False,  # default: document-level
    "asu_max_seg_len": 400,
}

import mma_bench_suite
mma_bench_suite.MAX_QA_PER_CASE = QA_PER_CASE

detector = train_procedural_detector(
    seed=0, dim=hp_base["procclf_dim"], device=device,
    lr=hp_base["procclf_lr"], weight_decay=hp_base["procclf_weight_decay"],
    epochs=hp_base["procclf_epochs"], batch_size=hp_base["procclf_batch_size"],
    max_grad_norm=hp_base["procclf_max_grad_norm"],
)

locomo_cases = load_mma_bench_cases()
print(f"Loaded {len(locomo_cases)} LoCoMo cases")

mm_path = Path("data/mm_browsecomp_cases_73.jsonl")
mm_cases = load_mm_browsecomp_cases(mm_path) if mm_path.exists() else []
print(f"Loaded {len(mm_cases)} MM-BrowseComp cases\n")

CONDITIONS = [
    "MMA_RetrieveTimeReliabilityScoring_Baseline",
    "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation",
]

results = {}

def run_and_record(label, cases, run_fn, conditions, hp, attacks):
    t0 = time.time()
    raw = run_fn(cases=cases, conditions=conditions, detector=detector,
                 hp=hp, seeds=SEEDS, attack_types=attacks)
    summary = aggregate_eval_metrics(raw)
    elapsed = time.time() - t0
    results[label] = {"summary": summary, "elapsed_s": round(elapsed, 1),
                      "n_cases": len(cases), "attacks": attacks}
    for cond, splits in summary.items():
        c = splits.get("clean", {}); p = splits.get("poisoned", {})
        short = cond.split("_")[0][:12]
        print(f"  [{short:12}] BCU-clean={c.get('BenignCompletionUnderAttack',0):.4f}  "
              f"ASR={p.get('ASR',0):.4f}  BCU-pois={p.get('BenignCompletionUnderAttack',0):.4f}  "
              f"n={p.get('n_cases',0)}")
    print(f"  ({elapsed:.1f}s)\n")

# ── LoCoMo: buried_payload only ─────────────────────────────────────────────
print("=== LoCoMo: buried_payload attack ONLY ===")
print("--- H1 document-level (no ASU) ---")
hp_no_asu = dict(hp_base, enable_asu_segmentation=False)
run_and_record("locomo_no_asu_buried", locomo_cases, run_mma_bench_eval,
               CONDITIONS, hp_no_asu, ATTACKS_BURIED_ONLY)

print("--- H1 + ASU segmentation (seg ≤400 chars) ---")
hp_asu = dict(hp_base, enable_asu_segmentation=True, asu_max_seg_len=400)
run_and_record("locomo_asu_buried", locomo_cases, run_mma_bench_eval,
               CONDITIONS, hp_asu, ATTACKS_BURIED_ONLY)

# ── LoCoMo: all attacks including buried_payload ─────────────────────────────
print("=== LoCoMo: ALL attacks (including buried_payload) ===")
print("--- H1 document-level ---")
run_and_record("locomo_no_asu_all", locomo_cases, run_mma_bench_eval,
               CONDITIONS, hp_no_asu, ATTACKS_ALL)

print("--- H1 + ASU ---")
run_and_record("locomo_asu_all", locomo_cases, run_mma_bench_eval,
               CONDITIONS, hp_asu, ATTACKS_ALL)

# ── MM-BrowseComp: buried_payload (if cases available) ──────────────────────
if mm_cases:
    print("=== MM-BrowseComp: buried_payload attack ONLY ===")
    print("--- document-level ---")
    run_and_record("mm_no_asu_buried", mm_cases, run_mm_browsecomp_eval,
                   CONDITIONS, hp_no_asu, ATTACKS_BURIED_ONLY)

    print("--- ASU ---")
    run_and_record("mm_asu_buried", mm_cases, run_mm_browsecomp_eval,
                   CONDITIONS, hp_asu, ATTACKS_BURIED_ONLY)

# ── Save ─────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"config": {"seeds": SEEDS, "qa_per_case": QA_PER_CASE,
                          "asu_max_seg_len": 400},
               "results": results}, f, indent=2)

# ── ASCII summary table ───────────────────────────────────────────────────────
cond_short = {
    "MMA_RetrieveTimeReliabilityScoring_Baseline": "MMA (retrieve-only)",
    "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation": "H1 ConstructorGuard",
}
lines = []
lines.append("=" * 100)
lines.append("  WARP vs Buried-Payload Attack: Document-Level vs ASU Segmentation")
lines.append("  Hypothesis: H1+ASU achieves equal/lower ASR AND equal/higher BCU-clean vs H1 doc-level")
lines.append("=" * 100)
lines.append(f"  {'Experiment':<35} {'Condition':<22} {'BCU-clean':>10} {'ASR':>8} {'BCU-pois':>10} {'n':>6}")
lines.append("-" * 100)

for exp_label, data in results.items():
    for cond, splits in data["summary"].items():
        c = splits.get("clean", {}); p = splits.get("poisoned", {})
        lines.append(
            f"  {exp_label:<35} {cond_short.get(cond, cond[:22]):<22} "
            f"{c.get('BenignCompletionUnderAttack',0):>10.4f} "
            f"{p.get('ASR',0):>8.4f} "
            f"{p.get('BenignCompletionUnderAttack',0):>10.4f} "
            f"{p.get('n_cases',0):>6}"
        )

lines.append("=" * 100)
lines.append("")
lines.append("  Key claim: ASU segmentation is a Pareto improvement — equal or lower ASR,")
lines.append("  equal or higher BCU-clean — because only the malicious segment is quarantined.")
lines.append("  MMA (retrieve-only) has no write-time content detection → ASR=1.0 on buried attacks.")
lines.append("=" * 100)

table = "\n".join(lines)
print("\n" + table)
with open(OUT_TABLE, "w") as f:
    f.write(table + "\n")

print(f"\nSaved: {OUT}")
print(f"Saved: {OUT_TABLE}")
