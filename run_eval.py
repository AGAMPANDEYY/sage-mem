"""
run_eval.py — Evaluate write-time memory defenses on MMA-Bench (LoCoMo-10)

Usage:
    python3 run_eval.py
    python3 run_eval.py --seeds 0 1 2 --qa-per-case 20 --out results/my_run.json
    python3 run_eval.py --quick        # single seed, 5 QA per case (~30 seconds)

Output:
    Prints per-condition BCU and ASR table.
    Saves full results to --out file (default: results/mma_eval_results.json).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root: src/ is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from mma_bench_suite import (
    MMA_BENCH_CONDITIONS,
    aggregate_mma_metrics,
    load_mma_bench_cases,
    run_mma_bench_eval,
)
from mma_bench_suite import MAX_QA_PER_CASE as _DEFAULT_QA
from procedural import train_procedural_detector
from utils import set_all_seeds


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MMA-Bench write-time poisoning evaluation")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds (default: 0 1 2)")
    p.add_argument("--qa-per-case", type=int, default=_DEFAULT_QA,
                   help=f"Max QA pairs evaluated per LoCoMo case (default: {_DEFAULT_QA})")
    p.add_argument("--attacks", nargs="+",
                   default=["semantic_mimicry", "constructor_launder", "label_gaming"],
                   help="Attack types to inject in poisoned split")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Subset of conditions to run (default: all 5)")
    p.add_argument("--out", type=str, default="results/mma_eval_results.json",
                   help="Output JSON file (default: mma_eval_results.json)")
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: seed=0 only, 5 QA per case")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_results_table(summary: dict) -> None:
    cond_short = {
        "MMA_RetrieveTimeReliabilityScoring_Baseline":                  "MMA (baseline)",
        "RecursiveSummarizationConsolidation_NoConstructorGuard":        "RSum (no guard)",
        "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation": "H1 ConstructorGuard",
        "MonotoneProvenanceLedger_ConservativeTrustScoring":             "H2 MonotoneLedger",
        "RiskSensitiveToolActionFirewall_CorroborateOrConfirm":          "H3 ActionFirewall",
    }

    header = f"{'Condition':<26}  {'BCU clean':>10}  {'BCU poison':>10}  {'ASR poison':>10}  {'n':>6}"
    print()
    print("=" * len(header))
    print("  MMA-Bench Write-Time Poisoning Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cond, splits in summary.items():
        name = cond_short.get(cond, cond[:26])
        clean = splits.get("clean", {})
        pois  = splits.get("poisoned", {})
        bcu_c = clean.get("BenignCompletionUnderAttack", float("nan"))
        bcu_p = pois.get("BenignCompletionUnderAttack", float("nan"))
        asr_p = pois.get("ASR", float("nan"))
        n     = pois.get("n_cases", 0)
        marker = " ✓" if bcu_p >= 0.9 else (" ✗" if bcu_p <= 0.1 else "  ")
        print(f"  {name:<26}  {bcu_c:>10.4f}  {bcu_p:>10.4f}  {asr_p:>10.4f}  {n:>6}{marker}")

    print("=" * len(header))
    print("  BCU = BenignCompletionUnderAttack (higher = better defense)")
    print("  ASR = Attack Success Rate (lower = better defense)")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick:
        args.seeds = [0]
        args.qa_per_case = 5
        print("[quick mode] seed=0, 5 QA per case")

    # Patch MAX_QA_PER_CASE at runtime
    import mma_bench_suite
    mma_bench_suite.MAX_QA_PER_CASE = args.qa_per_case

    conditions = args.conditions or MMA_BENCH_CONDITIONS
    device = pick_device()
    print(f"Device: {device} | Seeds: {args.seeds} | QA/case: {args.qa_per_case} | "
          f"Attacks: {args.attacks}")

    # Train procedural detector
    set_all_seeds(0)
    hp = {
        "embed_dim": 256, "top_k": 8, "consolidation_period_K": 4, "keep_last_M_raw": 4,
        "trigger_step": 10, "poison_rate": 0.30,
        "ocr_noise_prob_low": 0.05, "ocr_noise_prob_high": 0.25,
        "trust_accept_threshold": 0.55, "abstain_on_low_trust": False,
        "mma_w_source": 0.5, "mma_w_decay": 0.2, "mma_w_consensus": 0.3,
        "mma_decay_half_life_steps": 50,
        "procedural_classifier_threshold": 0.60, "quarantine_on_fail": True,
        "chain_decay": 0.85, "independence_bonus": 1.2, "max_chain_len": 5,
        "tool_attestation_required": True,
        "high_risk_requires_corroboration": 2, "require_user_confirmation": True,
        "max_confirmation_rate": 0.25, "clean_confirmation_token_probability": 0.10,
        "procclf_dim": 256, "procclf_lr": 1e-4, "procclf_weight_decay": 0.01,
        "procclf_epochs": 3, "procclf_batch_size": 32, "procclf_max_grad_norm": 1.0,
        "label_stream_n": 5000,
    }
    detector = train_procedural_detector(
        seed=0, dim=hp["procclf_dim"], device=device,
        lr=hp["procclf_lr"], weight_decay=hp["procclf_weight_decay"],
        epochs=hp["procclf_epochs"], batch_size=hp["procclf_batch_size"],
        max_grad_norm=hp["procclf_max_grad_norm"],
    )

    # Load data
    cases = load_mma_bench_cases()
    print(f"Loaded {len(cases)} LoCoMo cases")

    # Run evaluation
    t0 = time.time()
    raw = run_mma_bench_eval(
        cases=cases,
        conditions=conditions,
        detector=detector,
        hp=hp,
        seeds=args.seeds,
        attack_types=args.attacks,
    )
    elapsed = time.time() - t0

    # Aggregate and display
    summary = aggregate_mma_metrics(raw)
    print_results_table(summary)
    print(f"Total runtime: {elapsed:.1f}s")

    # Save
    out_path = Path(args.out)
    output = {
        "summary": summary,
        "config": {
            "seeds": args.seeds,
            "qa_per_case": args.qa_per_case,
            "attack_types": args.attacks,
            "conditions": conditions,
            "device": str(device),
            "runtime_sec": round(elapsed, 2),
        },
        "raw_sample_count": {
            c: {s: len(v) for s, v in splits.items()}
            for c, splits in raw.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
