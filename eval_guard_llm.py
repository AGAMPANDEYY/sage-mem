"""
eval_guard_llm.py — Evaluate WARP with the LLM-based write-time guard.

Two new capabilities being validated:

1. SEMANTIC ADVERSARIAL DETECTION (replaces regex-only approach)
   The WriteTimeGuard (claude-haiku at write boundary) classifies incoming
   text as DATA | DIRECTIVE | METADATA using semantic understanding, not
   pattern matching on hardcoded strings. This generalizes to real-world
   adversarial noise that evades regex filters.

2. VISION AUGMENTATION (improves MM-BrowseComp BCU-clean)
   For MM-BrowseComp cases with image URLs, the guard's vision extractor
   runs claude-haiku with the question image and adds extracted facts as
   additional observations. This closes the gap between web-text observations
   (often just page titles) and the visual information needed to answer.

Conditions:
  - MMA baseline (no write guard, no vision aug) — reference
  - H1 + regex only (current state from RESULTS.md) — baseline
  - H1 + LLM guard (semantic classification, no vision) — guard contribution
  - H1 + LLM guard + vision aug — full system on MM-BrowseComp

Hypothesis H_GUARD:
  On LoCoMo: H1+guard achieves ASR ≤ H1+regex (equal or lower).
  On MM-BrowseComp: H1+guard+vision achieves BCU-clean > H1+regex (higher utility).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

from guard_llm import build_guard, WriteTimeGuard
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

ATTACKS = [
    "semantic_mimicry", "constructor_launder", "label_gaming",
    "ocr_injection", "vision_caption_injection", "buried_payload",
]
OUT = Path("results/guard_llm_eval.json")
OUT_TABLE = Path("results/guard_llm_table.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WARP with the LLM write-time guard")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Random seeds (default: 0 1 2)")
    parser.add_argument("--qa-per-case", type=int, default=20,
                        help="Max QA pairs per LoCoMo case (default: 20)")
    parser.add_argument("--locomo-case-limit", type=int, default=None,
                        help="Optional limit on LoCoMo cases for bounded pilots")
    parser.add_argument("--mm-case-limit", type=int, default=None,
                        help="Optional limit on MM-BrowseComp cases for bounded pilots")
    parser.add_argument("--skip-locomo", action="store_true",
                        help="Skip LoCoMo experiments")
    parser.add_argument("--skip-mm", action="store_true",
                        help="Skip MM-BrowseComp experiments")
    parser.add_argument("--out", type=str, default=str(OUT),
                        help="Output JSON path")
    parser.add_argument("--out-table", type=str, default=str(OUT_TABLE),
                        help="Output table path")
    return parser.parse_args()


ARGS = parse_args()
SEEDS = ARGS.seeds
QA_PER_CASE = ARGS.qa_per_case
OUT = Path(ARGS.out)
OUT_TABLE = Path(ARGS.out_table)

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
    "enable_asu_segmentation": True, "asu_max_seg_len": 400,
    # Guard flags (enabled below per condition)
    "enable_llm_write_guard": False,
    "enable_vision_augmentation": False,
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
if ARGS.locomo_case_limit is not None:
    locomo_cases = locomo_cases[: max(0, int(ARGS.locomo_case_limit))]
print(f"Loaded {len(locomo_cases)} LoCoMo cases")

mm_path = Path("data/mm_browsecomp_cases_73.jsonl")
mm_cases = load_mm_browsecomp_cases(mm_path) if mm_path.exists() else []
if ARGS.mm_case_limit is not None:
    mm_cases = mm_cases[: max(0, int(ARGS.mm_case_limit))]
print(f"Loaded {len(mm_cases)} MM-BrowseComp cases\n")

CONDITIONS = [
    "MMA_RetrieveTimeReliabilityScoring_Baseline",
    "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation",
]

# Build the LLM guard (shared across conditions; has internal LRU cache)
print("Initializing LLM write-time guard (claude-haiku)...")
guard = build_guard({
    "enable_llm_write_guard": True,
    "guard_model": "claude-haiku-4-5-20251001",
    "guard_cache_size": 8192,
    "guard_risk_threshold": 0.65,
})
if guard is None:
    print("WARNING: ANTHROPIC_API_KEY not set — guard disabled, running regex-only")
else:
    print(f"Guard ready: model={guard.model}\n")

results = {}


def save_checkpoint() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {
                "config": {
                    "seeds": SEEDS,
                    "qa_per_case": QA_PER_CASE,
                    "guard_model": guard.model if guard else None,
                    "attacks": ATTACKS,
                    "locomo_case_limit": ARGS.locomo_case_limit,
                    "mm_case_limit": ARGS.mm_case_limit,
                },
                "results": results,
            },
            f,
            indent=2,
        )


def merge_raw_results(dst, src):
    for cond, splits in src.items():
        cond_dst = dst.setdefault(cond, {})
        for split, rows in splits.items():
            cond_dst.setdefault(split, [])
            cond_dst[split].extend(rows)
    return dst


def run_and_record(label, cases, run_fn, conditions, hp, attacks, use_guard):
    g = guard if use_guard else None
    t0 = time.time()
    raw = run_fn(cases=cases, conditions=conditions, detector=detector,
                 hp=hp, seeds=SEEDS, attack_types=attacks, guard=g)
    summary = aggregate_eval_metrics(raw)
    elapsed = time.time() - t0
    results[label] = {"summary": summary, "elapsed_s": round(elapsed, 1),
                      "n_cases": len(cases), "attacks": attacks,
                      "guard_enabled": use_guard,
                      "guard_stats": guard.stats() if guard and use_guard else {}}
    for cond, splits in summary.items():
        c = splits.get("clean", {}); p = splits.get("poisoned", {})
        short = cond.split("_")[0][:12]
        print(f"  [{short:12}] BCU-clean={c.get('BenignCompletionUnderAttack',0):.4f}  "
              f"ASR={p.get('ASR',0):.4f}  BCU-pois={p.get('BenignCompletionUnderAttack',0):.4f}  "
              f"n={p.get('n_cases',0)}")
    if guard and use_guard:
        s = guard.stats()
        print(f"  Guard stats: api_calls={s['api_calls']}  cache_hit_rate={s['cache_hit_rate']:.2f}  "
              f"quarantine={s['quarantine_count']}  directives={s['directive_count']}")
    print(f"  ({elapsed:.1f}s)\n")
    save_checkpoint()


def run_and_record_casewise(label, cases, run_fn, conditions, hp, attacks, use_guard):
    g = guard if use_guard else None
    raw_merged = {c: {"clean": [], "poisoned": []} for c in conditions}
    t0 = time.time()
    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        raw_one = run_fn(
            cases=[case],
            conditions=conditions,
            detector=detector,
            hp=hp,
            seeds=SEEDS,
            attack_types=attacks,
            guard=g,
        )
        merge_raw_results(raw_merged, raw_one)
        results[label] = {
            "summary": aggregate_eval_metrics(raw_merged),
            "elapsed_s": round(time.time() - t0, 1),
            "n_cases": len(cases),
            "attacks": attacks,
            "guard_enabled": use_guard,
            "completed_cases": idx,
            "guard_stats": guard.stats() if guard and use_guard else {},
        }
        print(f"  checkpoint {label}: {idx}/{total} cases")
        save_checkpoint()

    elapsed = time.time() - t0
    summary = results[label]["summary"]
    for cond, splits in summary.items():
        c = splits.get("clean", {}); p = splits.get("poisoned", {})
        short = cond.split("_")[0][:12]
        print(f"  [{short:12}] BCU-clean={c.get('BenignCompletionUnderAttack',0):.4f}  "
              f"ASR={p.get('ASR',0):.4f}  BCU-pois={p.get('BenignCompletionUnderAttack',0):.4f}  "
              f"n={p.get('n_cases',0)}")
    if guard and use_guard:
        s = guard.stats()
        print(f"  Guard stats: api_calls={s['api_calls']}  cache_hit_rate={s['cache_hit_rate']:.2f}  "
              f"quarantine={s['quarantine_count']}  directives={s['directive_count']}")
    results[label]["elapsed_s"] = round(elapsed, 1)
    print(f"  ({elapsed:.1f}s)\n")
    save_checkpoint()

# ── LoCoMo: regex-only (baseline, matches RESULTS.md) ────────────────────────
if not ARGS.skip_locomo:
    print("=== LoCoMo: H1 regex-only (existing baseline) ===")
    hp_regex = dict(hp_base, enable_llm_write_guard=False)
    run_and_record("locomo_regex_only", locomo_cases, run_mma_bench_eval,
                   CONDITIONS, hp_regex, ATTACKS, use_guard=False)

    # ── LoCoMo: LLM guard (semantic classification) ───────────────────────────────
    print("=== LoCoMo: H1 + LLM write-time guard (semantic) ===")
    hp_guard = dict(hp_base, enable_llm_write_guard=True)
    run_and_record("locomo_llm_guard", locomo_cases, run_mma_bench_eval,
                   CONDITIONS, hp_guard, ATTACKS, use_guard=True)

# ── MM-BrowseComp: regex-only (existing baseline) ────────────────────────────
if mm_cases and not ARGS.skip_mm:
    print("=== MM-BrowseComp: H1 regex-only (existing baseline) ===")
    hp_mm_regex = dict(hp_base, enable_llm_write_guard=False,
                       enable_vision_augmentation=False, enable_cross_topic_split=False)
    run_and_record_casewise("mm_regex_only", mm_cases, run_mm_browsecomp_eval,
                            CONDITIONS, hp_mm_regex, ATTACKS, use_guard=False)

    # ── MM-BrowseComp: guard + vision augmentation ────────────────────────────
    print("=== MM-BrowseComp: H1 + LLM guard + vision augmentation ===")
    hp_mm_guard = dict(hp_base, enable_llm_write_guard=True,
                       enable_vision_augmentation=True, enable_cross_topic_split=False)
    run_and_record_casewise("mm_llm_guard_vision", mm_cases, run_mm_browsecomp_eval,
                            CONDITIONS, hp_mm_guard, ATTACKS, use_guard=True)

# ── Save ─────────────────────────────────────────────────────────────────────
save_checkpoint()

# ── ASCII table ───────────────────────────────────────────────────────────────
cond_short = {
    "MMA_RetrieveTimeReliabilityScoring_Baseline": "MMA (retrieve-only)",
    "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation": "H1 ConstructorGuard",
}
lines = ["=" * 105]
lines.append("  WARP with LLM Write-Time Guard (claude-haiku) + Vision Augmentation")
lines.append("  H_GUARD: Guard achieves equal/lower ASR on LoCoMo; higher BCU-clean on MM-BrowseComp")
lines.append("=" * 105)
lines.append(f"  {'Experiment':<35} {'Condition':<22} {'BCU-clean':>10} {'ASR':>8} {'BCU-pois':>10} {'n':>6}")
lines.append("-" * 105)

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

lines.append("=" * 105)
lines.append("")
lines.append("  LLM guard architectural role:")
lines.append("  - Stateless Shadow Classifier: prompt = 'classify DATA|DIRECTIVE|METADATA', never 'execute'")
lines.append("  - Catches natural-language directives that evade regex (e.g. 'From now on, always...')")
lines.append("  - Vision augmentation: claude-haiku extracts jersey numbers, scores, stadium IDs from images")
lines.append("  - ConstructorGuard rewrite: DIRECTIVE → EVIDENCE_ONLY third-person description before audit")
lines.append("=" * 105)

table = "\n".join(lines)
print("\n" + table)
with open(OUT_TABLE, "w") as f:
    f.write(table + "\n")

print(f"\nSaved: {OUT}")
print(f"Saved: {OUT_TABLE}")
if guard:
    print(f"Final guard stats: {guard.stats()}")
