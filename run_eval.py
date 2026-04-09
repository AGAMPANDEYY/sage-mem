"""
run_eval.py — Evaluate write-time memory defenses on LoCoMo and MM-BrowseComp

Usage:
    python3 run_eval.py
    python3 run_eval.py --seeds 0 1 2 --qa-per-case 20 --out results/my_run.json
    python3 run_eval.py --quick        # single seed, 5 QA per case (~30 seconds)

Output:
    Prints per-condition BCU and ASR table.
    Saves full results to --out file (default: results/mma_eval_results.json).
"""

import argparse
import copy
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Allow running from repo root: src/ is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore

from mma_bench_suite import (
    ALL_ATTACK_SUITE,
    ABR_CONDITION,
    BROWSING_TRUST_PRIOR_CONDITION,
    MAIN_ATTACK_SUITE,
    MM_BROWSECOMP_ATTACK_SUITE,
    MMA_BENCH_CONDITIONS,
    MM_BROWSECOMP_PATH,
    TRUSTED_USER_STRESS_ATTACKS,
    aggregate_eval_metrics,
    load_mma_bench_cases,
    load_mm_browsecomp_cases,
    run_mma_bench_eval,
    run_mm_browsecomp_eval,
)
from mma_bench_suite import MAX_QA_PER_CASE as _DEFAULT_QA
from procedural import train_procedural_detector
from utils import set_all_seeds

# Default config path — override with --config-path
_DEFAULT_CONFIG_PATH = Path("configs/default_trust_config.json")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Write-time memory poisoning evaluation")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds (default: 0 1 2)")
    p.add_argument("--qa-per-case", type=int, default=_DEFAULT_QA,
                   help=f"Max QA pairs evaluated per LoCoMo case (default: {_DEFAULT_QA})")
    p.add_argument("--attack-suite", choices=["main", "trusted_user_stress", "all", "mm_browsecomp"], default="main",
                   help="Attack bundle to run when --attacks is not provided (default: main); "
                        "mm_browsecomp uses browsing-native attacks only (no self_summary/tool_echo)")
    p.add_argument("--mm-splits", nargs="+", choices=["clean", "poisoned"], default=None,
                   help="Which MM-BrowseComp splits to run (default: clean poisoned). "
                        "Use --mm-splits clean for clean-only, --mm-splits poisoned for adversarial-only")
    p.add_argument("--attacks", nargs="+", default=None,
                   help="Explicit attack types to inject in poisoned split; overrides --attack-suite")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Subset of conditions to run (default: all 5)")
    p.add_argument("--case-limit", type=int, default=None,
                   help="Optional limit on number of benchmark cases to run")
    p.add_argument("--case-fraction", type=float, default=None,
                   help="Optional random fraction of cases to evaluate, sampled without replacement (e.g. 0.2)")
    p.add_argument("--case-sample-seed", type=int, default=17,
                   help="Deterministic sampling seed used with --case-fraction")
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Optional specific case_ids/sample_ids to run")
    p.add_argument("--out", type=str, default="results/mma_eval_results.json",
                   help="Output JSON file (default: mma_eval_results.json)")
    p.add_argument("--enable-locomo-multimodal", action="store_true",
                   help="Enable the synthetic multimodal LoCoMo extension")
    p.add_argument("--multimodal-robustness-mode", choices=["none", "missing", "noisy", "missing_or_noisy"], default="none",
                   help="Apply missing/noisy perturbations to benign multimodal observations in the LoCoMo extension")
    p.add_argument("--multimodal-robustness-rate", type=float, default=0.0,
                   help="Fraction of multimodal observations to perturb under the selected robustness mode")
    p.add_argument("--vision-caption-mode", choices=["synthetic", "openai"], default="synthetic",
                   help="How to create vision_caption turns for the LoCoMo multimodal extension")
    p.add_argument("--vision-model", type=str, default="gpt-4o-mini",
                   help="Vision-capable OpenAI model to use when --vision-caption-mode openai")
    p.add_argument("--vision-cache-dir", type=str, default=".cache/openai_vision_captions",
                   help="Local cache directory for OpenAI VLM caption outputs")
    p.add_argument("--multimodal-adversary-mode", choices=["heuristic", "openai"], default="heuristic",
                   help="How to generate contradictory multimodal observations for LoCoMo")
    p.add_argument("--adversary-model", type=str, default="gpt-4o-mini",
                   help="OpenAI model to use when --multimodal-adversary-mode openai")
    p.add_argument("--adversary-cache-dir", type=str, default=".cache/openai_multimodal_attacks",
                   help="Local cache directory for frozen OpenAI-generated multimodal contradictions")
    p.add_argument("--disable-cross-topic", action="store_true",
                   help="Disable the delayed-trigger cross-topic split")
    p.add_argument("--run-mm-browsecomp", action="store_true",
                   help="Run the MM-BrowseComp benchmark if a local case file is available")
    p.add_argument("--mm-only", action="store_true",
                   help="Run only MM-BrowseComp and skip the LoCoMo benchmark")
    p.add_argument("--mm-browsecomp-path", type=str, default=None,
                   help=f"Path to official or augmented MM-BrowseComp JSONL (default: {MM_BROWSECOMP_PATH})")
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: seed=0 only, 5 QA per case")
    p.add_argument("--config-path", type=Path, default=None,
                   help=f"Path to SAGEMem trust config JSON (default: {_DEFAULT_CONFIG_PATH})")
    p.add_argument("--llm-eval", action="store_true",
                   help="Enable LLM-based behavioral ASR and answer entailment judges (adds API cost)")
    p.add_argument("--log-retrieved-source-types", action="store_true",
                   help="Log per-item source types for offline calibration pipeline")
    p.add_argument("--position-mode", choices=["random", "early", "late", "post_consolidation"],
                   default="random", help="Attack injection position mode (default: random)")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel workers across independent (seed, condition) units (default: 4)")
    p.add_argument("--sage-v2", action="store_true",
                   help="Include SAGEMemV2 condition (Bayesian trust + consistency graph + anomaly detection)")
    p.add_argument("--include-v2-ablations", action="store_true",
                   help="Include SAGEMemV2 component ablations (NoBayes / NoAnomaly / NoConsistency)")
    p.add_argument("--include-browsing-prior", action="store_true",
                   help="Include SAGEMemV2 browsing-context trust prior condition (H5, keyword-based) for web/tool observations")
    p.add_argument("--include-abr", action="store_true",
                   help="Include SAGEMemV2_ABR condition (H6): vocabulary-agnostic composite suspicion scorer for browser observations")
    p.add_argument("--resume", action="store_true",
                   help="Resume from an existing --out JSON by skipping completed (benchmark, seed, condition, split, case) units")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Checkpoint partial raw results to --out every N completed case units (default: 25)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_results_table(summary: dict, *, title: str) -> None:
    cond_short = {
        "ShortContext_NoLongTermMemory_Baseline":                     "ShortContext",
        "MMA_RetrieveTimeReliabilityScoring_Baseline":                  "MMA (baseline)",
        "Mem0_Platform_Baseline":                                      "mem0",
        "RecursiveSummarizationConsolidation_NoConstructorGuard":        "RSum (no guard)",
        "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation": "H1 ConstructorGuard",
        "SAGEMem_SourceAttestedGuardedEpisodicMemory":                  "SAGE-Mem v1",
        "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect":       "SAGE-Mem v2",
        "SAGEMemV2_BrowsingTrustPrior":                                 "v2 H5 BrowsePrior",
        "SAGEMemV2_ABR":                                                "v2 H6 ABR",
        "SAGEMemV2_NoBayes":                                            "v2 NoBayes",
        "SAGEMemV2_NoAnomaly":                                          "v2 NoAnom",
        "SAGEMemV2_NoConsistency":                                      "v2 NoCons",
        "MonotoneProvenanceLedger_ConservativeTrustScoring":             "H2 MonotoneLedger",
        "RiskSensitiveToolActionFirewall_CorroborateOrConfirm":          "H3 ActionFirewall",
    }

    has_cross = any(bool(splits.get("poisoned_cross_topic")) for splits in summary.values())
    has_behavioral = any(
        bool(splits.get("poisoned", {}).get("ASR_behavioral") is not None)
        for splits in summary.values()
    )
    has_poisoned = any(bool(splits.get("poisoned")) for splits in summary.values())

    if not has_poisoned:
        if Console is not None and Table is not None and Panel is not None:
            console = Console()
            table = Table(title=title, show_lines=False)
            table.add_column("Condition", style="bold cyan")
            table.add_column("BCU clean", justify="right")
            table.add_column("Answer rate", justify="right")
            table.add_column("Write ASR", justify="right")
            table.add_column("Belief ASR", justify="right")
            table.add_column("Retrieval", justify="right")
            table.add_column("FalseBel", justify="right")
            table.add_column("n", justify="right")
            for cond, splits in summary.items():
                name = cond_short.get(cond, cond[:26])
                clean = splits.get("clean", {})
                table.add_row(
                    name,
                    f"{clean.get('BenignCompletionUnderAttack', float('nan')):.4f}",
                    f"{clean.get('answered_rate', float('nan')):.4f}",
                    f"{clean.get('attack_write_admission_rate', float('nan')):.4f}",
                    f"{clean.get('attack_belief_formation_rate', float('nan')):.4f}",
                    f"{clean.get('attack_retrieval_rate', float('nan')):.4f}",
                    f"{clean.get('false_belief_rate', float('nan')):.4f}",
                    str(clean.get("n_cases", 0)),
                )
            console.print(table)
            console.print(
                Panel.fit(
                    "Clean-only run: no poisoned split was evaluated.\n"
                    "BCU clean = BenignCompletionUnderAttack on clean cases\n"
                    "Answer rate = fraction of QA pairs answered\n"
                    "Write/Belief/Retrieval/FalseBel should remain 0.0 on clean data",
                    title="Metrics",
                    border_style="dim",
                )
            )
            return

        print()
        print(title)
        print("-" * len(title))
        print(
            f"{'Condition':<26}  {'BCU clean':>10}  {'Answer rate':>12}  "
            f"{'Write ASR':>10}  {'Belief ASR':>10}  {'Retrieval':>10}  "
            f"{'FalseBel':>10}  {'n':>6}"
        )
        for cond, splits in summary.items():
            name = cond_short.get(cond, cond[:26])
            clean = splits.get("clean", {})
            print(
                f"  {name:<26}  "
                f"{clean.get('BenignCompletionUnderAttack', float('nan')):>10.4f}  "
                f"{clean.get('answered_rate', float('nan')):>12.4f}  "
                f"{clean.get('attack_write_admission_rate', float('nan')):>10.4f}  "
                f"{clean.get('attack_belief_formation_rate', float('nan')):>10.4f}  "
                f"{clean.get('attack_retrieval_rate', float('nan')):>10.4f}  "
                f"{clean.get('false_belief_rate', float('nan')):>10.4f}  "
                f"{clean.get('n_cases', 0):>6}"
            )
        print("  Clean-only run: no poisoned split was evaluated.")
        return

    header = (
        f"{'Condition':<26}  "
        f"{'Write ASR':>10}  "
        f"{'Belief ASR':>10}  "
        f"{'Retrieval':>10}  "
        f"{'FalseBel':>10}  "
        f"{'BCU pois':>10}  "
        f"{'BCU clean':>10}  "
        f"{'ASR poison':>10}"
    )
    if has_behavioral:
        header += f"  {'ASR (LLM)':>10}"
    if has_cross:
        header += f"  {'ASR cross':>10}"
    header += f"  {'n':>6}"
    if Console is not None and Table is not None and Panel is not None:
        console = Console()
        table = Table(title=title, show_lines=False)
        table.add_column("Condition", style="bold cyan")
        table.add_column("Write ASR", justify="right")
        table.add_column("Belief ASR", justify="right")
        table.add_column("Retrieval", justify="right")
        table.add_column("FalseBel", justify="right")
        table.add_column("BCU poison", justify="right")
        table.add_column("BCU clean", justify="right")
        table.add_column("ASR poison", justify="right")
        if has_behavioral:
            table.add_column("ASR (LLM)", justify="right")
        if has_cross:
            table.add_column("ASR cross", justify="right")
        table.add_column("n", justify="right")

        for cond, splits in summary.items():
            name = cond_short.get(cond, cond[:26])
            clean = splits.get("clean", {})
            pois = splits.get("poisoned", {})
            row = [
                name,
                f"{pois.get('attack_write_admission_rate', float('nan')):.4f}",
                f"{pois.get('attack_belief_formation_rate', float('nan')):.4f}",
                f"{pois.get('attack_retrieval_rate', float('nan')):.4f}",
                f"{pois.get('false_belief_rate', float('nan')):.4f}",
                f"{pois.get('BenignCompletionUnderAttack', float('nan')):.4f}",
                f"{clean.get('BenignCompletionUnderAttack', float('nan')):.4f}",
                f"{pois.get('ASR', float('nan')):.4f}",
            ]
            if has_behavioral:
                asr_llm = pois.get("ASR_behavioral", float("nan"))
                row.append("N/A" if isinstance(asr_llm, float) and math.isnan(asr_llm) else f"{asr_llm:.4f}")
            if has_cross:
                row.append(f"{splits.get('poisoned_cross_topic', {}).get('ASR', float('nan')):.4f}")
            row.append(str(pois.get("n_cases", 0)))
            table.add_row(*row)

        console.print(table)
        notes = (
            "Write ASR = attack_write_admission_rate (lower is better)\n"
            "Belief ASR = attack_belief_formation_rate (lower is better)\n"
            "Retrieval = attack_retrieval_rate (lower is better)\n"
            "FalseBel = false_belief_rate (lower is better)\n"
            "BCU = BenignCompletionUnderAttack (higher is better)\n"
            "ASR = Attack Success Rate (lower is better)"
        )
        if has_behavioral:
            notes += "\nASR (LLM) = Behavioral ASR via OpenAI judge"
        console.print(Panel.fit(notes, title="Metrics", border_style="dim"))
        return

    print()
    print("=" * len(header))
    print(title)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for cond, splits in summary.items():
        name = cond_short.get(cond, cond[:26])
        clean = splits.get("clean", {})
        pois = splits.get("poisoned", {})
        write_asr = pois.get("attack_write_admission_rate", float("nan"))
        belief_asr = pois.get("attack_belief_formation_rate", float("nan"))
        retrieval = pois.get("attack_retrieval_rate", float("nan"))
        false_bel = pois.get("false_belief_rate", float("nan"))
        bcu_p = pois.get("BenignCompletionUnderAttack", float("nan"))
        bcu_c = clean.get("BenignCompletionUnderAttack", float("nan"))
        asr_p = pois.get("ASR", float("nan"))
        line = (
            f"  {name:<26}  {write_asr:>10.4f}  {belief_asr:>10.4f}  "
            f"{retrieval:>10.4f}  {false_bel:>10.4f}  {bcu_p:>10.4f}  "
            f"{bcu_c:>10.4f}  {asr_p:>10.4f}"
        )
        if has_behavioral:
            asr_llm = pois.get("ASR_behavioral", float("nan"))
            line += f"  {asr_llm:>10.4f}" if not (isinstance(asr_llm, float) and math.isnan(asr_llm)) else f"  {'N/A':>10}"
        if has_cross:
            line += f"  {splits.get('poisoned_cross_topic', {}).get('ASR', float('nan')):>10.4f}"
        line += f"  {pois.get('n_cases', 0):>6}"
        print(line)
    print("=" * len(header))
    print("  Write ASR = attack_write_admission_rate (lower = better defense)")
    print("  Belief ASR = attack_belief_formation_rate (lower = better defense)")
    print("  Retrieval = attack_retrieval_rate (lower = better defense)")
    print("  FalseBel = false_belief_rate (lower = better defense)")
    print("  BCU = BenignCompletionUnderAttack (higher = better defense)")
    print("  ASR = Attack Success Rate (lower = better defense)")
    if has_behavioral:
        print("  ASR (LLM) = Behavioral ASR via OpenAI judge")
    print()

    # ── Latency table ──────────────────────────────────────────────────────
    lat_header = f"  {'Condition':<26}  {'Write ms':>10}  {'Retr ms':>10}  {'Items':>8}  {'Audit':>8}"
    has_latency = any(
        "write_avg_ms" in splits.get("clean", {}) or "write_avg_ms" in splits.get("poisoned", {})
        for splits in summary.values()
    )
    if has_latency:
        print(lat_header)
        print("  " + "-" * (len(lat_header) - 2))
        for cond, splits in summary.items():
            name = cond_short.get(cond, cond[:26])
            d = splits.get("poisoned") or splits.get("clean") or {}
            w = d.get("write_avg_ms", float("nan"))
            r = d.get("retrieve_avg_ms", float("nan"))
            items = d.get("items_in_memory_avg", float("nan"))
            audit = d.get("items_in_audit_avg", float("nan"))
            print(f"  {name:<26}  {w:>10.3f}  {r:>10.3f}  {items:>8.1f}  {audit:>8.1f}")
        print("  Write ms = avg ms per write() call | Retr ms = avg ms per retrieve() call")
        print("  Items = avg planning memory size | Audit = avg quarantine store size")
        print()


def _sample_cases(cases: list[dict], *, fraction: float | None, sample_seed: int) -> list[dict]:
    if fraction is None:
        return list(cases)
    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise SystemExit("--case-fraction must be in the range (0, 1]")
    if frac >= 1.0:
        return list(cases)
    import random
    n_total = len(cases)
    n_keep = max(1, math.ceil(n_total * frac))
    chosen = sorted(random.Random(sample_seed).sample(range(n_total), n_keep))
    return [cases[i] for i in chosen]


def _normalize_raw_results(
    raw: dict | None,
    *,
    conditions: list[str],
    split_names: list[str],
) -> dict:
    return {
        cond: {split: list((raw or {}).get(cond, {}).get(split, [])) for split in split_names}
        for cond in conditions
    }


def _load_resume_state(
    out_path: Path,
    *,
    conditions: list[str],
    locomo_splits: list[str],
    mm_splits: list[str],
) -> dict:
    state = {
        "completed_units": [],
        "locomo_raw": _normalize_raw_results({}, conditions=conditions, split_names=locomo_splits),
        "mm_raw": _normalize_raw_results({}, conditions=conditions, split_names=mm_splits),
    }
    if not out_path.exists():
        return state
    with open(out_path) as f:
        payload = json.load(f)
    resume_state = payload.get("resume_state", {})
    state["completed_units"] = list(resume_state.get("completed_units", []))
    benchmarks = payload.get("benchmarks", {})
    state["locomo_raw"] = _normalize_raw_results(
        benchmarks.get("locomo", {}).get("raw", {}),
        conditions=conditions,
        split_names=locomo_splits,
    )
    state["mm_raw"] = _normalize_raw_results(
        benchmarks.get("mm_browsecomp", {}).get("raw", {}),
        conditions=conditions,
        split_names=mm_splits,
    )
    return state


def _build_output_payload(
    *,
    locomo_raw: dict | None,
    locomo_summary: dict | None,
    mm_raw: dict | None,
    mm_summary: dict | None,
    args,
    conditions: list[str],
    config_path: Path,
    sage_cfg_loaded: bool,
    hp: dict,
    elapsed: float,
    completed_units: list[dict],
    device,
) -> dict:
    locomo_raw = locomo_raw or {}
    locomo_summary = locomo_summary or {}
    mm_raw = mm_raw or None
    mm_summary = mm_summary or None
    top_level_summary = locomo_summary if locomo_summary else (mm_summary or {})

    output = {
        "benchmarks": {
            "locomo": {
                "summary": locomo_summary,
                "raw": locomo_raw,
                "raw_sample_count": {
                    c: {s: len(v) for s, v in splits.items()}
                    for c, splits in locomo_raw.items()
                },
            },
        },
        "summary": top_level_summary,
        "config": {
            "seeds": args.seeds,
            "qa_per_case": args.qa_per_case,
            "attack_types": args.attacks,
            "conditions": conditions,
            "case_fraction": args.case_fraction,
            "case_sample_seed": args.case_sample_seed,
            "device": str(device),
            "enable_locomo_multimodal": bool(args.enable_locomo_multimodal),
            "vision_caption_mode": args.vision_caption_mode,
            "vision_model": args.vision_model,
            "vision_cache_dir": args.vision_cache_dir,
            "multimodal_adversary_mode": args.multimodal_adversary_mode,
            "adversary_model": args.adversary_model,
            "adversary_cache_dir": args.adversary_cache_dir,
            "enable_cross_topic_split": not bool(args.disable_cross_topic),
            "run_mm_browsecomp": bool(args.run_mm_browsecomp),
            "mm_only": bool(args.mm_only),
            "mm_browsecomp_path": str(args.mm_browsecomp_path or MM_BROWSECOMP_PATH),
            "mm_splits": args.mm_splits or ["clean", "poisoned"],
            "multimodal_turn_rate": hp["multimodal_turn_rate"],
            "ocr_noise_prob_low": hp["ocr_noise_prob_low"],
            "ocr_noise_prob_high": hp["ocr_noise_prob_high"],
            "h2_write_trust_threshold": hp["h2_write_trust_threshold"],
            "h3_write_trust_threshold": hp["h3_write_trust_threshold"],
            "sage_chain_decay": hp["sage_chain_decay"],
            "sage_write_trust_threshold": hp["sage_write_trust_threshold"],
            "sage_config_path": str(config_path),
            "sage_config_loaded": sage_cfg_loaded,
            "llm_eval": bool(args.llm_eval),
            "position_mode": args.position_mode,
            "max_workers": int(args.max_workers),
            "sage_v2": bool(args.sage_v2),
            "runtime_sec": round(elapsed, 2),
        },
        "raw_sample_count": {
            c: {s: len(v) for s, v in splits.items()}
            for c, splits in locomo_raw.items()
        },
        "resume_state": {
            "completed_units": completed_units,
        },
    }
    if mm_summary is not None and mm_raw is not None:
        output["benchmarks"]["mm_browsecomp"] = {
            "summary": mm_summary,
            "raw": mm_raw,
            "raw_sample_count": {
                c: {s: len(v) for s, v in splits.items()}
                for c, splits in mm_raw.items()
            },
        }
    return output


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    args = parse_args()

    if args.attacks is None:
        if args.attack_suite == "main":
            args.attacks = list(MAIN_ATTACK_SUITE)
        elif args.attack_suite == "trusted_user_stress":
            args.attacks = list(TRUSTED_USER_STRESS_ATTACKS)
        elif args.attack_suite == "mm_browsecomp":
            args.attacks = list(MM_BROWSECOMP_ATTACK_SUITE)
        else:
            args.attacks = list(ALL_ATTACK_SUITE)

    if args.quick:
        args.seeds = [0]
        args.qa_per_case = 5
        print("[quick mode] seed=0, 5 QA per case")
    args.max_workers = max(1, min(int(args.max_workers), multiprocessing.cpu_count()))

    out_path = Path(args.out)

    if args.enable_locomo_multimodal and args.vision_caption_mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required when --vision-caption-mode openai is used "
            "with --enable-locomo-multimodal"
        )
    if args.enable_locomo_multimodal and args.multimodal_adversary_mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required when --multimodal-adversary-mode openai is used "
            "with --enable-locomo-multimodal"
        )

    # Patch MAX_QA_PER_CASE at runtime
    import mma_bench_suite
    mma_bench_suite.MAX_QA_PER_CASE = args.qa_per_case

    # ── Load SAGEMem trust config ────────────────────────────────────────────
    sage_cfg = None
    config_path = args.config_path or _DEFAULT_CONFIG_PATH
    if config_path.exists():
        try:
            from config import SAGEMemConfig
            sage_cfg = SAGEMemConfig.from_file(config_path)
            print(f"[config] Loaded SAGEMemConfig from {config_path}")
        except Exception as exc:
            print(f"[config] WARNING: could not load config from {config_path}: {exc}")
    else:
        print(f"[config] No config file found at {config_path}. SAGEMemV2 will use defaults.")

    # ── Build SAGEMemV2 components ───────────────────────────────────────────
    # BayesianChannelTrust and SessionAnomalyDetector can be pre-built here.
    # MultiTurnConsistencyGraph and CorrectionPlausibilityScorer need an embedder,
    # so they are built per-case inside build_mma_condition.
    sage_bayes_trust = None
    sage_anomaly_detector = None
    if sage_cfg is not None:
        try:
            from trust_calibration import BayesianChannelTrust
            sage_bayes_trust = BayesianChannelTrust(sage_cfg)
        except ImportError:
            print("[config] WARNING: trust_calibration not available; SAGEMemV2 will skip Bayesian trust")
        try:
            from anomaly_detector import SessionAnomalyDetector
            sage_anomaly_detector = SessionAnomalyDetector(sage_cfg, embedding_dim=256)
        except ImportError:
            print("[config] WARNING: anomaly_detector not available; SAGEMemV2 will skip anomaly detection")

    # ── Build LLM evaluation judges (optional) ───────────────────────────────
    eval_judge = None
    answer_judge = None
    if args.llm_eval:
        if sage_cfg is None:
            print("[llm-eval] WARNING: --llm-eval requires a valid config; judges disabled")
        else:
            try:
                from eval_judge import BehavioralAttackJudge, LLMAnswerJudge
                eval_judge = BehavioralAttackJudge(sage_cfg)
                answer_judge = LLMAnswerJudge(sage_cfg)
                print(
                    "[llm-eval] LLM judges initialized "
                    f"(backend: {eval_judge.backend}, model: {eval_judge.model_name})"
                )
            except Exception as exc:
                print(f"[llm-eval] WARNING: could not initialize LLM judges: {exc}")

    conditions = args.conditions or [
        c for c in MMA_BENCH_CONDITIONS
        if c not in {
            "Mem0_Platform_Baseline",
            "SAGEMem_SourceAttestedGuardedEpisodicMemory",
            "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
            BROWSING_TRUST_PRIOR_CONDITION,
        }
    ]
    if args.sage_v2 and "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect" not in conditions:
        conditions.append("SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect")
    if args.include_v2_ablations:
        for cond in [
            "SAGEMemV2_NoBayes",
            "SAGEMemV2_NoAnomaly",
            "SAGEMemV2_NoConsistency",
        ]:
            if cond not in conditions:
                conditions.append(cond)
    if args.include_browsing_prior and BROWSING_TRUST_PRIOR_CONDITION not in conditions:
        conditions.append(BROWSING_TRUST_PRIOR_CONDITION)
    if args.include_abr and ABR_CONDITION not in conditions:
        conditions.append(ABR_CONDITION)

    locomo_splits = ["clean", "poisoned"]
    if not args.disable_cross_topic:
        locomo_splits.append("poisoned_cross_topic")
    mm_splits = ["clean", "poisoned"]

    resume_payload = _load_resume_state(
        out_path,
        conditions=conditions,
        locomo_splits=locomo_splits,
        mm_splits=mm_splits,
    ) if args.resume else None
    completed_units = set()
    completed_units_serialized = []
    if resume_payload is not None:
        completed_units_serialized = list(resume_payload["completed_units"])
        completed_units = {
            (
                str(unit["benchmark"]),
                int(unit["seed"]),
                str(unit["condition"]),
                str(unit["split"]),
                str(unit["case_id"]),
            )
            for unit in completed_units_serialized
        }
        print(f"[resume] Loaded {len(completed_units)} completed case units from {out_path}")

    device = pick_device()
    print(f"Device: {device} | Seeds: {args.seeds} | QA/case: {args.qa_per_case} | "
          f"Attacks: {args.attacks} | Position: {args.position_mode}"
          + (" | LLM-eval: ON" if args.llm_eval else ""))

    # Train procedural detector
    set_all_seeds(0)
    hp = {
        "embed_dim": 256, "top_k": 8, "consolidation_period_K": 4, "keep_last_M_raw": 4,
        "trigger_step": 10, "poison_rate": 0.30,
        "ocr_noise_prob_low": 0.05, "ocr_noise_prob_high": 0.25,
        "multimodal_turn_rate": 0.20,
        "vision_caption_mode": args.vision_caption_mode,
        "vision_model": args.vision_model,
        "vision_cache_dir": args.vision_cache_dir,
        "vision_max_output_tokens": 96,
        "multimodal_adversary_mode": args.multimodal_adversary_mode,
        "adversary_model": args.adversary_model,
        "adversary_cache_dir": args.adversary_cache_dir,
        "adversary_max_output_tokens": 160,
        "trust_accept_threshold": 0.55, "abstain_on_low_trust": False,
        "mma_w_source": 0.5, "mma_w_decay": 0.2, "mma_w_consensus": 0.3,
        "mma_decay_half_life_steps": 50,
        "short_context_keep_last_k": 8,
        "procedural_classifier_threshold": 0.60, "quarantine_on_fail": True,
        "chain_decay": 0.85, "independence_bonus": 1.2, "max_chain_len": 5,
        "sage_chain_decay": 0.90, "sage_write_trust_threshold": 0.25,
        "h2_write_trust_threshold": 0.25, "h3_write_trust_threshold": 0.20,
        "tool_attestation_required": True,
        "high_risk_requires_corroboration": 2, "require_user_confirmation": True,
        "max_confirmation_rate": 0.25, "clean_confirmation_token_probability": 0.10,
        "procclf_dim": 256, "procclf_lr": 1e-4, "procclf_weight_decay": 0.01,
        "procclf_epochs": 3, "procclf_batch_size": 32, "procclf_max_grad_norm": 1.0,
        "label_stream_n": 5000,
        "mem0_api_key": None,
        "mem0_infer": True,
        "enable_multimodal_locomo": bool(args.enable_locomo_multimodal),
        "multimodal_robustness_mode": str(args.multimodal_robustness_mode),
        "multimodal_robustness_rate": float(args.multimodal_robustness_rate),
        "enable_cross_topic_split": not bool(args.disable_cross_topic),
        # SAGEMem v2 components — passed into build_mma_condition for SAGEMemV2 condition
        # sage_consistency_graph and sage_correction_scorer are built per-case inside
        # build_mma_condition (they require the per-seed embedder instance)
        "sage_cfg": sage_cfg,
        "sage_bayes_trust": sage_bayes_trust,
        "sage_anomaly_detector": sage_anomaly_detector,
        "sage_consistency_graph": None,   # built per-case in build_mma_condition
        "sage_correction_scorer": None,   # built per-case in build_mma_condition
    }
    detector = train_procedural_detector(
        seed=0, dim=hp["procclf_dim"], device=device,
        lr=hp["procclf_lr"], weight_decay=hp["procclf_weight_decay"],
        epochs=hp["procclf_epochs"], batch_size=hp["procclf_batch_size"],
        max_grad_norm=hp["procclf_max_grad_norm"],
    )

    # Load and run LoCoMo unless this invocation is MM-BrowseComp-only.
    locomo_cases = []
    if not args.mm_only:
        locomo_cases = load_mma_bench_cases()
        if args.case_ids:
            wanted = {str(x) for x in args.case_ids}
            locomo_cases = [c for c in locomo_cases if str(c.get("case_id")) in wanted]
        locomo_cases = _sample_cases(
            locomo_cases,
            fraction=args.case_fraction,
            sample_seed=args.case_sample_seed,
        )
        if args.case_limit is not None:
            locomo_cases = locomo_cases[: max(0, int(args.case_limit))]
        print(f"Loaded {len(locomo_cases)} LoCoMo cases")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_state = {"n_completed": 0}

    locomo_raw_existing = copy.deepcopy(resume_payload["locomo_raw"]) if resume_payload is not None else None
    mm_raw_existing = copy.deepcopy(resume_payload["mm_raw"]) if resume_payload is not None else None
    locomo_raw = None
    mm_raw = None

    def _checkpoint_save() -> None:
        if locomo_raw is None:
            return
        locomo_summary_now = aggregate_eval_metrics(locomo_raw)
        mm_summary_now = aggregate_eval_metrics(mm_raw) if mm_raw is not None else None
        payload = _build_output_payload(
            locomo_raw=locomo_raw,
            locomo_summary=locomo_summary_now,
            mm_raw=mm_raw,
            mm_summary=mm_summary_now,
            args=args,
            conditions=conditions,
            config_path=config_path,
            sage_cfg_loaded=sage_cfg is not None,
            hp=hp,
            elapsed=0.0,
            completed_units=completed_units_serialized,
            device=device,
        )
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _progress_callback(*, benchmark_label: str, seed: int, condition: str, split: str, case_id: str, results: dict) -> None:
        nonlocal locomo_raw, mm_raw
        if benchmark_label == "locomo":
            locomo_raw = results
        elif benchmark_label == "mm_browsecomp":
            mm_raw = results
        completed_units_serialized.append(
            {
                "benchmark": benchmark_label,
                "seed": int(seed),
                "condition": condition,
                "split": split,
                "case_id": case_id,
            }
        )
        checkpoint_state["n_completed"] += 1
        if checkpoint_state["n_completed"] % max(1, int(args.checkpoint_every)) == 0:
            _checkpoint_save()

    locomo_summary = {}
    locomo_elapsed = 0.0
    if locomo_cases:
        t0 = time.time()
        locomo_raw = run_mma_bench_eval(
            cases=locomo_cases,
            conditions=conditions,
            detector=detector,
            hp=hp,
            seeds=args.seeds,
            attack_types=args.attacks,
            eval_judge=eval_judge,
            answer_judge=answer_judge,
            log_retrieved_source_types=bool(args.log_retrieved_source_types),
            position_mode=args.position_mode,
            existing_results=locomo_raw_existing,
            completed_units=completed_units,
            progress_callback=_progress_callback,
            max_workers=args.max_workers,
        )
        locomo_elapsed = time.time() - t0
        locomo_summary = aggregate_eval_metrics(locomo_raw)
        print_results_table(locomo_summary, title="LoCoMo Write-Time Poisoning Results")
        print(f"LoCoMo runtime: {locomo_elapsed:.1f}s")

    mm_summary = None
    mm_elapsed = 0.0
    if args.run_mm_browsecomp:
        mm_path = Path(args.mm_browsecomp_path) if args.mm_browsecomp_path else MM_BROWSECOMP_PATH
        mm_cases = load_mm_browsecomp_cases(mm_path)
        if args.case_ids:
            wanted = {str(x) for x in args.case_ids}
            mm_cases = [c for c in mm_cases if str(c.get("case_id")) in wanted]
        mm_cases = _sample_cases(
            mm_cases,
            fraction=args.case_fraction,
            sample_seed=args.case_sample_seed,
        )
        if args.case_limit is not None:
            mm_cases = mm_cases[: max(0, int(args.case_limit))]
        t1 = time.time()
        try:
            mm_raw = run_mm_browsecomp_eval(
                cases=mm_cases,
                conditions=conditions,
                detector=detector,
                hp=hp,
                seeds=args.seeds,
                splits=args.mm_splits if args.mm_splits else None,
                attack_types=args.attacks,
                eval_judge=eval_judge,
                answer_judge=answer_judge,
                log_retrieved_source_types=bool(args.log_retrieved_source_types),
                position_mode=args.position_mode,
                existing_results=mm_raw_existing,
                completed_units=completed_units,
                progress_callback=_progress_callback,
                max_workers=args.max_workers,
            )
        except ValueError as exc:
            raise SystemExit(f"MM-BrowseComp configuration error: {exc}") from exc
        mm_elapsed = time.time() - t1
        mm_summary = aggregate_eval_metrics(mm_raw)
        print_results_table(mm_summary, title="MM-BrowseComp Results")
        print(f"MM-BrowseComp runtime: {mm_elapsed:.1f}s")

    elapsed = locomo_elapsed + mm_elapsed
    print(f"Total runtime: {elapsed:.1f}s")

    # Save
    output = _build_output_payload(
        locomo_raw=locomo_raw,
        locomo_summary=locomo_summary,
        mm_raw=mm_raw,
        mm_summary=mm_summary,
        args=args,
        conditions=conditions,
        config_path=config_path,
        sage_cfg_loaded=sage_cfg is not None,
        hp=hp,
        elapsed=elapsed,
        completed_units=completed_units_serialized,
        device=device,
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
