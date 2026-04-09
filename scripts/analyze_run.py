#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except Exception:  # pragma: no cover - fallback when rich is unavailable
    Console = None
    Panel = None
    Table = None
    Text = None


COND_SHORT = {
    "ShortContext_NoLongTermMemory_Baseline": "ShortContext",
    "MMA_RetrieveTimeReliabilityScoring_Baseline": "MMA",
    "Mem0_Platform_Baseline": "mem0",
    "RecursiveSummarizationConsolidation_NoConstructorGuard": "RSum",
    "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation": "H1",
    "SAGEMem_SourceAttestedGuardedEpisodicMemory": "SAGE-Mem v1",
    "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect": "SAGE-Mem v2",
    "SAGEMemV2_NoBayes": "v2 NoBayes",
    "SAGEMemV2_NoAnomaly": "v2 NoAnom",
    "SAGEMemV2_NoConsistency": "v2 NoCons",
    "MonotoneProvenanceLedger_ConservativeTrustScoring": "H2",
    "RiskSensitiveToolActionFirewall_CorroborateOrConfirm": "H3",
}

FILE_TITLES = {
    "sagemem_main.json": "Main LoCoMo",
    "sagemem_main_llm.json": "Main LoCoMo + LLM Judge",
    "sagemem_vpi_llm.json": "VPI + LLM Judge",
    "sagemem_v2_ablations.json": "Main LoCoMo Ablations",
    "sagemem_multimodal_robustness.json": "Multimodal Robustness",
    "sagemem_multimodal_robustness_ablations.json": "Multimodal Robustness Ablations",
    "sagemem_multimodal_openai_frozen.json": "Frozen OpenAI Multimodal",
    "sagemem_mm_browsecomp.json": "MM-BrowseComp (clean + adversarial)",
    "sagemem_mm_browsecomp_clean.json": "MM-BrowseComp Clean",
    "sagemem_mm_browsecomp_adversarial.json": "MM-BrowseComp Adversarial",
    "sagemem_trusted_user_stress_llm.json": "Trusted-User Stress + LLM Judge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a results/<RUN_ID>/ folder and print Rich summary tables."
    )
    parser.add_argument("run_id", help="Run folder under results/, e.g. pilot_main_v1")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory containing run-id subfolders (default: results)",
    )
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def colorize_metric(metric: str, value: Any) -> str:
    if value is None or not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
        return "dim"
    high_good = {
        "BenignCompletionUnderAttack",
        "belief_traceability_score",
        # Higher quarantine on poisoned split = more attacks blocked = better defense
        "write_quarantine_per_case",
    }
    low_good = {
        "ASR",
        "ASR_behavioral",
        "false_belief_rate",
        "attack_retrieval_rate",
        "attack_write_admission_rate",
        "attack_belief_formation_rate",
    }
    if metric in high_good:
        return "green" if value >= 0.7 else "yellow" if value >= 0.35 else "red"
    if metric in low_good:
        return "green" if value <= 0.05 else "yellow" if value <= 0.2 else "red"
    return "none"


def discover_json_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("*.json") if p.is_file())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def render_metadata(console: Console, run_id: str, path: Path, data: dict[str, Any]) -> None:
    info = [
        f"[bold]Run ID:[/bold] {run_id}",
        f"[bold]File:[/bold] {path.name}",
    ]
    for key, label in (
        ("llm_eval", "LLM eval"),
        ("case_fraction", "Case fraction"),
        ("case_sample_seed", "Sample seed"),
        ("enable_locomo_multimodal", "LoCoMo multimodal"),
        ("run_mm_browsecomp", "MM-BrowseComp"),
    ):
        if key in data:
            info.append(f"[bold]{label}:[/bold] {data.get(key)}")
    if Panel is not None:
        console.print(Panel.fit("\n".join(info), title="Run Metadata", border_style="blue"))
    else:
        console.print("\n".join(info))


def render_summary_table(console: Console, title: str, summary: dict[str, Any]) -> None:
    metrics = [
        ("clean", "BenignCompletionUnderAttack", "BCU clean"),
        ("poisoned", "BenignCompletionUnderAttack", "BCU poison"),
        ("poisoned", "attack_write_admission_rate", "Write ASR"),
        ("poisoned", "attack_belief_formation_rate", "Belief ASR"),
        ("poisoned", "ASR", "ASR"),
        ("poisoned", "ASR_behavioral", "ASR LLM"),
        ("poisoned", "answer_consistent_rate_llm", "Answer LLM"),
        ("poisoned", "false_belief_rate", "False belief"),
        ("poisoned", "belief_traceability_score", "Traceability"),
        ("poisoned", "attack_retrieval_rate", "Retrieval"),
        ("poisoned", "write_quarantine_per_case", "Write q/case"),
        ("poisoned", "n_cases", "n"),
    ]
    has_cross = any(bool(splits.get("poisoned_cross_topic")) for splits in summary.values())

    table = Table(title=title, show_lines=False, header_style="bold magenta")
    table.add_column("Condition", style="bold cyan")
    for _, _, label in metrics:
        table.add_column(label, justify="right")
    if has_cross:
        table.add_column("ASR cross", justify="right")

    for condition, splits in summary.items():
        row: list[str | Text] = [COND_SHORT.get(condition, condition)]
        for split, metric, _ in metrics:
            value = splits.get(split, {}).get(metric)
            style = colorize_metric(metric, value)
            cell = Text(fmt(value), style=style) if Text is not None else fmt(value)
            row.append(cell)
        if has_cross:
            cross = splits.get("poisoned_cross_topic", {}).get("ASR")
            style = colorize_metric("ASR", cross)
            row.append(Text(fmt(cross), style=style) if Text is not None else fmt(cross))
        table.add_row(*row)

    console.print(table)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.results_dir) / args.run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run folder not found: {run_dir}")

    files = discover_json_files(run_dir)
    if not files:
        raise SystemExit(f"No JSON files found in {run_dir}")

    console = Console() if Console is not None else None
    if console is None:
        raise SystemExit("rich is required for scripts/analyze_run.py")

    title = Panel.fit(
        f"[bold]Run analysis[/bold]\n{run_dir}",
        border_style="green",
    )
    console.print(title)

    summary_files = 0
    for path in files:
        data = load_json(path)
        if not data or not isinstance(data.get("summary"), dict):
            continue
        summary_files += 1
        render_metadata(console, args.run_id, path, data)
        render_summary_table(console, FILE_TITLES.get(path.name, path.name), data["summary"])
        console.print()

    if summary_files == 0:
        raise SystemExit(f"No summary-bearing result JSON files found in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
