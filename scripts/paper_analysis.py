#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


COND_SHORT = {
    "ShortContext_NoLongTermMemory_Baseline": "ShortContext",
    "MMA_RetrieveTimeReliabilityScoring_Baseline": "MMA",
    "Mem0_Platform_Baseline": "mem0",
    "RecursiveSummarizationConsolidation_NoConstructorGuard": "RSum",
    "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation": "H1",
    "SAGEMem_SourceAttestedGuardedEpisodicMemory": "SAGE-Mem v1",
    "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect": "SAGE-Mem v2",
    "SAGEMemV2_BrowsingTrustPrior": "H5",
    "SAGEMemV2_ABR": "H6",
    "SAGEMemV2_NoBayes": "v2 NoBayes",
    "SAGEMemV2_NoAnomaly": "v2 NoAnom",
    "SAGEMemV2_NoConsistency": "v2 NoCons",
    "MonotoneProvenanceLedger_ConservativeTrustScoring": "H2",
    "RiskSensitiveToolActionFirewall_CorroborateOrConfirm": "H3",
}

PAPER_FILES = {
    "main": "sagemem_main_llm.json",
    "vpi": "sagemem_vpi_llm.json",
    "mmrobust": "sagemem_multimodal_robustness_ablations.json",
    "ablations": "sagemem_v2_ablations.json",
    "browse_clean": "sagemem_mm_browsecomp_abr_clean.json",
    "browse_adv": "sagemem_mm_browsecomp_abr_adversarial.json",
    "browse_clean_sem": "sagemem_mm_browsecomp_abr_clean_semantic.json",
    "browse_adv_sem": "sagemem_mm_browsecomp_abr_adversarial_semantic.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission-ready analysis from frozen paper artifacts.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("final_paper_results_20260410"),
        help="Directory containing frozen result JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/paper_submission_ready"),
        help="Output directory for CSV/Markdown/SVG artifacts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_summary(path: Path, benchmark: str) -> dict[str, Any]:
    data = load_json(path)
    return data["benchmarks"][benchmark]["summary"]


def select_rows(summary: dict[str, Any], split: str, conditions: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        metrics = summary.get(cond, {}).get(split, {})
        if not metrics:
            continue
        row = {
            "condition_id": cond,
            "condition": COND_SHORT.get(cond, cond),
            "BCU": metrics.get("BenignCompletionUnderAttack"),
            "WriteASR": metrics.get("attack_write_admission_rate"),
            "BeliefASR": metrics.get("attack_belief_formation_rate"),
            "Retrieval": metrics.get("attack_retrieval_rate"),
            "ASR": metrics.get("ASR"),
            "FalseBelief": metrics.get("false_belief_rate"),
            "n_cases": metrics.get("n_cases"),
            "n_qa_evals": metrics.get("n_qa_evals"),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def build_main_tables(results_dir: Path, out_dir: Path) -> None:
    main_summary = get_summary(results_dir / PAPER_FILES["main"], "locomo")
    main_conditions = [
        "ShortContext_NoLongTermMemory_Baseline",
        "MMA_RetrieveTimeReliabilityScoring_Baseline",
        "RecursiveSummarizationConsolidation_NoConstructorGuard",
        "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation",
        "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
    ]
    clean_rows = select_rows(main_summary, "clean", main_conditions)
    poison_rows = select_rows(main_summary, "poisoned", main_conditions)
    fieldnames = ["condition", "BCU", "WriteASR", "BeliefASR", "Retrieval", "ASR", "FalseBelief", "n_cases", "n_qa_evals"]
    write_csv(out_dir / "main_clean_table.csv", clean_rows, fieldnames)
    write_csv(out_dir / "main_poison_table.csv", poison_rows, fieldnames)


def build_browsing_tables(results_dir: Path, out_dir: Path) -> None:
    clean_summary = get_summary(results_dir / PAPER_FILES["browse_clean"], "mm_browsecomp")
    adv_summary = get_summary(results_dir / PAPER_FILES["browse_adv"], "mm_browsecomp")
    sem_clean_summary = get_summary(results_dir / PAPER_FILES["browse_clean_sem"], "mm_browsecomp")
    sem_adv_summary = get_summary(results_dir / PAPER_FILES["browse_adv_sem"], "mm_browsecomp")
    conditions = [
        "MMA_RetrieveTimeReliabilityScoring_Baseline",
        "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
        "SAGEMemV2_BrowsingTrustPrior",
        "SAGEMemV2_ABR",
    ]
    clean_rows = select_rows(clean_summary, "clean", conditions)
    adv_rows = select_rows(adv_summary, "poisoned", conditions)
    sem_adv_rows = select_rows(sem_adv_summary, "poisoned", conditions)
    sem_clean_rows = select_rows(sem_clean_summary, "clean", conditions)
    fieldnames = [
        "condition", "BCU", "WriteASR", "BeliefASR", "Retrieval", "ASR", "FalseBelief",
        "structured_claim_gate_fire_count_avg", "group_divergence_fire_count_avg",
        "group_outlier_score_avg", "memory_conflict_score_avg", "n_cases", "n_qa_evals",
    ]
    write_csv(out_dir / "browsing_clean_table.csv", clean_rows, fieldnames)
    write_csv(out_dir / "browsing_adversarial_table.csv", adv_rows, fieldnames)
    write_csv(out_dir / "browsing_clean_semantic_table.csv", sem_clean_rows, fieldnames)
    write_csv(out_dir / "browsing_adversarial_semantic_table.csv", sem_adv_rows, fieldnames)


def build_attack_proxy_tables(results_dir: Path, out_dir: Path) -> None:
    main_summary = get_summary(results_dir / PAPER_FILES["main"], "locomo")
    conditions = [
        "MMA_RetrieveTimeReliabilityScoring_Baseline",
        "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
        "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation",
        "RecursiveSummarizationConsolidation_NoConstructorGuard",
    ]
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        metrics = main_summary.get(cond, {}).get("poisoned", {})
        if not metrics:
            continue
        rows.append({
            "condition": COND_SHORT.get(cond, cond),
            "multimodal_attack_retrieval_rate": metrics.get("multimodal_attack_retrieval_rate"),
            "fact_overwrite_attack_retrieval_rate": metrics.get("fact_overwrite_attack_retrieval_rate"),
            "control_flow_attack_retrieval_rate": metrics.get("control_flow_attack_retrieval_rate"),
            "answer_relevant_attack_retrieval_rate": metrics.get("answer_relevant_attack_retrieval_rate"),
            "derived_memory_corruption_rate": metrics.get("derived_memory_corruption_rate"),
            "attack_write_admission_rate": metrics.get("attack_write_admission_rate"),
        })
    fieldnames = [
        "condition",
        "multimodal_attack_retrieval_rate",
        "fact_overwrite_attack_retrieval_rate",
        "control_flow_attack_retrieval_rate",
        "answer_relevant_attack_retrieval_rate",
        "derived_memory_corruption_rate",
        "attack_write_admission_rate",
    ]
    write_csv(out_dir / "attack_proxy_breakdown.csv", rows, fieldnames)


def build_systems_cost_table(results_dir: Path, out_dir: Path) -> None:
    browse_adv = get_summary(results_dir / PAPER_FILES["browse_adv"], "mm_browsecomp")
    conditions = [
        "MMA_RetrieveTimeReliabilityScoring_Baseline",
        "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
        "SAGEMemV2_BrowsingTrustPrior",
        "SAGEMemV2_ABR",
    ]
    rows: list[dict[str, Any]] = []
    for cond in conditions:
        m = browse_adv.get(cond, {}).get("poisoned", {})
        if not m:
            continue
        rows.append({
            "condition": COND_SHORT.get(cond, cond),
            "write_avg_ms": m.get("write_avg_ms"),
            "retrieve_avg_ms": m.get("retrieve_avg_ms"),
            "items_in_memory_avg": m.get("items_in_memory_avg"),
            "items_in_audit_avg": m.get("items_in_audit_avg"),
            "structured_claim_gate_fire_count_avg": m.get("structured_claim_gate_fire_count_avg"),
            "group_divergence_fire_count_avg": m.get("group_divergence_fire_count_avg"),
        })
    fieldnames = list(rows[0].keys()) if rows else ["condition"]
    write_csv(out_dir / "systems_cost_table.csv", rows, fieldnames)


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2
    frac = (value - src_min) / (src_max - src_min)
    return dst_min + frac * (dst_max - dst_min)


def write_svg_pareto(path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    ensure_dir(path.parent)
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [float(r[x_key]) for r in rows if r.get(x_key) is not None]
    ys = [float(r[y_key]) for r in rows if r.get(y_key) is not None]
    if not xs or not ys:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_min = min(0.0, x_min)
    y_min = min(0.0, y_min)
    x_max = max(1.0, x_max)
    y_max = max(1.0, y_max)
    palette = ["#0f766e", "#1d4ed8", "#dc2626", "#7c3aed", "#ea580c", "#111827", "#16a34a", "#c2410c"]

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#111827} .small{font-size:12px;fill:#374151} .title{font-size:22px;font-weight:700} .axis{stroke:#111827;stroke-width:2} .grid{stroke:#d1d5db;stroke-width:1} </style>",
        f"<text x='{width/2}' y='34' text-anchor='middle' class='title'>{title}</text>",
    ]
    for i in range(6):
        frac = i / 5
        x = margin_left + frac * plot_w
        y = margin_top + (1 - frac) * plot_h
        parts.append(f"<line x1='{x}' y1='{margin_top}' x2='{x}' y2='{margin_top+plot_h}' class='grid' />")
        parts.append(f"<line x1='{margin_left}' y1='{y}' x2='{margin_left+plot_w}' y2='{y}' class='grid' />")
        xv = x_min + frac * (x_max - x_min)
        yv = y_min + frac * (y_max - y_min)
        parts.append(f"<text x='{x}' y='{height-42}' text-anchor='middle' class='small'>{xv:.2f}</text>")
        parts.append(f"<text x='{margin_left-16}' y='{y+4}' text-anchor='end' class='small'>{yv:.2f}</text>")
    parts.append(f"<line x1='{margin_left}' y1='{margin_top+plot_h}' x2='{margin_left+plot_w}' y2='{margin_top+plot_h}' class='axis' />")
    parts.append(f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top+plot_h}' class='axis' />")
    parts.append(f"<text x='{margin_left + plot_w/2}' y='{height-10}' text-anchor='middle'>{x_key}</text>")
    parts.append(f"<text x='22' y='{margin_top + plot_h/2}' text-anchor='middle' transform='rotate(-90 22 {margin_top + plot_h/2})'>{y_key}</text>")
    for idx, row in enumerate(rows):
        x = scale(float(row[x_key]), x_min, x_max, margin_left, margin_left + plot_w)
        y = scale(float(row[y_key]), y_min, y_max, margin_top + plot_h, margin_top)
        color = palette[idx % len(palette)]
        label = str(row["condition"])
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='{color}' />")
        parts.append(f"<text x='{x+10:.1f}' y='{y-8:.1f}' class='small'>{label}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_pareto_artifacts(results_dir: Path, out_dir: Path) -> None:
    main_summary = get_summary(results_dir / PAPER_FILES["main"], "locomo")
    browse_summary = get_summary(results_dir / PAPER_FILES["browse_adv"], "mm_browsecomp")
    locomo_rows = select_rows(
        main_summary,
        "poisoned",
        [
            "ShortContext_NoLongTermMemory_Baseline",
            "MMA_RetrieveTimeReliabilityScoring_Baseline",
            "RecursiveSummarizationConsolidation_NoConstructorGuard",
            "ConstructorGuardedStateUpdateSandbox_NonProceduralConsolidation",
            "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
        ],
    )
    browse_rows = select_rows(
        browse_summary,
        "poisoned",
        [
            "MMA_RetrieveTimeReliabilityScoring_Baseline",
            "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect",
            "SAGEMemV2_BrowsingTrustPrior",
            "SAGEMemV2_ABR",
        ],
    )
    write_svg_pareto(out_dir / "pareto_locomo_bcu_vs_write_asr.svg", "LoCoMo Pareto: BCU vs Write ASR", locomo_rows, "WriteASR", "BCU")
    write_svg_pareto(out_dir / "pareto_browsing_bcu_vs_write_asr.svg", "MM-BrowseComp Pareto: BCU vs Write ASR", browse_rows, "WriteASR", "BCU")


def build_schema_gap_report(results_dir: Path, out_dir: Path) -> None:
    main_data = load_json(results_dir / PAPER_FILES["main"])
    sample_raw = next(iter(next(iter(main_data["benchmarks"]["locomo"]["raw"].values())).values()))[0]
    missing = []
    for field in ("seed", "attack_type", "attack_types", "attack_write_attempt_count_by_type", "benign_write_admitted_count"):
        if field not in sample_raw:
            missing.append(field)
    lines = [
        "# Schema Gap Report",
        "",
        "The current frozen canonical artifacts are sufficient for core summary tables, systems-cost tables, browsing mechanism comparisons, and Pareto figures.",
        "",
        "The following reviewer-facing analyses cannot be computed honestly from the current frozen raws because the fields are not present in saved rows:",
        "",
    ]
    for field in missing:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "Implications:",
            "- true per-attack breakdown requires saved attack labels per QA row or per case unit",
            "- true mean±std / CI over seeds requires saved seed identifiers per QA row or per case unit",
            "- benign write recall requires saved benign write admission counts",
            "",
            "Recommended next rerun policy:",
            "- do not rerun full suites immediately",
            "- first stabilize the richer result schema",
            "- then rerun only the smallest paper-critical subset needed for each missing table",
        ]
    )
    ensure_dir(out_dir)
    (out_dir / "schema_gap_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_submission_summary(results_dir: Path, out_dir: Path) -> None:
    main_summary = get_summary(results_dir / PAPER_FILES["main"], "locomo")
    vpi_summary = get_summary(results_dir / PAPER_FILES["vpi"], "locomo")
    mmrobust_summary = get_summary(results_dir / PAPER_FILES["mmrobust"], "locomo")
    browse_adv_summary = get_summary(results_dir / PAPER_FILES["browse_adv"], "mm_browsecomp")
    browse_clean_summary = get_summary(results_dir / PAPER_FILES["browse_clean"], "mm_browsecomp")

    def metric(summary: dict[str, Any], cond: str, split: str, key: str) -> Any:
        return summary.get(cond, {}).get(split, {}).get(key)

    h6 = "SAGEMemV2_ABR"
    h5 = "SAGEMemV2_BrowsingTrustPrior"
    sage = "SAGEMemV2_BayesianTrust_ConsistencyGraph_AnomalyDetect"
    mma = "MMA_RetrieveTimeReliabilityScoring_Baseline"

    lines = [
        "# Submission-Ready Analysis Summary",
        "",
        "## Strongest Supported Results",
        "",
        f"- LoCoMo main: `{COND_SHORT[sage]}` has `Write ASR={fmt_num(metric(main_summary, sage, 'poisoned', 'attack_write_admission_rate'))}`, `Retrieval={fmt_num(metric(main_summary, sage, 'poisoned', 'attack_retrieval_rate'))}`, `ASR={fmt_num(metric(main_summary, sage, 'poisoned', 'ASR'))}`.",
        f"- VPI-only: `{COND_SHORT[sage]}` has `Write ASR={fmt_num(metric(vpi_summary, sage, 'poisoned', 'attack_write_admission_rate'))}`, `ASR={fmt_num(metric(vpi_summary, sage, 'poisoned', 'ASR'))}`.",
        f"- Multimodal robustness: `{COND_SHORT[sage]}` has `BCU poison={fmt_num(metric(mmrobust_summary, sage, 'poisoned', 'BenignCompletionUnderAttack'))}`, `Write ASR={fmt_num(metric(mmrobust_summary, sage, 'poisoned', 'attack_write_admission_rate'))}`, `ASR={fmt_num(metric(mmrobust_summary, sage, 'poisoned', 'ASR'))}`.",
        f"- Browsing benchmark: `{COND_SHORT[h6]}` has `BCU clean={fmt_num(metric(browse_clean_summary, h6, 'clean', 'BenignCompletionUnderAttack'))}`, `BCU poison={fmt_num(metric(browse_adv_summary, h6, 'poisoned', 'BenignCompletionUnderAttack'))}`, `Write ASR={fmt_num(metric(browse_adv_summary, h6, 'poisoned', 'attack_write_admission_rate'))}`, `ASR={fmt_num(metric(browse_adv_summary, h6, 'poisoned', 'ASR'))}`.",
        "",
        "## Browsing Comparison",
        "",
        f"- `{COND_SHORT[mma]}` collapses under attack: `BCU poison={fmt_num(metric(browse_adv_summary, mma, 'poisoned', 'BenignCompletionUnderAttack'))}`, `Write ASR={fmt_num(metric(browse_adv_summary, mma, 'poisoned', 'attack_write_admission_rate'))}`, `ASR={fmt_num(metric(browse_adv_summary, mma, 'poisoned', 'ASR'))}`.",
        f"- `{COND_SHORT[h5]}` reduces admission but not retrieval contamination: `Write ASR={fmt_num(metric(browse_adv_summary, h5, 'poisoned', 'attack_write_admission_rate'))}`, `Retrieval={fmt_num(metric(browse_adv_summary, h5, 'poisoned', 'attack_retrieval_rate'))}`, `ASR={fmt_num(metric(browse_adv_summary, h5, 'poisoned', 'ASR'))}`.",
        f"- `{COND_SHORT[h6]}` is the strongest browsing defense: `Write ASR={fmt_num(metric(browse_adv_summary, h6, 'poisoned', 'attack_write_admission_rate'))}`, `Retrieval={fmt_num(metric(browse_adv_summary, h6, 'poisoned', 'attack_retrieval_rate'))}`, `ASR={fmt_num(metric(browse_adv_summary, h6, 'poisoned', 'ASR'))}`.",
        "",
        "## Important Boundaries",
        "",
        "- The current frozen canonical artifacts do not include per-row `seed` or `attack_type`, so true seed-CI and true per-attack tables still require a targeted rerun after the richer schema patch.",
        "- The semantic observation-group rerun is useful as a secondary browsing-mechanism ablation, but it does not replace the canonical grouped H5/H6 pair.",
    ]
    ensure_dir(out_dir)
    (out_dir / "submission_ready_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    build_main_tables(args.results_dir, args.out_dir)
    build_browsing_tables(args.results_dir, args.out_dir)
    build_attack_proxy_tables(args.results_dir, args.out_dir)
    build_systems_cost_table(args.results_dir, args.out_dir)
    build_pareto_artifacts(args.results_dir, args.out_dir)
    build_schema_gap_report(args.results_dir, args.out_dir)
    build_submission_summary(args.results_dir, args.out_dir)
    print(f"Wrote paper analysis artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
