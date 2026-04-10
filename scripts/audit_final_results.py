#!/usr/bin/env python3
"""Audit frozen paper result artifacts for provenance and metric consistency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = {
    "sagemem_main_llm.json": {
        "benchmark": "locomo",
        "llm_eval": True,
        "run_mm_browsecomp": False,
        "mm_only": False,
        "enable_locomo_multimodal": False,
        "attack_types": {
            "constructor_launder",
            "label_gaming",
            "ocr_injection",
            "vision_caption_injection",
            "visual_prompt_injection",
            "fact_overwrite_injection",
            "adaptive_nl_evasion",
            "buried_payload",
        },
        "splits": {"clean", "poisoned", "poisoned_cross_topic"},
        "n_cases": 10,
    },
    "sagemem_v2_ablations.json": {
        "benchmark": "locomo",
        "llm_eval": False,
        "run_mm_browsecomp": False,
        "mm_only": False,
        "enable_locomo_multimodal": False,
        "splits": {"clean", "poisoned", "poisoned_cross_topic"},
        "n_cases": 10,
    },
    "sagemem_vpi_llm.json": {
        "benchmark": "locomo",
        "llm_eval": True,
        "run_mm_browsecomp": False,
        "mm_only": False,
        "enable_locomo_multimodal": False,
        "attack_types": {"visual_prompt_injection"},
        "splits": {"clean", "poisoned"},
        "n_cases": 10,
    },
    "sagemem_multimodal_robustness_ablations.json": {
        "benchmark": "locomo",
        "llm_eval": False,
        "run_mm_browsecomp": False,
        "mm_only": False,
        "enable_locomo_multimodal": True,
        "splits": {"clean", "poisoned", "poisoned_cross_topic"},
        "n_cases": 10,
    },
    "sagemem_mm_browsecomp_clean.json": {
        "benchmark": "mm_browsecomp",
        "llm_eval": False,
        "run_mm_browsecomp": True,
        "mm_only": True,
        "vision_caption_mode": "openai",
        "mm_splits": ["clean"],
        "splits": {"clean"},
    },
    "sagemem_mm_browsecomp_adversarial.json": {
        "benchmark": "mm_browsecomp",
        "llm_eval": False,
        "run_mm_browsecomp": True,
        "mm_only": True,
        "vision_caption_mode": "openai",
        "mm_splits": ["poisoned"],
        "attack_types": {"fact_overwrite_injection"},
        "splits": {"poisoned"},
    },
}

REQUIRED_METRICS = {
    "BenignCompletionUnderAttack",
    "ASR",
    "attack_write_admission_rate",
    "attack_belief_formation_rate",
    "attack_retrieval_rate",
    "false_belief_rate",
    "n_cases",
    "n_qa_evals",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _value_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, set):
        return set(actual or []) == expected
    return actual == expected


def _benchmark_payload(data: dict[str, Any], name: str) -> dict[str, Any]:
    benches = data.get("benchmarks") or {}
    if name not in benches:
        raise AssertionError(f"missing benchmark payload {name!r}")
    return benches[name]


def _check_metric_range(path: Path, condition: str, split: str, metric: str, value: Any) -> list[str]:
    problems: list[str] = []
    if value is None:
        return problems
    if isinstance(value, float) and math.isnan(value):
        # NaN is allowed only for row-level belief_traceability_score, not summary metrics.
        problems.append(f"{path.name}: {condition}/{split}/{metric} is NaN")
        return problems
    if metric in {
        "BenignCompletionUnderAttack",
        "ASR",
        "ASR_behavioral",
        "answer_consistent_rate",
        "answer_consistent_rate_llm",
        "answered_rate",
        "attack_write_admission_rate",
        "attack_belief_formation_rate",
        "attack_retrieval_rate",
        "false_belief_rate",
        "belief_traceability_score",
    } and isinstance(value, (int, float)):
        if not (0.0 <= float(value) <= 1.0):
            problems.append(f"{path.name}: {condition}/{split}/{metric} out of [0,1]: {value}")
    return problems


def audit_file(path: Path, spec: dict[str, Any], *, mm_case_count: int | None) -> list[str]:
    problems: list[str] = []
    data = _load(path)
    config = data.get("config") or {}
    benchmark_name = spec["benchmark"]
    try:
        payload = _benchmark_payload(data, benchmark_name)
    except AssertionError as exc:
        return [f"{path.name}: {exc}"]

    for key in ("llm_eval", "run_mm_browsecomp", "mm_only", "enable_locomo_multimodal", "vision_caption_mode", "mm_splits"):
        if key in spec and not _value_equal(config.get(key), spec[key]):
            problems.append(f"{path.name}: config {key}={config.get(key)!r}, expected {spec[key]!r}")
    if "attack_types" in spec and not _value_equal(config.get("attack_types"), spec["attack_types"]):
        problems.append(f"{path.name}: attack_types={config.get('attack_types')!r}, expected {sorted(spec['attack_types'])!r}")

    summary = payload.get("summary") or {}
    raw = payload.get("raw") or {}
    if not summary:
        problems.append(f"{path.name}: empty summary for {benchmark_name}")
    if not raw:
        problems.append(f"{path.name}: empty raw rows for {benchmark_name}")

    for condition, splits in summary.items():
        got_splits = set(splits)
        if got_splits != spec["splits"]:
            problems.append(f"{path.name}: {condition} splits={sorted(got_splits)}, expected={sorted(spec['splits'])}")
        for split, metrics in splits.items():
            missing = REQUIRED_METRICS - set(metrics)
            if missing:
                problems.append(f"{path.name}: {condition}/{split} missing metrics {sorted(missing)}")
            expected_n = spec.get("n_cases")
            if expected_n is None and benchmark_name == "mm_browsecomp":
                expected_n = mm_case_count
            if expected_n is not None and metrics.get("n_cases") != expected_n:
                problems.append(
                    f"{path.name}: {condition}/{split} n_cases={metrics.get('n_cases')}, expected {expected_n}"
                )
            for metric, value in metrics.items():
                problems.extend(_check_metric_range(path, condition, split, metric, value))

    for condition, splits in raw.items():
        for split, rows in splits.items():
            if not rows:
                problems.append(f"{path.name}: {condition}/{split} has no raw rows")
            for idx, row in enumerate(rows[:5]):
                if row.get("split") != split:
                    problems.append(f"{path.name}: {condition}/{split} row {idx} split={row.get('split')!r}")
                if row.get("condition") != condition:
                    problems.append(f"{path.name}: {condition}/{split} row {idx} condition={row.get('condition')!r}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("final_paper_results_20260410"),
        help="Frozen result artifact directory.",
    )
    parser.add_argument(
        "--mm-cases",
        type=Path,
        default=Path("data/mm_browsecomp_cases_filtered.jsonl"),
        help="Current filtered MM-BrowseComp case file used to check n_cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problems: list[str] = []
    if not args.results_dir.is_dir():
        raise SystemExit(f"results directory not found: {args.results_dir}")

    mm_case_count = None
    if args.mm_cases.exists():
        mm_case_count = sum(1 for line in args.mm_cases.read_text().splitlines() if line.strip())

    for fname, spec in EXPECTED.items():
        path = args.results_dir / fname
        if not path.exists():
            problems.append(f"missing expected result file: {path}")
            continue
        problems.extend(audit_file(path, spec, mm_case_count=mm_case_count))

    if problems:
        print("AUDIT FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("AUDIT PASSED")
    print(f"checked_results_dir={args.results_dir}")
    print(f"mm_case_count={mm_case_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
