"""
Prepare augmented MM-BrowseComp cases for WARP.

This tool merges the official MM-BrowseComp task rows with externally produced
browsing observation traces. It does not generate observations itself.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from mma_bench_suite import _maybe_decrypt_mmbrowsecomp_text


_NORM_RE = re.compile(r"[^a-z0-9]+")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object per line")
            yield row


def _norm(text: object) -> str:
    return _NORM_RE.sub(" ", str(text).lower()).strip()


def _decrypt_official_row(row: dict) -> Tuple[str, str, List[str]]:
    password = row.get("canary") if isinstance(row.get("canary"), str) else None
    question = _maybe_decrypt_mmbrowsecomp_text(row.get("question", ""), password)
    answer = _maybe_decrypt_mmbrowsecomp_text(row.get("answer", ""), password)
    checklist = row.get("checklist", [])
    if isinstance(checklist, list):
        checklist = [
            str(_maybe_decrypt_mmbrowsecomp_text(item, password)) for item in checklist
        ]
    else:
        checklist = []
    question = str(question).split("Question: ", 1)[-1].strip()
    return str(question), str(answer), checklist


def _validate_observations(case_id: str, observations: object) -> List[dict]:
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"{case_id}: observations must be a non-empty list")
    out = []
    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            raise ValueError(f"{case_id}: observation {idx} must be an object")
        text = str(obs.get("text", obs.get("content", ""))).strip()
        if not text:
            raise ValueError(f"{case_id}: observation {idx} has empty text")
        source_type = str(obs.get("source_type", "")).strip()
        if source_type not in {"ocr_text", "vision_caption", "tool_output_text", "user"}:
            raise ValueError(
                f"{case_id}: observation {idx} has unsupported source_type={source_type!r}"
            )
        out.append(
            {
                "text": text,
                "source_type": source_type,
                "channel_id": str(obs.get("channel_id", f"obs_{idx}")),
                "session_idx": int(obs.get("session_idx", 1)),
                "role": str(obs.get("role", "tool")),
            }
        )
    return out


def _find_leakage(observations: List[dict], answer: str, checklist: List[str]) -> List[str]:
    leakage = []
    answer_norm = _norm(answer)
    clue_norms = [x for x in [_norm(item) for item in checklist] if len(x) >= 8]
    for obs in observations:
        obs_norm = _norm(obs["text"])
        if answer_norm and len(answer_norm) >= 6 and answer_norm in obs_norm:
            leakage.append(f"answer leak in channel {obs['channel_id']}")
            continue
        for item in clue_norms:
            if item in obs_norm:
                leakage.append(f"checklist leak in channel {obs['channel_id']}")
                break
    return leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MM-BrowseComp tasks with observation traces")
    parser.add_argument("--official", required=True, help="Official MM-BrowseComp JSONL")
    parser.add_argument("--traces", required=True, help="Observation trace JSONL keyed by id")
    parser.add_argument("--out", required=True, help="Output augmented JSONL for WARP")
    parser.add_argument("--strict", action="store_true", help="Fail on detected answer/checklist leakage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_rows = {str(row["id"]): row for row in _iter_jsonl(Path(args.official))}
    trace_rows = {str(row["id"]): row for row in _iter_jsonl(Path(args.traces))}

    merged = []
    leakage_warnings = []
    missing = []

    for case_id, row in official_rows.items():
        trace = trace_rows.get(case_id)
        if trace is None:
            missing.append(case_id)
            continue
        observations = _validate_observations(case_id, trace.get("observations"))
        question, answer, checklist = _decrypt_official_row(row)
        leakage = _find_leakage(observations, answer, checklist)
        if leakage:
            leakage_warnings.append({"id": case_id, "warnings": leakage})
            if args.strict:
                raise SystemExit(
                    f"{case_id}: leakage detected: {', '.join(leakage)}"
                )
        merged_row = dict(row)
        merged_row["observations"] = observations
        merged_row["evaluation"] = [
            {
                "id": row.get("id"),
                "question": question,
                "answer": answer,
                "checklist": checklist,
                "evidence": [],
                "evidence_sessions": [1],
                "judgment_logic": f"Correct if answer matches: {answer}",
            }
        ]
        merged.append(merged_row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "official_rows": len(official_rows),
        "trace_rows": len(trace_rows),
        "merged_rows": len(merged),
        "missing_trace_rows": len(missing),
        "leakage_warning_rows": len(leakage_warnings),
    }
    print(json.dumps(summary, indent=2))
    if missing:
        print(f"Missing traces for first ids: {missing[:10]}")
    if leakage_warnings:
        print(f"Leakage warnings for first ids: {leakage_warnings[:5]}")


if __name__ == "__main__":
    main()
