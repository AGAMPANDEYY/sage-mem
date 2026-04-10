#!/usr/bin/env python3
"""Audit MM-BrowseComp trace/case artifacts for benchmark-quality issues."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LOW_SIGNAL_PATTERNS = (
    "just a moment",
    "enable javascript and cookies",
    "please enable javascript",
    "unusual traffic from your computer",
    "verifying you are human",
    "verify you are human",
    "are you a robot",
    "captcha",
    "验证码",
    "access denied",
    "403 forbidden",
    "performance & security by cloudflare",
    "browser does not support",
    "your browser is not supported",
    "about this page",
    "about press copyright contact us creator",
)

NORM_RE = re.compile(r"[^a-z0-9]+")


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _derive_xor_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _maybe_decrypt(value: object, password: str | None) -> str:
    if not isinstance(value, str) or not password:
        return str(value or "")
    candidate = value.strip()
    if not candidate or not re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
        return str(value or "")
    try:
        encrypted = base64.b64decode(candidate)
        key = _derive_xor_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")
    except Exception:
        return str(value or "")
    return decrypted if decrypted.strip() else str(value or "")


def _norm(text: object) -> str:
    return NORM_RE.sub(" ", str(text).lower()).strip()


def _obs(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in (case.get("observations") or case.get("dialogue_history") or []) if isinstance(o, dict)]


def _text(obs: dict[str, Any]) -> str:
    return str(obs.get("text", obs.get("content", ""))).strip()


def _source(obs: dict[str, Any]) -> str:
    return str(obs.get("source_type", obs.get("source", ""))).strip()


def _looks_junk(text: str) -> str | None:
    low = text.lower()
    for pattern in LOW_SIGNAL_PATTERNS:
        if pattern in low:
            return pattern
    return None


def audit_cases(path: Path) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    rows = list(_iter_jsonl(path))
    obs_counts = Counter()
    effective_counts = Counter()
    source_counts = Counter()
    image_counts = Counter()
    junk_cases = []
    duplicate_cases = []
    leakage_cases = []
    one_text_no_image = []

    for row in rows:
        case_id = str(row.get("case_id", row.get("id", "?")))
        observations = _obs(row)
        images = row.get("images") or row.get("metadata", {}).get("images") or []
        evals = row.get("evaluation") or []
        answer = ""
        checklist: list[str] = []
        if evals and isinstance(evals, list):
            answer = str(evals[0].get("answer", ""))
            checklist = [str(x) for x in evals[0].get("checklist", [])]
        elif "answer" in row:
            answer = _maybe_decrypt(row.get("answer"), row.get("canary"))
            checklist = [
                _maybe_decrypt(x, row.get("canary"))
                for x in row.get("checklist", [])
                if isinstance(x, str)
            ]

        obs_counts[len(observations)] += 1
        image_counts[len(images)] += 1
        effective_counts[len(observations) + min(1, len(images))] += 1
        if len(observations) == 1 and not images:
            one_text_no_image.append(case_id)

        seen = set()
        for obs in observations:
            text = _text(obs)
            source_counts[_source(obs)] += 1
            junk = _looks_junk(text)
            if junk:
                junk_cases.append((case_id, junk))
            if text in seen:
                duplicate_cases.append(case_id)
            seen.add(text)

        haystack = _norm("\n".join(_text(o) for o in observations))
        ans_norm = _norm(answer)
        if len(ans_norm) >= 6 and ans_norm in haystack:
            leakage_cases.append((case_id, "answer"))
        for item in checklist:
            item_norm = _norm(item)
            if len(item_norm) >= 12 and item_norm in haystack:
                leakage_cases.append((case_id, "checklist"))
                break

    if one_text_no_image:
        problems.append(f"{len(one_text_no_image)} cases have one text observation and no image")
    if junk_cases:
        problems.append(f"{len(set(c for c, _ in junk_cases))} cases still contain low-signal/junk observations")
    if duplicate_cases:
        problems.append(f"{len(set(duplicate_cases))} cases still contain duplicate observation text")
    if leakage_cases:
        problems.append(f"{len(set(c for c, _ in leakage_cases))} cases have answer/checklist leakage")

    stats = {
        "rows": len(rows),
        "obs_count_distribution": dict(sorted(obs_counts.items())),
        "effective_obs_count_distribution": dict(sorted(effective_counts.items())),
        "image_count_distribution": dict(sorted(image_counts.items())),
        "source_type_distribution": dict(sorted(source_counts.items())),
        "junk_case_examples": junk_cases[:10],
        "duplicate_case_examples": duplicate_cases[:10],
        "leakage_case_examples": leakage_cases[:10],
    }
    return problems, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/mm_browsecomp_cases_filtered.jsonl"),
        help="Filtered MM-BrowseComp case JSONL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cases.exists():
        raise SystemExit(f"case file not found: {args.cases}")
    problems, stats = audit_cases(args.cases)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if problems:
        print("AUDIT FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
