"""
Prepare augmented MM-BrowseComp cases for mem-shield evaluation.

This tool merges the official MM-BrowseComp task rows with externally produced
browsing observation traces.  It does NOT generate observations itself.

Quality filtering (NEW):
  By default the script applies per-observation junk removal and case-level
  quality gates before writing output.  Pass --no-quality-filter to disable.
  Quality filter parameters can be tuned with:
    --min-good-obs      (default: 2)
    --min-total-chars   (default: 600)
    --min-ocr-chars     (default: 60)
  This filtering is content-based only — it is not method-specific.

Leakage guard:
  Cases where the normalised gold answer appears verbatim in a surviving
  observation are always warned.  With --drop-leakage they are excluded.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mma_bench_suite import _maybe_decrypt_mmbrowsecomp_text


_NORM_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Junk patterns  (same set as filter_mm_browsecomp_traces.py)
# ---------------------------------------------------------------------------

_LOW_SIGNAL_PATTERNS: Tuple[str, ...] = (
    "just a moment",
    "enable javascript and cookies",
    "please enable javascript",
    "unusual traffic from your computer",
    "our systems have detected unusual",
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
)

_PAGE_NOT_FOUND_PATTERNS: Tuple[str, ...] = (
    "404 not found",
    "page not found",
    "this page doesn't exist",
    "this page could not be found",
    "error 404",
    "404 error",
)


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


# ---------------------------------------------------------------------------
# Quality filter (per-observation)
# ---------------------------------------------------------------------------

_YOUTUBE_STUB_PATTERN: str = "about press copyright contact us creator"
_MIN_TOOL_CHARS: int = 150


def _is_junk_obs(obs: dict, *, min_ocr_chars: int, min_tool_chars: int = _MIN_TOOL_CHARS) -> Tuple[bool, str]:
    """Returns (is_junk, reason)."""
    text = str(obs.get("text", obs.get("content", ""))).strip()
    source_type = str(obs.get("source_type", "")).strip()
    text_lower = text.lower()

    for pat in _LOW_SIGNAL_PATTERNS:
        if pat in text_lower:
            return True, f"low_signal:{pat!r}"

    if source_type == "tool_output_text":
        for pat in _PAGE_NOT_FOUND_PATTERNS:
            if pat in text_lower:
                return True, f"page_not_found:{pat!r}"
        # YouTube/JS-blocked stub
        if _YOUTUBE_STUB_PATTERN in text_lower:
            return True, "youtube_js_stub"
        # Title-only stub
        if len(text) < min_tool_chars:
            return True, f"title_only_stub:len={len(text)}"

    if source_type == "ocr_text":
        stripped = re.sub(r"\s+", "", text)
        if len(stripped) < min_ocr_chars:
            return True, f"garbage_ocr:len={len(stripped)}"

    return False, ""


def _filter_observations_for_quality(
    observations: List[dict],
    *,
    min_ocr_chars: int,
    min_tool_chars: int = _MIN_TOOL_CHARS,
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """
    Returns (good_obs, junk_log).

    junk_log is a list of (channel_id, reason) for removed observations.
    Deduplicates within the case: identical text kept only on first occurrence.
    """
    good: List[dict] = []
    junk_log: List[Tuple[str, str]] = []
    seen_texts: set = set()
    for obs in observations:
        text = str(obs.get("text", obs.get("content", ""))).strip()
        if text in seen_texts:
            junk_log.append((str(obs.get("channel_id", "?")), "duplicate"))
            continue
        seen_texts.add(text)
        is_junk, reason = _is_junk_obs(obs, min_ocr_chars=min_ocr_chars, min_tool_chars=min_tool_chars)
        if is_junk:
            junk_log.append((str(obs.get("channel_id", "?")), reason))
        else:
            good.append(obs)
    return good, junk_log


# ---------------------------------------------------------------------------
# Legacy structural validator (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MM-BrowseComp tasks with observation traces")
    parser.add_argument("--official", required=True, help="Official MM-BrowseComp JSONL")
    parser.add_argument("--traces", required=True, help="Observation trace JSONL keyed by id")
    parser.add_argument("--out", required=True, help="Output augmented JSONL for mem-shield")
    # Quality filter options
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable per-observation junk removal and case quality gates")
    parser.add_argument("--min-good-obs", type=int, default=2,
                        help="Minimum non-junk observations after filtering (default: 2)")
    parser.add_argument("--min-total-chars", type=int, default=1500,
                        help="Minimum combined chars after junk removal (default: 1500)")
    parser.add_argument("--min-ocr-chars", type=int, default=60,
                        help="Minimum non-whitespace chars for OCR observation (default: 60)")
    parser.add_argument("--min-tool-chars", type=int, default=150,
                        help="Minimum chars for tool_output_text observation (default: 150)")
    # Leakage options
    parser.add_argument("--strict", action="store_true",
                        help="Fail on detected answer/checklist leakage (legacy flag)")
    parser.add_argument("--drop-leakage", action="store_true",
                        help="Silently drop (rather than fail) cases with answer leakage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_rows: Dict[str, dict] = {
        str(row["id"]): row for row in _iter_jsonl(Path(args.official))
    }
    trace_rows: Dict[str, dict] = {
        str(row["id"]): row for row in _iter_jsonl(Path(args.traces))
    }

    merged = []
    leakage_warnings = []
    missing = []
    quality_dropped = []
    total_junk_removed = 0

    for case_id, row in official_rows.items():
        trace = trace_rows.get(case_id)
        if trace is None:
            missing.append(case_id)
            continue

        raw_observations = trace.get("observations") or []

        # ── Quality filter (per-observation) ─────────────────────────────────
        if not args.no_quality_filter:
            good_obs_raw, junk_log = _filter_observations_for_quality(
                raw_observations,
                min_ocr_chars=args.min_ocr_chars,
                min_tool_chars=args.min_tool_chars,
            )
            total_junk_removed += len(junk_log)

            # Per-case gate. Official images become vision_caption observations
            # during evaluation, so they count toward the effective evidence
            # support for multimodal cases. This keeps image-grounded tasks
            # while still dropping one-text-observation cases with no image.
            n_good = len(good_obs_raw)
            official_images = [
                u for u in (row.get("images") or [])
                if isinstance(u, str) and u.strip()
            ]
            n_effective_obs = n_good + min(1, len(official_images))
            total_chars = sum(
                len(str(o.get("text", o.get("content", "")))) for o in good_obs_raw
            )
            if n_good < 1 or n_effective_obs < args.min_good_obs or total_chars < args.min_total_chars:
                quality_dropped.append({
                    "id": case_id,
                    "n_raw": len(raw_observations),
                    "n_good": n_good,
                    "n_effective_obs": n_effective_obs,
                    "n_images": len(official_images),
                    "total_chars": total_chars,
                    "junk_removed": junk_log,
                })
                continue

            raw_observations = good_obs_raw

        # ── Structural validation ─────────────────────────────────────────────
        try:
            observations = _validate_observations(case_id, raw_observations)
        except ValueError as exc:
            quality_dropped.append({"id": case_id, "validation_error": str(exc)})
            continue

        # ── Decrypt and check leakage ─────────────────────────────────────────
        question, answer, checklist = _decrypt_official_row(row)
        leakage = _find_leakage(observations, answer, checklist)
        if leakage:
            leakage_warnings.append({"id": case_id, "warnings": leakage})
            if args.strict:
                raise SystemExit(
                    f"{case_id}: leakage detected: {', '.join(leakage)}"
                )
            if args.drop_leakage:
                quality_dropped.append({"id": case_id, "leakage": leakage})
                continue

        # ── Build merged row ──────────────────────────────────────────────────
        merged_row = dict(row)
        merged_row["dialogue_history"] = observations
        # Preserve structured observations field for compatibility
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
        if "metadata" not in merged_row:
            merged_row["metadata"] = {}
        merged_row["metadata"]["benchmark"] = "mm_browsecomp"
        merged_row["metadata"]["n_observations"] = len(observations)
        # Mirror official image URLs into metadata.images so augment_mm_browsecomp_with_vision
        # can find them regardless of which lookup path it uses.
        official_images = [u for u in (row.get("images") or []) if isinstance(u, str) and u.strip()]
        if official_images:
            merged_row["metadata"]["images"] = official_images
        if not args.no_quality_filter:
            merged_row["metadata"]["quality_filtered"] = True
            merged_row["metadata"]["effective_observation_count"] = (
                len(observations) + min(1, len(official_images))
            )
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
        "quality_dropped_rows": len(quality_dropped),
        "total_junk_obs_removed": total_junk_removed,
        "leakage_warning_rows": len(leakage_warnings),
        "filter_params": {
            "quality_filter_enabled": not args.no_quality_filter,
            "min_good_obs": args.min_good_obs,
            "min_total_chars": args.min_total_chars,
            "min_ocr_chars": args.min_ocr_chars,
            "min_tool_chars": args.min_tool_chars,
            "drop_leakage": args.drop_leakage,
        },
    }
    print(json.dumps(summary, indent=2))

    if missing:
        print(f"Missing traces for first ids: {missing[:10]}")
    if quality_dropped:
        print(f"\nQuality-dropped cases ({len(quality_dropped)}):")
        for d in quality_dropped[:15]:
            print(f"  {d}")
    if leakage_warnings:
        print(f"\nLeakage warnings for first ids: {leakage_warnings[:5]}")


if __name__ == "__main__":
    main()
