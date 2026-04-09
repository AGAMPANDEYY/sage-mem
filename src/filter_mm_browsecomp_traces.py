"""
Filter raw MM-BrowseComp observation traces by quality.

Applies two levels of filtering:

  1. Per-observation: removes junk observations (Cloudflare barriers, CAPTCHA
     pages, 404/403 error pages, garbage OCR, YouTube JS-blocked stubs, and
     title-only tool stubs with < MIN_TOOL_CHARS meaningful characters).
     Duplicate observations (identical text within the same case) are also
     removed.

  2. Per-case: drops the entire case unless, after junk removal, at least
     MIN_GOOD_OBS non-junk observations survive and their combined text is at
     least MIN_TOTAL_CHARS characters.

  3. Leakage guard: if the trace JSONL carries a decrypted answer field (from
     prepare_mm_browsecomp_cases.py output), cases where the normalised answer
     string appears in any remaining observation are flagged and optionally
     dropped (--drop-leakage).

New junk categories (v2):
  - YouTube JS stubs: pages that returned only the "About Press Copyright..."
    footer because the fetcher could not execute JavaScript.  These look
    authoritative (tool_output_text) but contain zero retrievable facts.
  - Title-only stubs: tool_output_text observations with < MIN_TOOL_CHARS
    characters (page title + redirect / login page only).
  - Duplicate observations: identical text repeated within the same case.

This script does NOT filter based on which method wins or loses.
Quality gates are entirely content-based.

Usage (applied to raw trace file):
    python src/filter_mm_browsecomp_traces.py \\
        --traces data/mm_browsecomp_traces_ok.jsonl \\
        --out    data/mm_browsecomp_traces_filtered.jsonl \\
        --report

Usage (applied to merged case file, also strips junk obs in-place):
    python src/filter_mm_browsecomp_traces.py \\
        --cases  data/mm_browsecomp_cases_73.jsonl \\
        --out    data/mm_browsecomp_cases_filtered.jsonl \\
        --report --drop-leakage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Junk-detection constants
# ---------------------------------------------------------------------------

# Patterns that indicate the page returned a barrier / error instead of content.
# Matched case-insensitively against the full observation text.
LOW_SIGNAL_PATTERNS: Tuple[str, ...] = (
    # Cloudflare challenge page
    "just a moment",
    "enable javascript and cookies",
    "please enable javascript",
    # Generic bot/robot barriers
    "unusual traffic from your computer",
    "our systems have detected unusual",
    "verifying you are human",
    "verify you are human",
    "are you a robot",
    # CAPTCHA
    "captcha",
    "验证码",           # Mandarin "verification code / CAPTCHA"
    # Access / HTTP errors
    "access denied",
    "403 forbidden",
    # Cloudflare brand marker in error pages
    "performance & security by cloudflare",
    # Generic "browser doesn't support" placeholder
    "browser does not support",
    "your browser is not supported",
    # Google "about this page" interstitial
    "about this page",
)

# Separate patterns that indicate a 404 page ONLY when the observation is a
# tool_output_text (web page) — not applied to OCR results.
PAGE_NOT_FOUND_PATTERNS: Tuple[str, ...] = (
    "404 not found",
    "page not found",
    "this page doesn't exist",
    "this page could not be found",
    "error 404",
    "404 error",
)

# YouTube/JS-blocked stubs: the page returned a bare "About Press Copyright"
# footer because JavaScript was not executed. This signature is unique to
# YouTube (and similar JS-gated sites) and contains zero factual content.
# Matched against tool_output_text only, case-insensitively.
YOUTUBE_STUB_PATTERN: str = "about press copyright contact us creator"

# Minimum character count for a tool_output_text observation to be kept.
# Observations shorter than this are title-only stubs (login redirects, MSN,
# Instagram, etc.) that fetched a redirect or gated landing page.
MIN_TOOL_CHARS: int = 150

# Minimum meaningful character count for an OCR observation to be kept.
# Below this, the text is likely OCR garbage (misread image, blank image, etc.).
MIN_OCR_CHARS: int = 60

# Per-case gate: minimum number of surviving non-junk observations.
MIN_GOOD_OBS: int = 2

# Per-case gate: minimum combined character count of surviving observations.
# Raised from 600 to 1500 to ensure there is meaningful evidence before eval.
MIN_TOTAL_CHARS: int = 1500

# Answer leakage: minimum normalised answer length to trigger a check.
MIN_ANSWER_NORM_LEN: int = 6

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: object) -> str:
    return _NORM_RE.sub(" ", str(text).lower()).strip()


# ---------------------------------------------------------------------------
# Per-observation filter
# ---------------------------------------------------------------------------

def _is_junk_obs(obs: dict) -> Tuple[bool, str]:
    """
    Returns (is_junk, reason).

    Does NOT modify the observation.
    """
    text = str(obs.get("text", obs.get("content", ""))).strip()
    source_type = str(obs.get("source_type", "")).strip()
    text_lower = text.lower()

    # 1. Universal low-signal patterns
    for pat in LOW_SIGNAL_PATTERNS:
        if pat in text_lower:
            return True, f"low_signal:{pat!r}"

    # 2. 404 / page-not-found patterns on web pages only
    if source_type == "tool_output_text":
        for pat in PAGE_NOT_FOUND_PATTERNS:
            if pat in text_lower:
                return True, f"page_not_found:{pat!r}"

        # 3. YouTube/JS-blocked stubs: JS-gated page returned only footer boilerplate.
        #    Signature: "about press copyright contact us creator" in the text.
        #    These have a title but zero factual content.
        if YOUTUBE_STUB_PATTERN in text_lower:
            return True, "youtube_js_stub"

        # 4. Title-only stub: fetched a redirect / login-gated page.
        #    The text is just "PAGE_TITLE: X\nPAGE_TEXT: X" with nothing else.
        if len(text) < MIN_TOOL_CHARS:
            return True, f"title_only_stub:len={len(text)}"

    # 5. Garbage OCR (too short to be meaningful)
    if source_type == "ocr_text":
        # Strip whitespace and count real characters
        stripped = re.sub(r"\s+", "", text)
        if len(stripped) < MIN_OCR_CHARS:
            return True, f"garbage_ocr:len={len(stripped)}"

    return False, ""


def filter_observations(observations: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Split observations into (good, junk) lists, also removing exact duplicates.

    Returns (good_obs, junk_obs).
    Deduplication is within-case only: identical text appearing twice is kept
    only on its first occurrence.
    """
    good: List[dict] = []
    junk: List[dict] = []
    seen_texts: set = set()
    for obs in observations:
        text = str(obs.get("text", obs.get("content", ""))).strip()
        # Deduplicate
        if text in seen_texts:
            junk.append(obs)
            continue
        seen_texts.add(text)
        is_junk, _ = _is_junk_obs(obs)
        if is_junk:
            junk.append(obs)
        else:
            good.append(obs)
    return good, junk


# ---------------------------------------------------------------------------
# Per-case quality gate
# ---------------------------------------------------------------------------

def _check_leakage(good_obs: List[dict], answer: str) -> Optional[str]:
    """Returns the channel_id where leakage is found, or None."""
    ans_norm = _norm(answer)
    if len(ans_norm) < MIN_ANSWER_NORM_LEN:
        return None
    for obs in good_obs:
        obs_norm = _norm(str(obs.get("text", obs.get("content", ""))))
        if ans_norm in obs_norm:
            return str(obs.get("channel_id", "?"))
    return None


def case_passes_quality_gate(
    case_id: str,
    good_obs: List[dict],
    *,
    answer: str = "",
    drop_leakage: bool = False,
) -> Tuple[bool, str]:
    """
    Returns (passes, reason_if_not).

    good_obs: observations that already passed the per-observation filter.
    answer: decrypted gold answer (for leakage check), may be empty.
    drop_leakage: if True, cases with answer leakage fail the gate.
    """
    if len(good_obs) < MIN_GOOD_OBS:
        return False, f"too_few_good_obs:{len(good_obs)}<{MIN_GOOD_OBS}"

    total_chars = sum(len(str(obs.get("text", obs.get("content", "")))) for obs in good_obs)
    if total_chars < MIN_TOTAL_CHARS:
        return False, f"too_short:{total_chars}<{MIN_TOTAL_CHARS}"

    if drop_leakage and answer:
        leak_chan = _check_leakage(good_obs, answer)
        if leak_chan is not None:
            return False, f"answer_leakage:channel={leak_chan}"

    return True, ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_obs(row: dict) -> List[dict]:
    """Works for both raw trace rows (observations) and merged case rows (dialogue_history)."""
    obs = row.get("observations") or row.get("dialogue_history") or []
    return [o for o in obs if isinstance(o, dict)]


def _get_answer(row: dict) -> str:
    """Extract gold answer from merged case row."""
    evals = row.get("evaluation") or []
    if evals and isinstance(evals, list):
        return str(evals[0].get("answer", ""))
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MM-BrowseComp traces by quality")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path,
                        help="Raw trace JSONL (from build_mm_browsecomp_traces.py)")
    source.add_argument("--cases", type=Path,
                        help="Merged case JSONL (from prepare_mm_browsecomp_cases.py)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output filtered JSONL")
    p.add_argument("--min-good-obs", type=int, default=MIN_GOOD_OBS,
                   help=f"Minimum non-junk observations per case (default: {MIN_GOOD_OBS})")
    p.add_argument("--min-total-chars", type=int, default=MIN_TOTAL_CHARS,
                   help=f"Minimum combined chars after junk removal (default: {MIN_TOTAL_CHARS})")
    p.add_argument("--min-ocr-chars", type=int, default=MIN_OCR_CHARS,
                   help=f"Minimum non-whitespace chars for OCR obs (default: {MIN_OCR_CHARS})")
    p.add_argument("--min-tool-chars", type=int, default=MIN_TOOL_CHARS,
                   help=f"Minimum chars for tool_output_text obs (default: {MIN_TOOL_CHARS})")
    p.add_argument("--drop-leakage", action="store_true",
                   help="Drop cases where gold answer appears in any observation")
    p.add_argument("--report", action="store_true",
                   help="Print per-case quality report to stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Override module-level constants with CLI values
    global MIN_GOOD_OBS, MIN_TOTAL_CHARS, MIN_OCR_CHARS, MIN_TOOL_CHARS
    MIN_GOOD_OBS = args.min_good_obs
    MIN_TOTAL_CHARS = args.min_total_chars
    MIN_OCR_CHARS = args.min_ocr_chars
    MIN_TOOL_CHARS = args.min_tool_chars

    src_path = args.traces or args.cases
    is_cases_mode = args.cases is not None  # merged case file (has evaluation + dialogue_history)

    rows = list(_iter_jsonl(src_path))

    total = len(rows)
    kept = 0
    dropped = 0
    drop_reasons: Dict[str, int] = {}
    junk_obs_removed = 0
    report_lines = []

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out_fh:
        for row in rows:
            case_id = str(row.get("id", row.get("case_id", "?")))
            raw_obs = _get_obs(row)
            answer = _get_answer(row) if is_cases_mode else row.get("answer", "")

            # Per-observation filter
            good_obs, junk_obs = filter_observations(raw_obs)
            junk_obs_removed += len(junk_obs)
            junk_reasons = []
            for obs in junk_obs:
                _, reason = _is_junk_obs(obs)
                junk_reasons.append((obs.get("channel_id", "?"), reason))

            # Per-case quality gate
            passes, fail_reason = case_passes_quality_gate(
                case_id, good_obs,
                answer=answer,
                drop_leakage=args.drop_leakage,
            )

            if args.report:
                total_good_chars = sum(
                    len(str(o.get("text", o.get("content", "")))) for o in good_obs
                )
                status = "KEEP" if passes else f"DROP({fail_reason})"
                line = (
                    f"  {case_id:>8}  raw={len(raw_obs)} good={len(good_obs)}"
                    f"  chars={total_good_chars:>5}  {status}"
                )
                if junk_reasons:
                    line += f"  junk_removed={junk_reasons}"
                report_lines.append(line)

            if not passes:
                dropped += 1
                drop_reasons[fail_reason] = drop_reasons.get(fail_reason, 0) + 1
                continue

            # Write filtered row — replace obs/dialogue_history with cleaned version
            out_row = dict(row)
            if is_cases_mode:
                out_row["dialogue_history"] = good_obs
                # Update metadata if present
                if "metadata" in out_row and isinstance(out_row["metadata"], dict):
                    out_row["metadata"]["n_filtered_obs"] = len(good_obs)
                    out_row["metadata"]["n_junk_obs_removed"] = len(junk_obs)
            else:
                out_row["observations"] = good_obs
                # Update stats if present
                if "stats" in out_row and isinstance(out_row["stats"], dict):
                    out_row["stats"]["n_observations"] = len(good_obs)
                    out_row["stats"]["n_junk_removed"] = len(junk_obs)

            out_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input_rows": total,
        "kept_rows": kept,
        "dropped_rows": dropped,
        "junk_obs_removed_total": junk_obs_removed,
        "drop_reasons": drop_reasons,
        "filter_params": {
            "min_good_obs": MIN_GOOD_OBS,
            "min_total_chars": MIN_TOTAL_CHARS,
            "min_ocr_chars": MIN_OCR_CHARS,
            "min_tool_chars": MIN_TOOL_CHARS,
            "drop_leakage": args.drop_leakage,
            "youtube_stub_removed": True,
            "dedup_within_case": True,
        },
        "out": str(args.out),
    }

    if args.report:
        print(f"\n{'='*70}")
        print(f"MM-BrowseComp Quality Filter Report")
        print(f"Input:  {src_path}  ({total} rows)")
        print(f"Output: {args.out}")
        print(f"{'='*70}")
        print(f"\nPer-case results:")
        for line in report_lines:
            print(line)
        print(f"\nSummary:")
        print(f"  Kept:            {kept} / {total} ({kept/max(1,total)*100:.1f}%)")
        print(f"  Dropped:         {dropped} / {total} ({dropped/max(1,total)*100:.1f}%)")
        print(f"  Junk obs removed:{junk_obs_removed}")
        print(f"  Drop reasons:    {drop_reasons}")
        print()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
