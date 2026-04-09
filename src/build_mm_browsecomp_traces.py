"""
Build MM-BrowseComp observation traces with local tools only.

This script fetches the official source URLs and image URLs, extracts:
  - tool_output_text observations from HTML pages (via curl)
  - ocr_text observations from benchmark images (via tesseract)

It does not use paid model calls and does not synthesize answers.

Key features:
  --max-workers   Parallel workers (default: 8). 400 cases × 20s ÷ 8 ≈ 17 min.
  --resume        Skip case IDs already present in --out (safe to Ctrl-C and restart).
  --max-page-chars  Per-source-URL text budget (default: 4000). Raised from 2000
                  so more cases survive the 1500-char quality gate downstream.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mma_bench_suite import _maybe_decrypt_mmbrowsecomp_text

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:
    Console = None  # type: ignore
    Progress = None  # type: ignore


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _decrypt_question(row: dict) -> str:
    password = row.get("canary") if isinstance(row.get("canary"), str) else None
    question = _maybe_decrypt_mmbrowsecomp_text(row.get("question", ""), password)
    return str(question).split("Question: ", 1)[-1].strip()


def _run(cmd: List[str], timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # binary — callers decode with errors="replace"
        timeout=timeout_s + 2,
    )


def _fetch_url_text(url: str, timeout_s: int, max_chars: int) -> Optional[str]:
    try:
        proc = _run(
            [
                "curl",
                "-L",
                "--silent",
                "--show-error",
                "--compressed",
                "--max-time",
                str(timeout_s),
                "-A",
                "Mozilla/5.0 (compatible; research-bot/1.0)",
                url,
            ],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout or not proc.stdout.strip():
        return None
    html = proc.stdout.decode("utf-8", errors="replace")
    title_match = _TITLE_RE.search(html)
    title = unescape(title_match.group(1)).strip() if title_match else ""
    body = _SCRIPT_RE.sub(" ", html)
    body = _STYLE_RE.sub(" ", body)
    body = _TAG_RE.sub(" ", body)
    body = unescape(body)
    body = _WS_RE.sub(" ", body).strip()
    if not body:
        return None
    text = body[:max_chars]
    if title:
        return f"PAGE_TITLE: {title}\nPAGE_TEXT: {text}"
    return f"PAGE_TEXT: {text}"


def _fetch_image_ocr(url: str, timeout_s: int, max_chars: int) -> Optional[str]:
    try:
        with tempfile.TemporaryDirectory(prefix="mmbc_img_") as tmpdir:
            img_path = Path(tmpdir) / "image.bin"
            proc = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    str(timeout_s),
                    "-o",
                    str(img_path),
                    url,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s + 2,
            )
            if proc.returncode != 0 or not img_path.exists() or img_path.stat().st_size == 0:
                return None
            txt_base = Path(tmpdir) / "ocr_out"
            ocr = subprocess.run(
                ["tesseract", str(img_path), str(txt_base)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            txt_path = txt_base.with_suffix(".txt")
            if ocr.returncode != 0 or not txt_path.exists():
                return None
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            text = _WS_RE.sub(" ", text).strip()
            if not text:
                return None
            return text[:max_chars]
    except (subprocess.TimeoutExpired, OSError):
        return None


def _process_row(row: dict, timeout_s: int, max_page_chars: int, max_ocr_chars: int) -> dict:
    case_id = str(row.get("id"))
    observations = []

    for idx, url in enumerate(row.get("source", []) or []):
        if not isinstance(url, str) or not url.strip():
            continue
        text = _fetch_url_text(url.strip(), timeout_s, max_page_chars)
        if text:
            observations.append(
                {
                    "text": text,
                    "source_type": "tool_output_text",
                    "channel_id": f"source_{idx}",
                    "session_idx": 1,
                    "role": "tool",
                }
            )

    for idx, url in enumerate(row.get("images", []) or []):
        if not isinstance(url, str) or not url.strip():
            continue
        ocr_text = _fetch_image_ocr(url.strip(), timeout_s, max_ocr_chars)
        if ocr_text:
            observations.append(
                {
                    "text": ocr_text,
                    "source_type": "ocr_text",
                    "channel_id": f"image_{idx}_ocr",
                    "session_idx": 1,
                    "role": "vision",
                }
            )

    return {
        "id": case_id,
        "question": _decrypt_question(row),
        "observations": observations,
        "stats": {
            "n_source_urls": len(row.get("source", []) or []),
            "n_image_urls": len(row.get("images", []) or []),
            "n_observations": len(observations),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MM-BrowseComp observation traces")
    parser.add_argument("--official", required=True, help="Official MM-BrowseComp JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL with observations only")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional specific case ids")
    parser.add_argument("--timeout-s", type=int, default=20, help="Per-request timeout (seconds)")
    parser.add_argument("--max-page-chars", type=int, default=4000,
                        help="Max chars stored per source page (default: 4000)")
    parser.add_argument("--max-ocr-chars", type=int, default=1200,
                        help="Max chars stored per OCR observation (default: 1200)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel fetch workers (default: 8)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip case IDs already present in --out (safe restart after interruption)")
    return parser.parse_args()


def main() -> None:
    if shutil.which("curl") is None:
        raise SystemExit("curl is required")
    if shutil.which("tesseract") is None:
        raise SystemExit("tesseract is required")

    args = parse_args()
    case_ids_filter = {str(x) for x in args.case_ids} if args.case_ids else None
    rows = list(_iter_jsonl(Path(args.official)))

    if case_ids_filter is not None:
        rows = [row for row in rows if str(row.get("id")) in case_ids_filter]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-done case IDs from existing output
    already_done: set = set()
    if args.resume and out_path.exists():
        for existing_row in _iter_jsonl(out_path):
            already_done.add(str(existing_row.get("id")))
        print(f"Resume: skipping {len(already_done)} already-built cases")

    pending = [row for row in rows if str(row.get("id")) not in already_done]
    use_rich = Console is not None and Progress is not None
    console = Console() if use_rich else None
    if console is not None:
        console.print(
            f"[bold cyan]MM-BrowseComp trace build[/bold cyan]  "
            f"pending={len(pending)} / total={len(rows)}  workers={args.max_workers}"
        )
    else:
        print(f"Processing {len(pending)} / {len(rows)} cases with {args.max_workers} workers ...")

    built = 0
    errors = 0
    nonempty = 0

    # Append to existing file when resuming, otherwise create fresh
    file_mode = "a" if args.resume and out_path.exists() else "w"

    with open(out_path, file_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures: Dict = {
                pool.submit(
                    _process_row, row, args.timeout_s, args.max_page_chars, args.max_ocr_chars
                ): row
                for row in pending
            }
            if console is not None:
                progress = Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=None),
                    TaskProgressColumn(),
                    TextColumn("done={task.completed}/{task.total}"),
                    TextColumn("nonempty={task.fields[nonempty]}"),
                    TextColumn("errors={task.fields[errors]}"),
                    TextColumn("last={task.fields[last_case]}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                )
                with progress:
                    task_id = progress.add_task(
                        "build_mm_traces",
                        total=len(pending),
                        nonempty=0,
                        errors=0,
                        last_case="-",
                    )
                    for future in as_completed(futures):
                        row = futures[future]
                        case_id = str(row.get("id"))
                        try:
                            payload = future.result()
                            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            f.flush()
                            built += 1
                            n_obs = payload["stats"]["n_observations"]
                            if n_obs > 0:
                                nonempty += 1
                            progress.advance(
                                task_id,
                                1,
                                nonempty=nonempty,
                                errors=errors,
                                last_case=f"{case_id}:{n_obs}",
                            )
                        except Exception as exc:
                            errors += 1
                            progress.advance(
                                task_id,
                                1,
                                nonempty=nonempty,
                                errors=errors,
                                last_case=f"{case_id}:ERR",
                            )
                            console.print(f"[red]ERROR[/red] id={case_id}: {exc}")
            else:
                for future in as_completed(futures):
                    row = futures[future]
                    case_id = str(row.get("id"))
                    try:
                        payload = future.result()
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        f.flush()
                        built += 1
                        n_obs = payload["stats"]["n_observations"]
                        if n_obs > 0:
                            nonempty += 1
                        if (built + errors) % 20 == 0 or n_obs > 0:
                            print(
                                f"  [{built + errors + len(already_done)}/{len(rows)}] "
                                f"id={case_id} obs={n_obs} nonempty={nonempty} errors={errors}"
                            )
                    except Exception as exc:
                        errors += 1
                        print(f"  ERROR id={case_id}: {exc}")

    total_in_file = len(already_done) + built
    summary = {
        "input_rows": len(rows),
        "already_done": len(already_done),
        "newly_built": built,
        "nonempty_traces": nonempty,
        "errors": errors,
        "total_in_output": total_in_file,
        "out": str(out_path),
    }
    if console is not None:
        console.print("[bold green]Trace build complete[/bold green]")
        console.print_json(data=summary)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
