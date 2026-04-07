"""
Build MM-BrowseComp observation traces with local tools only.

This script fetches the official source URLs and image URLs, extracts:
  - tool_output_text observations from HTML pages
  - ocr_text observations from benchmark images using local tesseract

It does not use paid model calls and does not synthesize answers.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional

from mma_bench_suite import _maybe_decrypt_mmbrowsecomp_text


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
        timeout=timeout_s,
    )


def _fetch_url_text(url: str, timeout_s: int, max_chars: int) -> Optional[str]:
    proc = _run(
        [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--compressed",
            "--max-time",
            str(timeout_s),
            url,
        ],
        timeout_s=timeout_s + 2,
    )
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
        )
        txt_path = txt_base.with_suffix(".txt")
        if ocr.returncode != 0 or not txt_path.exists():
            return None
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = _WS_RE.sub(" ", text).strip()
        if not text:
            return None
        return text[:max_chars]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MM-BrowseComp observation traces")
    parser.add_argument("--official", required=True, help="Official MM-BrowseComp JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL with observations only")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional specific case ids")
    parser.add_argument("--timeout-s", type=int, default=20, help="Per-request timeout")
    parser.add_argument("--max-page-chars", type=int, default=2000, help="Max chars stored per source page")
    parser.add_argument("--max-ocr-chars", type=int, default=1000, help="Max chars stored per OCR observation")
    return parser.parse_args()


def main() -> None:
    if shutil.which("curl") is None:
        raise SystemExit("curl is required")
    if shutil.which("tesseract") is None:
        raise SystemExit("tesseract is required")

    args = parse_args()
    case_ids = {str(x) for x in args.case_ids} if args.case_ids else None
    rows = list(_iter_jsonl(Path(args.official)))
    if case_ids is not None:
        rows = [row for row in rows if str(row.get("id")) in case_ids]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    built = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            case_id = str(row.get("id"))
            observations = []

            for idx, url in enumerate(row.get("source", []) or []):
                if not isinstance(url, str) or not url.strip():
                    continue
                text = _fetch_url_text(url.strip(), args.timeout_s, args.max_page_chars)
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
                ocr_text = _fetch_image_ocr(url.strip(), args.timeout_s, args.max_ocr_chars)
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

            payload = {
                "id": case_id,
                "question": _decrypt_question(row),
                "observations": observations,
                "stats": {
                    "n_source_urls": len(row.get("source", []) or []),
                    "n_image_urls": len(row.get("images", []) or []),
                    "n_observations": len(observations),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            built += 1

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "built_rows": built,
                "out": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
