"""
Browser-backed MM-BrowseComp trace builder.

This script uses Playwright to render official MM-BrowseComp source URLs, extract
visible text, save screenshots, and optionally OCR the rendered page. It also
downloads official benchmark image URLs and OCRs them locally with tesseract.

No paid model calls are used here.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

from mma_bench_suite import _maybe_decrypt_mmbrowsecomp_text

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


_WS_RE = re.compile(r"\s+")
_LOW_SIGNAL_PATTERNS = (
    "enable javascript",
    "unusual traffic",
    "our systems have detected",
    "captcha",
    "robot",
    "about this page",
)


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


def _ocr_file(path: Path, max_chars: int) -> Optional[str]:
    with tempfile.TemporaryDirectory(prefix="mmbc_ocr_") as tmpdir:
        txt_base = Path(tmpdir) / "ocr_out"
        proc = subprocess.run(
            ["tesseract", str(path), str(txt_base)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        txt_path = txt_base.with_suffix(".txt")
        if proc.returncode != 0 or not txt_path.exists():
            return None
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = _WS_RE.sub(" ", text).strip()
        if not text:
            return None
        return text[:max_chars]


def _download_file(url: str, out_path: Path, timeout_s: int) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            out_path.write_bytes(resp.read())
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def _looks_low_signal(text: str) -> bool:
    lowered = text.lower()
    return any(pat in lowered for pat in _LOW_SIGNAL_PATTERNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build browser-backed MM-BrowseComp traces")
    parser.add_argument("--official", required=True, help="Official MM-BrowseComp JSONL")
    parser.add_argument("--out", required=True, help="Output traces JSONL")
    parser.add_argument("--artifacts-dir", default="data/mmbc_browser_artifacts", help="Artifact directory")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional specific case ids")
    parser.add_argument("--page-timeout-ms", type=int, default=20000, help="Page timeout in ms")
    parser.add_argument("--max-page-chars", type=int, default=4000, help="Max visible page chars")
    parser.add_argument("--max-ocr-chars", type=int, default=1200, help="Max OCR chars")
    parser.add_argument("--include-page-ocr", action="store_true", help="OCR rendered page screenshots too")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    return parser.parse_args()


def main() -> None:
    if sync_playwright is None:
        raise SystemExit(
            "playwright is not installed in .venv. Install it there before using this script."
        )
    if shutil.which("tesseract") is None:
        raise SystemExit("tesseract is required")

    args = parse_args()
    case_ids = {str(x) for x in args.case_ids} if args.case_ids else None
    rows = list(_iter_jsonl(Path(args.official)))
    if case_ids is not None:
        rows = [row for row in rows if str(row.get("id")) in case_ids]
    if args.limit > 0:
        rows = rows[: args.limit]

    artifacts_root = Path(args.artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw, open(out_path, "w", encoding="utf-8") as f:
        browser = pw.chromium.launch(headless=bool(args.headless))
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.set_default_timeout(int(args.page_timeout_ms))

        built = 0
        for row in rows:
            case_id = str(row.get("id"))
            case_dir = artifacts_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            observations = []
            page_signal = {"useful_pages": 0, "low_signal_pages": 0, "image_ocr_hits": 0}

            for idx, url in enumerate(row.get("source", []) or []):
                if not isinstance(url, str) or not url.strip():
                    continue
                url = url.split("://", 1)[0] + "://" + url.split("://", 1)[1] if "://" in url else url
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)
                    title = page.title().strip()
                    body_text = page.locator("body").inner_text(timeout=5000)
                    body_text = _WS_RE.sub(" ", body_text).strip()
                except Exception:
                    continue
                screenshot_path = case_dir / f"source_{idx}.png"
                try:
                    page.screenshot(path=str(screenshot_path), full_page=True)
                except Exception:
                    pass
                text = body_text[: args.max_page_chars]
                if not text:
                    continue
                formatted = f"PAGE_TITLE: {title}\nPAGE_TEXT: {text}" if title else f"PAGE_TEXT: {text}"
                group_id = f"mmbrowse:{case_id}:source_{idx}"
                observations.append(
                    {
                        "text": formatted,
                        "source_type": "tool_output_text",
                        "channel_id": f"source_{idx}",
                        "observation_group": group_id,
                        "page_id": group_id,
                        "source_url": url.strip(),
                        "session_idx": 1,
                        "role": "tool",
                    }
                )
                if _looks_low_signal(text):
                    page_signal["low_signal_pages"] += 1
                else:
                    page_signal["useful_pages"] += 1
                if args.include_page_ocr and screenshot_path.exists():
                    ocr_text = _ocr_file(screenshot_path, args.max_ocr_chars)
                    if ocr_text:
                        observations.append(
                            {
                                "text": ocr_text,
                                "source_type": "ocr_text",
                                "channel_id": f"source_{idx}_page_ocr",
                                "observation_group": group_id,
                                "page_id": group_id,
                                "source_url": url.strip(),
                                "session_idx": 1,
                                "role": "vision",
                            }
                        )

            for idx, url in enumerate(row.get("images", []) or []):
                if not isinstance(url, str) or not url.strip():
                    continue
                img_path = case_dir / f"image_{idx}.bin"
                if not _download_file(url.strip(), img_path, timeout_s=max(10, args.page_timeout_ms // 1000)):
                    continue
                ocr_text = _ocr_file(img_path, args.max_ocr_chars)
                if ocr_text:
                    page_signal["image_ocr_hits"] += 1
                    group_id = f"mmbrowse:{case_id}:image_{idx}"
                    observations.append(
                        {
                            "text": ocr_text,
                            "source_type": "ocr_text",
                            "channel_id": f"image_{idx}_ocr",
                            "observation_group": group_id,
                            "page_id": group_id,
                            "source_url": url.strip(),
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
                    **page_signal,
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            built += 1

        browser.close()

    print(json.dumps({"input_rows": len(rows), "built_rows": built, "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
