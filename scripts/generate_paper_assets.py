#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "paper_submission_ready"
ASSETS_DIR = ROOT / "assets"
PLOTS_DIR = ASSETS_DIR / "plots"
MEDIA_DIR = ASSETS_DIR / "media"


PLOT_FILES = [
    "pareto_locomo_bcu_vs_write_asr.svg",
    "pareto_browsing_bcu_vs_write_asr.svg",
]


ARCH_DOT = r"""
digraph SAGEMemArchitecture {
  rankdir=LR;
  graph [pad="0.2", nodesep="0.5", ranksep="0.8", bgcolor="white"];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, penwidth=1.1, color="#5B6570"];

  subgraph cluster_sources {
    label="Observation Sources";
    color="#D8E2EA";
    style="rounded";
    user [label="User / Tool / Browser", fillcolor="#EEF4F8", color="#90A4B4"];
    vision [label="OCR / Vision Caption", fillcolor="#EEF4F8", color="#90A4B4"];
  }

  subgraph cluster_gate {
    label="Write-Time Governance";
    color="#CFE4D1";
    style="rounded";
    ingest [label="Channel + Source Provenance", fillcolor="#F1F8F2", color="#6FA079"];
    trust [label="Trust + Anomaly +\nConsistency Checks", fillcolor="#F1F8F2", color="#6FA079"];
    browse [label="BrowseGuard:\nStructured Claim Gate", fillcolor="#E8F6EA", color="#4F8B5B"];
    dep [label="Dependent-Evidence\nHandling", fillcolor="#E8F6EA", color="#4F8B5B"];
  }

  subgraph cluster_memory {
    label="Typed Memory";
    color="#F2D9C7";
    style="rounded";
    evidence [label="Evidence", fillcolor="#FFF5EE", color="#C28B62"];
    belief [label="Belief", fillcolor="#FFF5EE", color="#C28B62"];
    control [label="Control", fillcolor="#FFF5EE", color="#C28B62"];
    audit [label="Audit / Quarantine", fillcolor="#FFF5EE", color="#C28B62"];
  }

  subgraph cluster_downstream {
    label="Downstream Use";
    color="#D9D4EF";
    style="rounded";
    retrieve [label="Provenance-Aware Retrieval", fillcolor="#F5F2FD", color="#8B79B3"];
    qa [label="Planner / QA / Action", fillcolor="#F5F2FD", color="#8B79B3"];
  }

  user -> ingest [label="raw observations"];
  vision -> ingest [label="multimodal signals"];
  ingest -> trust [label="typed write request"];
  trust -> browse [label="browser claims"];
  trust -> dep [label="cross-modal support"];
  browse -> evidence [label="admit"];
  dep -> evidence [label="admit"];
  trust -> audit [label="quarantine / reject", style=dashed, color="#B96A5B"];
  evidence -> belief [label="sufficiency-gated promotion"];
  evidence -> control [label="typed control memory"];
  belief -> retrieve;
  control -> retrieve;
  evidence -> retrieve;
  audit -> retrieve [label="lineage only", style=dashed, color="#B96A5B"];
  retrieve -> qa [label="grounded context"];
}
"""


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def copy_plots() -> None:
    for name in PLOT_FILES:
        src = ANALYSIS_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing plot artifact: {src}")
        shutil.copy2(src, PLOTS_DIR / name)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf(objects: list[bytes]) -> bytes:
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


def _write_simple_scatter_pdf(path: Path, title: str, x_label: str, y_label: str, rows: list[dict[str, str]]) -> None:
    width, height = 620, 420
    left, bottom, plot_w, plot_h = 80, 70, 430, 270
    x_max = 1.0
    y_vals = [float(r["BCU"]) for r in rows]
    y_min = 0.0
    y_max = max(y_vals) if y_vals else 1.0
    y_max = max(0.2, y_max)
    colors = {
        "MMA": (0.80, 0.29, 0.29),
        "SAGE-Mem v2": (0.20, 0.47, 0.74),
        "SAGE-Mem": (0.20, 0.47, 0.74),
        "RSum": (0.50, 0.50, 0.50),
        "ShortContext": (0.60, 0.60, 0.60),
        "H1": (0.35, 0.62, 0.35),
        "H5": (0.93, 0.69, 0.13),
        "H6": (0.31, 0.39, 0.64),
    }

    def sx(x: float) -> float:
        return left + (x / x_max) * plot_w

    def sy(y: float) -> float:
        denom = (y_max - y_min) or 1.0
        return bottom + ((y - y_min) / denom) * plot_h

    content: list[str] = []
    content.append("1 w 0 0 0 RG")
    content.append(f"{left} {bottom} m {left+plot_w} {bottom} l S")
    content.append(f"{left} {bottom} m {left} {bottom+plot_h} l S")

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = sx(tick)
        content.append("0.85 0.85 0.85 RG 0.5 w")
        content.append(f"{x:.1f} {bottom} m {x:.1f} {bottom+plot_h} l S")
        content.append("0 0 0 RG 1 w")
        content.append(
            f"BT /F1 9 Tf {x-8:.1f} {bottom-18} Td ({_pdf_escape(f'{tick:.2f}')}) Tj ET"
        )

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        y = sy(yv)
        content.append("0.85 0.85 0.85 RG 0.5 w")
        content.append(f"{left} {y:.1f} m {left+plot_w} {y:.1f} l S")
        content.append("0 0 0 RG 1 w")
        content.append(
            f"BT /F1 9 Tf {left-38:.1f} {y-3:.1f} Td ({_pdf_escape(f'{yv:.2f}')}) Tj ET"
        )

    content.append(f"BT /F2 14 Tf 120 {height-28} Td ({_pdf_escape(title)}) Tj ET")
    content.append(f"BT /F1 11 Tf {left+140} 30 Td ({_pdf_escape(x_label)}) Tj ET")
    content.append(f"BT /F1 11 Tf 18 {bottom+120} Td ({_pdf_escape(y_label)}) Tj ET")

    for row in rows:
        name = row["condition"]
        x = float(row["WriteASR"])
        y = float(row["BCU"])
        r, g, b = colors.get(name, (0.25, 0.25, 0.25))
        px, py = sx(x), sy(y)
        size = 6
        content.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        content.append(f"{px-size/2:.1f} {py-size/2:.1f} {size} {size} re f")
        content.append("0 0 0 rg")
        content.append(f"BT /F1 9 Tf {px+6:.1f} {py+3:.1f} Td ({_pdf_escape(name)}) Tj ET")

    stream = "\n".join(content).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]
    path.write_bytes(_build_pdf(objects))


def generate_plot_pdfs() -> None:
    specs = [
        (
            ANALYSIS_DIR / "main_poison_table.csv",
            PLOTS_DIR / "pareto_locomo_bcu_vs_write_asr.pdf",
            "LoCoMo Pareto: BCU vs Write ASR",
        ),
        (
            ANALYSIS_DIR / "browsing_adversarial_table.csv",
            PLOTS_DIR / "pareto_browsing_bcu_vs_write_asr.pdf",
            "MM-BrowseComp Pareto: BCU vs Write ASR",
        ),
    ]
    for src, out, title in specs:
        with src.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        _write_simple_scatter_pdf(out, title, "Attack write admission rate", "BCU under attack", rows)


def generate_architecture() -> None:
    dot_path = MEDIA_DIR / "sagemem_architecture.dot"
    dot_path.write_text(ARCH_DOT.strip() + "\n", encoding="utf-8")
    for fmt in ("svg", "png", "pdf"):
        out = MEDIA_DIR / f"sagemem_architecture.{fmt}"
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(out)],
            check=True,
        )


def write_manifest() -> None:
    manifest = ASSETS_DIR / "README.md"
    manifest.write_text(
        "\n".join(
            [
                "# Paper Assets",
                "",
                "Generated by `scripts/generate_paper_assets.py`.",
                "",
                "## Plots",
                "",
                "- `assets/plots/pareto_locomo_bcu_vs_write_asr.svg`",
                "- `assets/plots/pareto_browsing_bcu_vs_write_asr.svg`",
                "",
                "## Media",
                "",
                "- `assets/media/sagemem_architecture.svg`",
                "- `assets/media/sagemem_architecture.png`",
                "- `assets/media/sagemem_architecture.pdf`",
                "",
                "These are the paper-ready figure asset paths to reference from LaTeX.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    copy_plots()
    generate_plot_pdfs()
    generate_architecture()
    write_manifest()
    print("Generated paper assets:")
    for path in sorted(ASSETS_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
