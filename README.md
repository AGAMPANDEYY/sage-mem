<a href="https://www.iitr.ac.in/"><img src="assets/logo/iitr_logo.png" alt="IIT Roorkee" height="56" align="left"/></a>
<a href="https://dsgiitr.com/"><img src="assets/logo/dsg.png" alt="Data Science Group, IIT Roorkee" height="56" align="left"/></a>
<a href="https://icml.cc/"><img src="assets/logo/icml.png" alt="ICML 2026" height="56" align="right"/></a>

<h1 align="center">SAGE-Mem</h1>

<h3 align="center">Before It Persists: Write-Time Defense for Multimodal Agent Memory</h3>

<br clear="all"/>

<p align="center">
  <a href="https://icml.cc/"><img src="https://img.shields.io/badge/ICML_2026-SCALE_Workshop-blue" alt="Workshop"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
</p>

<p align="center">
  Official implementation of the ICML 2026 SCALE Workshop paper<br/>
  <em><strong>"Before It Persists: Write-Time Defense for Multimodal Agent Memory"</strong></em>
</p>

---

## TL;DR

Persistent memory makes multimodal agents more capable, but it also creates a new attack
surface: once unsupported content is written into memory, later retrieval and consolidation
can reuse it as if it were reliable state. **SAGE-Mem** is a write-time memory layer that
separates transient *evidence* from durable *belief*: observations may enter evidence,
but they are promoted to belief only when sufficiently supported, independent, and
non-conflicting. On the LoCoMo-Adv benchmark, SAGE-Mem reduces write admission from
1.000 (retrieval-time baseline) to 0.004 and retrieval contamination from 0.158 to a
95% rule-of-three upper bound of ≤ 0.002. On the broader five-attack MM-BrowseComp-Adv
suite, BrowseGuard-Extended reduces Write ASR from 0.255 to 0.037 and Retrieval ASR
from 0.564 to 0.369.

> *For persistent-memory agents, robustness should be evaluated not only at retrieval,
> but also at the point where observations become persistent state.*

---

## Links

- **Paper**: published at the ICML 2026 SCALE Workshop (OpenReview link forthcoming)
- **Poster**: see [`assets/poster/`](assets/poster/) (added after the workshop)
- **Code**: this repository
- **Author**: Agam Pandey — Indian Institute of Technology Roorkee, Mem0

---

## What's in SAGE-Mem

SAGE-Mem is a **governed memory layer**, not a complete autonomous agent. It sits
between heterogeneous observation sources and a downstream planner, and controls:

- what gets written into memory (*write-time admission gate*),
- what gets promoted from `evidence` into durable `belief` (*belief-promotion gate*),
- what retrieved memory is allowed to support downstream reasoning (*provenance-aware retrieval*).

Components in this implementation:

- typed memory (`evidence`, `belief`, `control`)
- write-time admission control with semantic-guard classification (DATA / DIRECTIVE / METADATA)
- sufficiency-gated evidence → belief promotion (independent support, non-conflict)
- Bayesian channel-trust calibration
- session-relative anomaly detection
- consistency-graph checks
- dependent-evidence handling for multimodal inputs (OCR + caption from the same image
  are *not* independent corroboration)
- provenance-aware retrieval (channel-trust floor, partition multiplier, conflict gate)
- **BrowseGuard**: browsing-specific write policy with S1–S4 semantic composite scoring
  across browser, OCR, and vision-caption channels

---

## Reproducing the Paper

All numeric tables in the paper are backed by frozen CSV artifacts checked into
[`analysis/`](analysis/). Every cell
referenced in the paper traces back to one of these files.

### 1. Install

```bash
git clone https://github.com/AGAMPANDEYY/mem-shield.git
cd mem-shield
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The default configuration uses a deterministic 256-dimensional hashed-text embedder
and does not require external API keys. The optional LLM guard and LLM-as-judge
evaluation paths require either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

### 2. Verify the install

```bash
make test
```

### 3. LoCoMo-Adv runs

```bash
make full-main-llm        RUN_ID=paper_main_full_v1
make full-v2-ablations    RUN_ID=paper_ablations_full_v1
make full-vpi-llm         RUN_ID=paper_vpi_full_v1
make full-mm-robust-ablations RUN_ID=paper_mmrobust_full_v1
```

### 4. MM-BrowseComp-Adv pipeline

```bash
# Build and filter traces
make build-mm-traces
make filter-mm-cases

# Clean and adversarial tracks
make full-mm-abr-clean       RUN_ID=paper_mmclean_abr_group_v1
make full-mm-abr-adversarial RUN_ID=paper_mmadv_abr_group_v1
```

### 5. Inspect a run

```bash
make analyze-run RUN_ID=paper_main_full_v1
```

The analyzer reads only `results/<RUN_ID>/` and renders a summary table for every JSON
in that folder.

### 6. Regenerate the paper-facing tables

The frozen CSVs underlying the paper tables live in
[`analysis/`](analysis/). To regenerate
them from raw result JSONs:

```bash
python scripts/paper_analysis.py
```

---

## Repository Structure

```
mem-shield/
├── src/                          # SAGE-Mem implementation
│   ├── memory.py                 # typed memory + write/promote/retrieve
│   ├── guard_llm.py              # single-agent LLM write guard
│   ├── guard_ensemble.py         # skeptic-advocate ensemble (paired classifiers)
│   ├── trust_calibration.py      # Bayesian channel trust updates
│   ├── anomaly_detector.py       # session-relative anomaly scoring
│   ├── consistency_graph.py      # cross-write consistency checks
│   ├── mma_bench_suite.py        # benchmark harness
│   ├── eval_judge.py             # LLM-as-judge evaluation
│   ├── build_mm_browsecomp_traces.py
│   ├── prepare_mm_browsecomp_cases.py
│   └── ...
├── scripts/                      # reproducibility + analysis scripts
│   ├── paper_analysis.py         # regenerate paper-facing CSVs
│   ├── analyze_run.py            # inspect a single run
│   ├── audit_final_results.py    # sanity-check final paper artifacts
│   ├── generate_paper_assets.py  # plots / figures
│   ├── sweep_pareto.py           # operating-point sweeps
│   ├── eval_buried_payload.py    # buried-payload attack evaluation
│   └── eval_guard_llm.py         # LLM guard evaluation
├── configs/
│   └── default_trust_config.json # source priors and threshold defaults
├── analysis/                     # frozen CSVs backing every paper table
│   ├── locomo_adv/               # LoCoMo-Adv tables
│   ├── mm_browsecomp_adv/        # MM-BrowseComp-Adv tables
│   └── plots/                    # Pareto SVGs
├── datasets/                     # publication-facing dataset bundle (gitignored bulk data)
│   ├── locomo/                   # LoCoMo-Adv inputs
│   └── mm_browsecomp/            # MM-BrowseComp-Adv inputs
├── data/                         # working data directory referenced by scripts
├── assets/
│   ├── media/                    # paper figures (architecture, dataset creation)
│   ├── plots/                    # Pareto plots
│   └── poster/                   # workshop poster (after the workshop)
├── tests/                        # unit tests
├── Makefile                      # reproducibility entrypoint
├── run_eval.py                   # main evaluation script
├── requirements.txt
├── COMMANDS.md                   # detailed command reference
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Lifecycle Metrics

SAGE-Mem is a *write-time* defense, so metrics are organized by memory lifecycle stage.

**Write-time metrics**
- `attack_write_admission_rate` (Write ASR) — fraction of injected attack observations
  admitted past the write boundary
- `attack_belief_formation_rate` — fraction of evaluations where attack-derived content
  is promoted into belief memory
- `write_quarantine_per_case`

**Retrieval-time metrics**
- `attack_retrieval_rate` (Retrieval ASR) — fraction of QA evaluations in which
  attack-derived content appears in the retrieved set
- `false_belief_rate` — fraction of retrieved belief items descending from poisoned lineage
- `belief_traceability_score`

**Downstream utility**
- `BCU` (Benign Completion Under Attack) — joint per-QA indicator
  `1[answer consistent ∧ ¬attack survived]`, averaged across evaluations

These four families together separate *admission*, *contamination*, *belief poisoning*,
and *task utility* rather than collapsing them into a single ASR.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{pandey2026sagemem,
  title     = {Before It Persists: Write-Time Defense for Multimodal Agent Memory},
  author    = {Pandey, Agam},
  booktitle = {ICML 2026 Workshop on SCALE},
  year      = {2026},
  url       = {https://github.com/AGAMPANDEYY/mem-shield}
}
```

A machine-readable citation file is also provided at [`CITATION.cff`](CITATION.cff).

---

## License

Released under the [MIT License](LICENSE).

---

## Acknowledgements

This work was conducted at the Indian Institute of Technology Roorkee and Mem0. The
LoCoMo and MM-BrowseComp base benchmarks belong to their respective authors; this
repository contributes the adversarial extensions (LoCoMo-Adv, MM-BrowseComp-Adv) and
the SAGE-Mem write-time defense.
