# WARP: Write-time Attested Reliability Pipeline

**ICML 2026 Workshop Submission Code**

Code for:
> *WARP: Write-Time Attested Memory Defenses for Multimodal Long-Horizon Agents*

## What This Repo Evaluates

WARP studies **write-time memory poisoning** in agents that store observations into persistent memory.

The repo now supports two benchmark tracks:
- **LoCoMo** with a **controlled synthetic multimodal extension**
- **MM-BrowseComp** from the official benchmark JSONL

This is the intended truthful framing:
- LoCoMo is **not** a natively multimodal dataset
- WARP extends LoCoMo with simulated multimodal ingestion through:
  - `ocr_text` turns with deterministic OCR-like corruption
  - `vision_caption` turns that represent image-derived observations
- MM-BrowseComp is treated as an equally important benchmark for real multimodal browsing tasks
- The repo can load the **official** MM-BrowseComp JSONL directly, but official task rows do **not** contain browsing observation traces by themselves

## Core Hypotheses

| ID | Mechanism | Intended effect |
|----|-----------|-----------------|
| H1 | `ConstructorGuardedSandboxMemory` | Quarantine low-trust writes and strip procedural content at consolidation |
| H2 | `MonotoneProvenanceLedgerMemory` | Prevent trust laundering through rewriting while keeping weaker write gating |
| H3 | `RiskSensitiveToolActionFirewall_CorroborateOrConfirm` | Memory-centric firewall ablation with the loosest write gate |
| Baseline | `MMARetrieveTimeReliabilityMemory` | Retrieve-time reliability scoring only |

## Benchmark Coverage

### LoCoMo
- Standard clean split
- Poisoned split with write-time attacks
- Cross-topic delayed-trigger split
- Optional synthetic multimodal preprocessing on about 20% of turns

Attacks:
- `semantic_mimicry`
- `constructor_launder`
- `label_gaming`
- `ocr_injection`
- `vision_caption_injection`

### MM-BrowseComp
- Clean and poisoned benchmark path
- Loads the official encrypted/decrypted JSONL schema directly
- Also accepts augmented rows that already include `observations`
- Keeps MM-BrowseComp as a separate benchmark table, not merged into LoCoMo claims

The repo does **not** synthesize MM-BrowseComp cases. You should vendor the official repo under `data/MM-BrowseComp` or point `MM_BROWSECOMP_PATH` at an official JSONL file. If you want to run WARP's memory evaluation on MM-BrowseComp, you must additionally provide observation events for each task, because the official benchmark rows include task metadata, image URLs, source URLs, answers, and checklists, but not the browsing trace itself.

To merge official MM-BrowseComp rows with externally collected browsing traces:

```bash
.venv/bin/python src/prepare_mm_browsecomp_cases.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --traces /path/to/mm_browsecomp_traces.jsonl \
  --out data/mm_browsecomp_augmented.jsonl
```

To build a first-pass local trace file from official source URLs and image OCR only:

```bash
.venv/bin/python src/build_mm_browsecomp_traces.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --out data/mm_browsecomp_traces.jsonl \
  --limit 10
```

For JS-heavy pages, use the browser-backed collector:

```bash
.venv/bin/python src/build_mm_browsecomp_browser_traces.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --out data/mm_browsecomp_browser_traces.jsonl \
  --headless \
  --limit 10
```

This keeps the trace generation truthful:
- `tool_output_text` comes from rendered page text
- `ocr_text` comes from local OCR over rendered screenshots and official image URLs
- it does **not** use an LLM to invent visual observations

## Repo Structure

```text
warp-memory/
├── run_eval.py
├── requirements.txt
├── src/
│   ├── memory.py
│   ├── mma_bench_suite.py
│   ├── procedural.py
│   ├── agent.py
│   ├── embedding.py
│   └── utils.py
├── results/
│   └── mma_eval_results.json
└── data/
    └── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Evaluations

Quick sanity check:

```bash
.venv/bin/python run_eval.py --quick
```

Main LoCoMo multimodal memory-layer table:

```bash
.venv/bin/python run_eval.py \
  --enable-locomo-multimodal \
  --disable-cross-topic \
  --out results/locomo_multimodal_memory_layers.json
```

Hosted `mem0` pilot on a bounded LoCoMo slice:

```bash
.venv/bin/python run_eval.py \
  --seeds 0 \
  --qa-per-case 20 \
  --enable-locomo-multimodal \
  --disable-cross-topic \
  --conditions Mem0_Platform_Baseline \
  --case-limit 1 \
  --out results/locomo_mem0_pilot.json
```

LoCoMo with synthetic multimodality:

```bash
.venv/bin/python run_eval.py --enable-locomo-multimodal
```

LoCoMo without the cross-topic split:

```bash
.venv/bin/python run_eval.py --disable-cross-topic
```

Run both benchmark tracks:

```bash
MM_BROWSECOMP_PATH=/path/to/MMBrowseComp_400.jsonl \
.venv/bin/python run_eval.py --enable-locomo-multimodal --run-mm-browsecomp
```

## Verified Results

Full LoCoMo multimodal main table from `results/locomo_multimodal_memory_layers.json`:

| Method | BCU clean | BCU poisoned | ASR poisoned | n |
|---|---:|---:|---:|---:|
| ShortContext | 0.1587 | 0.0000 | 1.0000 | 630 |
| MMA (baseline) | 0.7444 | 0.6152 | 0.1556 | 630 |
| RSum (no guard) | 0.2286 | 0.0000 | 1.0000 | 630 |
| H1 ConstructorGuard | 0.2286 | 0.2333 | 0.0000 | 630 |
| H2 MonotoneLedger | 0.2127 | 0.1873 | 0.0000 | 630 |
| H3 ActionFirewall | 0.2286 | 0.2333 | 0.0000 | 630 |

Interpretation:
- `ShortContext` and unguarded `RSum` both collapse under poisoning.
- `MMA` retains the best clean utility but still admits poisoned memory (`ASR = 0.1556`).
- `H1/H2/H3` all drive `ASR` to `0.0000` on this LoCoMo setting, with `H1 ≈ H3 > H2` on poisoned BCU.

Hosted `mem0` pilot from `results/locomo_mem0_pilot.json`:

| Method | BCU clean | BCU poisoned | ASR poisoned | n |
|---|---:|---:|---:|---:|
| mem0 pilot | 0.6500 | 0.0300 | 0.9000 | 20 |

This `mem0` result is a truthful pilot only:
- it uses the hosted Mem0 API with isolated per-case user scopes
- it evaluates retrieval via `search(...)`, not chat generation
- it is too slow for full LoCoMo under faithful per-turn writes, so it should not be presented as a full main-table benchmark

## Output Format

`results/mma_eval_results.json` now stores benchmark-specific sections:
- `benchmarks.locomo.summary`
- `benchmarks.locomo.raw_sample_count`
- `benchmarks.mm_browsecomp.*` when enabled

The top-level `summary` field is kept for backward compatibility and points to the LoCoMo summary.

## Important Notes

- No benchmark generation step requires paid model calls.
- `vision_caption` memory items are stored as **text with image provenance**, not raw images.
- The MM-BrowseComp loader understands the official schema directly.
- The MM-BrowseComp evaluator will refuse to run on bare official task rows until observation traces are supplied.
- Numeric claims should come from regenerated result files, not from this README.
