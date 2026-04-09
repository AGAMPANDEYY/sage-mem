# SAGE-Mem: Write-Time Governance for Multimodal Agent Memory

Research code for an ICML 2026 SCALE workshop submission on **robustness in multimodal noisy memory for AI agents**.

SAGE-Mem is a **governed memory layer**, not a complete autonomous agent stack. It sits between heterogeneous observation sources and a downstream planner, and controls:
- what gets written into memory,
- what gets promoted from `evidence` into durable `belief`,
- what retrieved memory is allowed to support downstream reasoning.

The repository studies a specific question:

> How should a multimodal agent defend its persistent memory against adversarial and noisy observations, especially when OCR outputs, image captions, and tool outputs can be mutually dependent or conflicting?

---

## Why This Matters

Retrieval-time defenses help rank or filter memories at query time, but they do **not** protect the memory state itself. Once a poisoned observation has entered memory, it can:
- survive consolidation,
- distort later summaries,
- occupy token budget,
- increase retrieval noise,
- and repeatedly burden downstream reranking.

For multimodal agents this is worse: OCR and caption outputs from the **same image** may look like corroborating evidence even though they are not independent. SAGE-Mem treats this as a memory-governance problem at the write and belief boundaries, not only a retrieval problem.

---

## What SAGE-Mem Does

SAGE-Mem combines:
- **typed memory**: `evidence`, `belief`, `control`
- **write-time admission control**
- **sufficiency-gated evidence → belief promotion**
- **Bayesian channel trust**
- **session-relative anomaly detection**
- **consistency-graph checks**
- **dependent-evidence handling for multimodal inputs**
- **provenance-aware retrieval**

This design is meant to protect the **persistent store**, not just the current prompt.

---

## Main Research Claims

The strongest claims supported by the current evidence are:

1. **Write-time governed memory is a distinct robustness primitive.**
   On controlled long-horizon multimodal memory benchmarks, SAGE-Mem sharply reduces attack write admission and downstream retrieval contamination compared with retrieval-time filtering alone.

2. **Multimodal agreement is not always evidence of truth.**
   OCR and caption outputs derived from the same adversarial image can create false corroboration; Visual Prompt Injection (VPI) is a concrete instance of this failure mode.

3. **Lifecycle metrics are required for fair evaluation.**
   A write-time defense should not be judged only by retrieval-time ASR or final QA utility. The evaluation should separately measure admission, belief formation, retrieval contamination, and downstream completion.

4. **The utility-robustness tradeoff is real.**
   MMA, the retrieval-time baseline, preserves higher QA utility. SAGE-Mem provides stronger memory-state integrity.

This repo does **not** support the stronger claim that SAGE-Mem is the best overall memory system on all metrics.

---

## Benchmarks

### 1. LoCoMo-10 with multimodal adversarial extension

Used for the paper’s primary evidence.

Tracks:
- `main`: main untrusted-channel attack suite
- `visual_prompt_injection`: multimodal false-corroboration attack
- `multimodal_robustness`: missing/noisy modality stress

These are controlled long-horizon memory benchmarks, not natural browsing traces.

### 2. MM-BrowseComp with observation-trace augmentation

Used as an external browsing-style multimodal stress test.

Tracks:
- `clean`
- `adversarial`

This benchmark is more realistic as a browsing-memory setting, but currently remains much harsher and less discriminative than the LoCoMo-based evaluations. It should be interpreted carefully.

---

## Primary Metrics

SAGE-Mem is a **write-time** defense, so the main metrics are organized by memory lifecycle stage.

### Write / memory-formation metrics
- `attack_write_admission_rate`
- `attack_belief_formation_rate`
- `write_quarantine_per_case`

### Retrieval / corruption metrics
- `attack_retrieval_rate`
- `false_belief_rate`
- `belief_traceability_score`
- `ASR`

### Downstream utility metric
- `BenignCompletionUnderAttack (BCU)`

BCU is defined as the **joint per-QA indicator**
\[
\mathbf{1}[\text{answer consistent} \land \neg \text{attack survived}]
\]
averaged over evaluated QA pairs.

Secondary metrics such as behavioral LLM ASR are useful, but they are not the core evidence for this paper.

---

## Current Empirical Picture

### Strongest evidence

The strongest paper-quality evidence comes from the full LoCoMo runs:
- `paper_main_full_v1`
- `paper_vpi_full_v1`
- `paper_mmrobust_full_v1`
- `paper_ablations_full_v1`

Key pattern:
- **MMA** preserves higher clean and attacked QA utility
- **SAGE-Mem** strongly improves write-time containment and retrieval integrity

### Main LoCoMo result

| Method | BCU clean | BCU poison | Write ASR | Retrieval | ASR |
|---|---:|---:|---:|---:|---:|
| MMA | 0.7500 | 0.6417 | 1.0000 | 0.1333 | 0.1333 |
| SAGE-Mem | 0.3950 | 0.4600 | 0.0000 | 0.0000 | 0.0000 |

### VPI result

| Method | BCU clean | BCU poison | Write ASR | Retrieval | ASR |
|---|---:|---:|---:|---:|---:|
| MMA | 0.7650 | 0.7117 | 1.0000 | 0.0700 | 0.0700 |
| SAGE-Mem | 0.3800 | 0.3800 | 0.0000 | 0.0000 | 0.0000 |

### MM robustness result

| Method | BCU clean | BCU poison | Write ASR | Retrieval | ASR |
|---|---:|---:|---:|---:|---:|
| MMA | 0.7117 | 0.5483 | 1.0000 | 0.2600 | 0.2600 |
| SAGE-Mem | 0.3950 | 0.4367 | 0.0111 | 0.0050 | 0.0050 |

### MM-BrowseComp

MM-BrowseComp is now technically correct with VLM-backed image observations, but the adversarial run remains highly saturated and should currently be treated as:
- an external stress test,
- useful evidence of a browsing-specific limitation,
- not the main anchor of the paper.

---

## Authoritative Result Provenance

The local paper artifact set is:

- [results/final_paper_results_20260410](/Users/agampandey/work/mem-shield/results/final_paper_results_20260410)

Use the following mapping as the authoritative latest result for each experiment family:

| Local artifact | Source run ID | Notes |
|---|---|---|
| `sagemem_main_llm.json` | `paper_main_full_v1` | main LoCoMo + LLM-eval |
| `sagemem_v2_ablations.json` | `paper_ablations_full_v1` | main LoCoMo ablations |
| `sagemem_vpi_llm.json` | `paper_vpi_full_v1` | VPI-only run |
| `sagemem_multimodal_robustness_ablations.json` | `paper_mmrobust_full_v1` | noisy/missing-modality robustness |
| `sagemem_mm_browsecomp_clean.json` | `paper_mmclean_full_v4` | latest clean MM-BrowseComp rerun after the VLM-caption fix |
| `sagemem_mm_browsecomp_adversarial.json` | `paper_mmadv_full_v3` | latest adversarial MM-BrowseComp rerun after the VLM-caption fix |

There is intentionally no `paper_mmadv_full_v4`. The clean and adversarial MM-BrowseComp runs have different version numbers because the clean track required one extra rerun before the final VLM-backed pair was frozen.

---

## Project Layout

```text
configs/
  default_trust_config.json

data/
  MM-BrowseComp/
  mm_browsecomp_traces_all.jsonl
  mm_browsecomp_cases_filtered.jsonl

datasets/
  publication-facing dataset bundle
  - clean LoCoMo source
  - LoCoMo attack-generation manifests
  - MM-BrowseComp official rows
  - MM-BrowseComp traces
  - filtered clean MM-BrowseComp cases
  - adversarial benchmark manifests

results/
  <RUN_ID>/

scripts/
  analyze_run.py

src/
  mma_bench_suite.py
  eval_judge.py
  build_mm_browsecomp_traces.py
  prepare_mm_browsecomp_cases.py

run_eval.py
Makefile
paper_draft.md
COMMANDS.md
```

---

## Reproducible Workflow

Use `make` as the primary interface.

### 1. Tests

```bash
make test
```

### 2. Core LoCoMo paper runs

```bash
make full-main-llm RUN_ID=paper_main_full_v1
make full-v2-ablations RUN_ID=paper_ablations_full_v1
make full-vpi-llm RUN_ID=paper_vpi_full_v1
make full-mm-robust-ablations RUN_ID=paper_mmrobust_full_v1
```

### 3. MM-BrowseComp pipeline

Build traces:

```bash
make build-mm-traces
make filter-mm-cases
```

Run clean and adversarial tracks separately:

```bash
make full-mm-clean RUN_ID=paper_mmclean_full_v4
make full-mm-adversarial RUN_ID=paper_mmadv_full_v3
```

### 4. Analysis

```bash
make analyze-run RUN_ID=paper_main_full_v1
```

The analyzer reads only `results/<RUN_ID>/` and renders a Rich summary table for every JSON in that folder.

---

## MM-BrowseComp Caveat

MM-BrowseComp should currently be described carefully.

What is true:
- the clean and adversarial tracks are now separate,
- image observations are captioned with OpenAI vision,
- the benchmark is useful as an external stress test.

What is **not** yet true:
- the adversarial setting does not yet provide a strong, well-separated main-paper comparison,
- the current browsing-style `fact_overwrite_injection` is admitted by all methods,
- end-to-end adversarial performance remains near-collapse for every method.

So MM-BrowseComp is useful, but should not currently be oversold.

---

## Paper-Appropriate Positioning

The most defensible workshop framing is:

> SAGE-Mem introduces write-time governance for multimodal agent memory and shows that preventing unsupported observations from hardening into durable beliefs yields stronger memory-state integrity than retrieval-only filtering, especially under multimodal prompt injection and noisy/missing modalities, albeit with a utility tradeoff.

That is the claim this repository currently supports.
