# Commands

Use `make` as the primary interface.

This file is organized around the actual research workflow:
1. environment sanity,
2. benchmark construction,
3. smoke runs,
4. full paper runs,
5. analysis.

---

## 1. Sanity Checks

```bash
make test
make print-run-dir
```

Important defaults:
- `PYTHON=./.venv/bin/python`
- `SEEDS="0 1 2"`
- `POSITION_MODE=random`
- `CHECKPOINT_EVERY=25`
- `MAX_WORKERS=4`
- `RUN_ID` defaults to a timestamp
- `RUN_DIR` defaults to `results/$RUN_ID`

Useful overrides:

```bash
make full-main MAX_WORKERS=1
make full-main CASE_FRACTION=0.5 CASE_SAMPLE_SEED=17
make full-main-llm RESUME=1 RUN_ID=paper_main_full_v1
```

---

## 2. MM-BrowseComp Data Pipeline

### Build traces from the official 400-case file

```bash
make build-mm-traces
```

Equivalent direct call:

```bash
.venv/bin/python src/build_mm_browsecomp_traces.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --out data/mm_browsecomp_traces_all.jsonl \
  --max-workers 8
```

### Filter and prepare augmented cases

```bash
make filter-mm-cases
```

This produces:
- `data/mm_browsecomp_cases_filtered.jsonl`

The current filter requires:
- at least 2 clean observations,
- minimum OCR/tool-text quality thresholds,
- no obvious JS/YouTube/title-only junk stubs,
- deduplication within cases.

---

## 3. Smoke Runs

### LoCoMo

```bash
make smoke-main
make smoke-main-llm
make smoke-vpi
make smoke-vpi-llm
make smoke-v2-ablations
make smoke-mm-robust
make smoke-stress-llm
```

### MM-BrowseComp

```bash
make smoke-mm-clean
make smoke-mm-adversarial
```

Notes:
- MM-BrowseComp smoke/full runs use `--vision-caption-mode openai`
- clean and adversarial are intentionally split

---

## 4. Full Paper Runs

These are the core runs that currently support the paper.

### Main LoCoMo

```bash
make full-main-llm RUN_ID=paper_main_full_v1
```

Output:
- `results/paper_main_full_v1/sagemem_main_llm.json`

### Main ablations

```bash
make full-v2-ablations RUN_ID=paper_ablations_full_v1
```

Output:
- `results/paper_ablations_full_v1/sagemem_v2_ablations.json`

### Visual Prompt Injection

```bash
make full-vpi-llm RUN_ID=paper_vpi_full_v1
```

Output:
- `results/paper_vpi_full_v1/sagemem_vpi_llm.json`

### Multimodal robustness (missing/noisy modalities)

```bash
make full-mm-robust-ablations RUN_ID=paper_mmrobust_full_v1
```

Output:
- `results/paper_mmrobust_full_v1/sagemem_multimodal_robustness_ablations.json`

---

## 5. MM-BrowseComp Full Runs

### Baseline clean track

```bash
make full-mm-clean RUN_ID=paper_mmclean_h5_v1
```

Output:
- `results/paper_mmclean_h5_v1/sagemem_mm_browsecomp_clean.json`

### Baseline adversarial track

```bash
make full-mm-adversarial RUN_ID=paper_mmadv_h5_v1
```

Output:
- `results/paper_mmadv_h5_v1/sagemem_mm_browsecomp_adversarial.json`

Interpretation guidance:
- these runs include `--include-browsing-prior`,
- `SAGEMemV2_BrowsingTrustPrior` is the explicit H5 condition,
- compare it against generic `SAGE-Mem v2` to test whether browsing-derived external text needs a source-context prior.

### Canonical grouped H5/H6 browsing runs

```bash
make full-mm-abr-clean RUN_ID=paper_mmclean_abr_group_v1
make full-mm-abr-adversarial RUN_ID=paper_mmadv_abr_group_v1
```

Outputs:
- `results/paper_mmclean_abr_group_v1/sagemem_mm_browsecomp_abr_clean.json`
- `results/paper_mmadv_abr_group_v1/sagemem_mm_browsecomp_abr_adversarial.json`

Interpretation guidance:
- these are the canonical browsing runs for the paper,
- they include the grouped H5/H6 comparison,
- `SAGEMemV2_BrowsingTrustPrior` is H5,
- `SAGEMemV2_ABR` is H6,
- use these runs, not the older H5-only pair, when citing browsing results in the draft.

### Secondary semantic observation-group browsing reruns

These were run directly on EC2 with the semantic observation-group path enabled and then copied back into the local final artifact folder.

Source run IDs:
- `paper_mmclean_abr_sem_v1`
- `paper_mmadv_abr_sem_v1`

Local artifact paths:
- [sagemem_mm_browsecomp_abr_clean_semantic.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_abr_clean_semantic.json)
- [sagemem_mm_browsecomp_abr_adversarial_semantic.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_abr_adversarial_semantic.json)

Interpretation guidance:
- these are secondary mechanism ablations,
- they preserve the main H6 result,
- they are not the canonical paper pair,
- use them only when discussing page-group signal behavior or semantic rerun stability.

---

## 6. Analysis

Analyze any run folder:

```bash
make analyze-run RUN_ID=paper_main_full_v1
```

Examples:

```bash
make analyze-run RUN_ID=paper_main_full_v1
make analyze-run RUN_ID=paper_ablations_full_v1
make analyze-run RUN_ID=paper_vpi_full_v1
make analyze-run RUN_ID=paper_mmrobust_full_v1
make analyze-run RUN_ID=paper_mmclean_h5_v1
make analyze-run RUN_ID=paper_mmadv_h5_v1
make analyze-run RUN_ID=paper_mmclean_abr_group_v1
make analyze-run RUN_ID=paper_mmadv_abr_group_v1
```

For the copied semantic reruns, inspect the local final artifact files directly or load them with a custom comparison script; they are not stored under a local `results/<RUN_ID>/` folder.

### Submission-ready paper bundle

Generate the paper-facing summary tables, systems-cost table, and Pareto SVGs from the frozen final result artifacts:

```bash
./.venv/bin/python scripts/paper_analysis.py
```

Outputs:

```bash
analysis/paper_submission_ready/submission_ready_summary.md
analysis/paper_submission_ready/schema_gap_report.md
analysis/paper_submission_ready/main_clean_table.csv
analysis/paper_submission_ready/main_poison_table.csv
analysis/paper_submission_ready/browsing_clean_table.csv
analysis/paper_submission_ready/browsing_adversarial_table.csv
analysis/paper_submission_ready/systems_cost_table.csv
analysis/paper_submission_ready/pareto_locomo_bcu_vs_write_asr.svg
analysis/paper_submission_ready/pareto_browsing_bcu_vs_write_asr.svg
```

Interpretation guidance:
- use these files for the current paper draft and figures
- do not back-fill unsupported per-attack or seed-CI claims from the frozen canonical JSONs
- read `schema_gap_report.md` before deciding whether an additional rerun is worth the compute

---

## 7. Resume and Partial Runs

Use a fraction for bounded pilots:

```bash
make full-main-llm RUN_ID=pilot_main_p50 CASE_FRACTION=0.5 CASE_SAMPLE_SEED=17
```

Resume a run:

```bash
make full-main-llm RUN_ID=pilot_main_p50 RESUME=1
```

Important:
- if metric definitions or saved fields change, do **not** resume old runs
- start a fresh `RUN_ID` after any benchmark-logic change

---

## 8. Recommended Submission Workflow

If you need to rerun the full evidence set cleanly:

```bash
make test
make full-main-llm RUN_ID=paper_main_full_v1
make full-v2-ablations RUN_ID=paper_ablations_full_v1
make full-vpi-llm RUN_ID=paper_vpi_full_v1
make full-mm-robust-ablations RUN_ID=paper_mmrobust_full_v1
make full-mm-clean RUN_ID=paper_mmclean_h5_v1
make full-mm-adversarial RUN_ID=paper_mmadv_h5_v1
make full-mm-abr-clean RUN_ID=paper_mmclean_abr_group_v1
make full-mm-abr-adversarial RUN_ID=paper_mmadv_abr_group_v1
```

Then analyze:

```bash
make analyze-run RUN_ID=paper_main_full_v1
make analyze-run RUN_ID=paper_ablations_full_v1
make analyze-run RUN_ID=paper_vpi_full_v1
make analyze-run RUN_ID=paper_mmrobust_full_v1
make analyze-run RUN_ID=paper_mmclean_h5_v1
make analyze-run RUN_ID=paper_mmadv_h5_v1
make analyze-run RUN_ID=paper_mmclean_abr_group_v1
make analyze-run RUN_ID=paper_mmadv_abr_group_v1
```

---

## 9. What Is Main-Table Ready

Main paper evidence:
- `paper_main_full_v1`
- `paper_ablations_full_v1`
- `paper_vpi_full_v1`
- `paper_mmrobust_full_v1`

Appendix / external stress test:
- `paper_mmclean_h5_v1`
- `paper_mmadv_h5_v1`

Canonical browsing comparison for the paper:
- `paper_mmclean_abr_group_v1`
- `paper_mmadv_abr_group_v1`

Secondary browsing mechanism ablation:
- `paper_mmclean_abr_sem_v1`
- `paper_mmadv_abr_sem_v1`

## 10. Latest Result Mapping

The tracked local final artifact folder is:

- [final_paper_results_20260410](/Users/agampandey/work/mem-shield/final_paper_results_20260410)

Use this provenance mapping when citing results in the draft:

| Local file | Source run ID |
|---|---|
| `sagemem_main_llm.json` | `paper_main_full_v1` |
| `sagemem_v2_ablations.json` | `paper_ablations_full_v1` |
| `sagemem_vpi_llm.json` | `paper_vpi_full_v1` |
| `sagemem_multimodal_robustness_ablations.json` | `paper_mmrobust_full_v1` |
| `sagemem_mm_browsecomp_clean.json` | `paper_mmclean_h5_v1` |
| `sagemem_mm_browsecomp_adversarial.json` | `paper_mmadv_h5_v1` |
| `sagemem_mm_browsecomp_abr_clean.json` | `paper_mmclean_abr_group_v1` |
| `sagemem_mm_browsecomp_abr_adversarial.json` | `paper_mmadv_abr_group_v1` |
| `sagemem_mm_browsecomp_abr_clean_semantic.json` | `paper_mmclean_abr_sem_v1` |
| `sagemem_mm_browsecomp_abr_adversarial_semantic.json` | `paper_mmadv_abr_sem_v1` |

For the paper, the canonical browsing pair is `paper_mmclean_abr_group_v1` / `paper_mmadv_abr_group_v1`.

Exact local final artifact paths:
- [sagemem_main_llm.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_main_llm.json)
- [sagemem_v2_ablations.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_v2_ablations.json)
- [sagemem_vpi_llm.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_vpi_llm.json)
- [sagemem_multimodal_robustness_ablations.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_multimodal_robustness_ablations.json)
- [sagemem_mm_browsecomp_clean.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_clean.json)
- [sagemem_mm_browsecomp_adversarial.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_adversarial.json)
- [sagemem_mm_browsecomp_abr_clean.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_abr_clean.json)
- [sagemem_mm_browsecomp_abr_adversarial.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_abr_adversarial.json)
- [sagemem_mm_browsecomp_abr_clean_semantic.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_abr_clean_semantic.json)
- [sagemem_mm_browsecomp_abr_adversarial_semantic.json](/Users/agampandey/work/mem-shield/final_paper_results_20260410/sagemem_mm_browsecomp_abr_adversarial_semantic.json)

## 11. Audit Commands

Validate the current filtered MM-BrowseComp dataset:

```bash
python3 scripts/audit_mm_browsecomp_dataset.py
```

Validate frozen result provenance against the current dataset:

```bash
python3 scripts/audit_final_results.py
```

Frozen MM-BrowseComp augmented-case cache used on EC2:

```text
/home/ec2-user/mem-shield-full/data/mm_browsecomp_cases_augmented_openai.jsonl
```
