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

### Clean track

```bash
make full-mm-clean RUN_ID=paper_mmclean_h5_v1
```

Output:
- `results/paper_mmclean_h5_v1/sagemem_mm_browsecomp_clean.json`

### Adversarial track

```bash
make full-mm-adversarial RUN_ID=paper_mmadv_h5_v1
```

Output:
- `results/paper_mmadv_h5_v1/sagemem_mm_browsecomp_adversarial.json`

Interpretation guidance:
- these runs include `--include-browsing-prior`,
- `SAGEMemV2_BrowsingTrustPrior` is the explicit H5 condition,
- compare it against generic `SAGE-Mem v2` to test whether browsing-derived external text needs a source-context prior.

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
```

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
```

Then analyze:

```bash
make analyze-run RUN_ID=paper_main_full_v1
make analyze-run RUN_ID=paper_ablations_full_v1
make analyze-run RUN_ID=paper_vpi_full_v1
make analyze-run RUN_ID=paper_mmrobust_full_v1
make analyze-run RUN_ID=paper_mmclean_h5_v1
make analyze-run RUN_ID=paper_mmadv_h5_v1
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

The corrected H5 MM-BrowseComp pair is `paper_mmclean_h5_v1` / `paper_mmadv_h5_v1`.

## 11. Audit Commands

Validate the current filtered MM-BrowseComp dataset:

```bash
python3 scripts/audit_mm_browsecomp_dataset.py
```

Validate frozen result provenance against the current dataset:

```bash
python3 scripts/audit_final_results.py
```
