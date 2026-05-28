# Paper Analysis Artifacts

Frozen, small CSV/SVG files that back **every numerical claim** in the
ICML 2026 SCALE workshop paper *"Before It Persists: Write-Time Defense for
Multimodal Agent Memory"*.

Each cell printed in the paper traces back to one of the files here.

## Layout

```
analysis/
├── locomo_adv/              # LoCoMo-Adv benchmark artifacts
├── mm_browsecomp_adv/       # MM-BrowseComp-Adv benchmark artifacts
├── plots/                   # Pareto SVGs used in the appendix
├── systems_cost_table.csv   # Cross-benchmark systems cost summary
├── schema_gap_report.md     # Schema-gap audit note for reviewers
├── submission_ready_summary.md
└── README.md                # this file
```

## Paper element ↔ backing file map

### Main paper

| Paper element | Source file |
|---|---|
| **Table 1** (LoCoMo-Adv main) | [`locomo_adv/main_focus_seed_stats.csv`](locomo_adv/main_focus_seed_stats.csv) |
| **Table 2** (MM-BrowseComp-Adv, 5-attack rows) | [`mm_browsecomp_adv/`](mm_browsecomp_adv/) (canonical 5-attack appendix artifact) |
| **Table 2** (MM-BrowseComp-Adv, 2-attack rows) | [`mm_browsecomp_adv/browse_focus_seed_stats.csv`](mm_browsecomp_adv/browse_focus_seed_stats.csv) |
| Abstract numbers | Same as Table 1 and Table 2 above |

### Appendix

| Appendix element | Source file |
|---|---|
| **Table 5** (Seed stability) | [`locomo_adv/main_focus_seed_stats.csv`](locomo_adv/main_focus_seed_stats.csv) + [`mm_browsecomp_adv/browse_focus_seed_stats.csv`](mm_browsecomp_adv/browse_focus_seed_stats.csv) |
| **Table 7** (Five-attack extended benchmark) | [`mm_browsecomp_adv/browsing_adversarial_table.csv`](mm_browsecomp_adv/browsing_adversarial_table.csv) |
| **Table 8** (Clean utility) | [`mm_browsecomp_adv/browsing_clean_table.csv`](mm_browsecomp_adv/browsing_clean_table.csv) |
| **Table 9** (Per-attack breakdown narrative) | [`locomo_adv/main_focus_per_attack_breakdown.csv`](locomo_adv/main_focus_per_attack_breakdown.csv) + [`mm_browsecomp_adv/browse_focus_per_attack_breakdown.csv`](mm_browsecomp_adv/browse_focus_per_attack_breakdown.csv) |
| **Table 10** (Benign-write recall) | [`locomo_adv/main_focus_benign_write_recall.csv`](locomo_adv/main_focus_benign_write_recall.csv) + [`mm_browsecomp_adv/browse_focus_benign_write_recall.csv`](mm_browsecomp_adv/browse_focus_benign_write_recall.csv) |
| **Table 11** (K-sweep) | (single-run sweep artifact, not in this folder) |
| **Table 12** (Systems cost) | [`systems_cost_table.csv`](systems_cost_table.csv) |
| **Figure 4 (left)** Pareto LoCoMo-Adv | [`plots/pareto_locomo_bcu_vs_write_asr.svg`](plots/pareto_locomo_bcu_vs_write_asr.svg) |
| **Figure 4 (right)** Pareto MM-BrowseComp-Adv | [`plots/pareto_browsing_bcu_vs_write_asr.svg`](plots/pareto_browsing_bcu_vs_write_asr.svg) |
| Attack-proxy analysis | [`locomo_adv/attack_proxy_breakdown.csv`](locomo_adv/attack_proxy_breakdown.csv) |

### Limitations subsection (Write-time vs Retrieval-time contribution)

| Claim in paper | Source |
|---|---|
| "All five guarded methods admit the same 20 OCR injections" | [`locomo_adv/main_focus_per_attack_breakdown.csv`](locomo_adv/main_focus_per_attack_breakdown.csv) — `ocr_injection` row, 20 admitted across all of `RSum`, `H1`, `H2`, `H3`, `SAGE-Mem v2` |
| Per-method Retrieval ASR spread (0.008 / 0.005 / 0.003 / 0.100 / ≤0.002) | [`locomo_adv/main_focus_seed_stats.csv`](locomo_adv/main_focus_seed_stats.csv) — `ASR_mean` column |

## Regenerating these artifacts

From the repo root:

```bash
python scripts/paper_analysis.py
```

The script reads frozen result JSONs (kept on the maintainer's local machine,
gitignored from this repo) and writes back into this directory structure.

## Format notes

- **3 decimal places** throughout (matches the paper).
- **Rule-of-three upper bounds** (`≤ 0.002`, `≤ 0.003`) are computed in the
  paper's tables, not in these CSVs. The CSVs report observed event rates
  (`0.000` when no event was observed). See the paper's table captions for the
  conversion rule.
- Seed-mean CSVs include per-seed standard deviations, but the paper does not
  report `mean ± std` for ASR proportions near zero (the symmetric interval
  would extend below zero); see Appendix Table 5 caption.
