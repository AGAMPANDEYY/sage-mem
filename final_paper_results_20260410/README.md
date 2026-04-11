# Final Paper Result Artifacts

This folder is the local paper artifact set copied from the authoritative EC2 runs.

These are the true latest runs currently copied into the repo. In particular:
- clean MM-BrowseComp baseline = `paper_mmclean_h5_v1`
- adversarial MM-BrowseComp baseline = `paper_mmadv_h5_v1`
- clean MM-BrowseComp H5/H6 grouped run = `paper_mmclean_abr_group_v1`
- adversarial MM-BrowseComp H5/H6 grouped run = `paper_mmadv_abr_group_v1`

For the paper, use the grouped H5/H6 pair as the canonical browsing comparison:
- `sagemem_mm_browsecomp_abr_clean.json`
- `sagemem_mm_browsecomp_abr_adversarial.json`

Additional browsing ablation copied from EC2:
- `sagemem_mm_browsecomp_abr_clean_semantic.json`
- `sagemem_mm_browsecomp_abr_adversarial_semantic.json`

These semantic observation-group reruns are useful for mechanism analysis, but they are **not** the canonical paper pair. The semantic rerun preserves the main H6 result (`Write ASR=0.000`, `ASR=0.000`) while changing the page-group signal statistics only modestly. It does not overturn the main interpretation that the structured browser claim gate is the dominant mechanism.

Important audit note:
- The MM-BrowseComp artifacts below are the corrected reruns on the 194-case
  leakage-clean pool after enforcing `--min-good-obs` and dropping
  answer/checklist leakage.

## Main-table runs

- `sagemem_main_llm.json`
  - `paper_main_full_v1`
  - main LoCoMo + LLM-eval
- `sagemem_v2_ablations.json`
  - `paper_ablations_full_v1`
  - main LoCoMo ablations
- `sagemem_vpi_llm.json`
  - `paper_vpi_full_v1`
  - VPI-only multimodal run
- `sagemem_multimodal_robustness_ablations.json`
  - `paper_mmrobust_full_v1`
  - noisy/missing-modality robustness run

## Appendix / external stress-test runs

- `sagemem_mm_browsecomp_clean.json`
  - `paper_mmclean_h5_v1`
  - MM-BrowseComp clean baseline, VLM-backed, includes `SAGEMemV2_BrowsingTrustPrior`
- `sagemem_mm_browsecomp_adversarial.json`
  - `paper_mmadv_h5_v1`
  - MM-BrowseComp adversarial baseline, VLM-backed, includes `SAGEMemV2_BrowsingTrustPrior`
- `sagemem_mm_browsecomp_abr_clean.json`
  - `paper_mmclean_abr_group_v1`
  - MM-BrowseComp clean grouped browsing run with H5 and H6/ABR
- `sagemem_mm_browsecomp_abr_adversarial.json`
  - `paper_mmadv_abr_group_v1`
  - MM-BrowseComp adversarial grouped browsing run with injection + adaptive attacks and H5/H6 comparison
- `sagemem_mm_browsecomp_abr_clean_semantic.json`
  - `paper_mmclean_abr_sem_v1`
  - semantic observation-group clean rerun; secondary ablation, not canonical
- `sagemem_mm_browsecomp_abr_adversarial_semantic.json`
  - `paper_mmadv_abr_sem_v1`
  - semantic observation-group adversarial rerun; secondary ablation, not canonical

## Interpretation

Use the LoCoMo full runs as the primary evidence for the paper.

Use MM-BrowseComp as:
- external validation,
- a browsing-style multimodal stress test,
- and evidence that browsing-derived external text needs source-context-sensitive write priors and browser-specific claim typing.
