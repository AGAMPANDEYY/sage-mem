# Final Paper Result Artifacts

This folder is the local paper artifact set copied from the authoritative EC2 runs.

These are the true latest runs currently copied into the repo. In particular:
- clean MM-BrowseComp = `paper_mmclean_h5_v1`
- adversarial MM-BrowseComp = `paper_mmadv_h5_v1`

The corrected MM-BrowseComp pair is `paper_mmclean_h5_v1` / `paper_mmadv_h5_v1`.

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
  - MM-BrowseComp clean, VLM-backed, includes `SAGEMemV2_BrowsingTrustPrior`
- `sagemem_mm_browsecomp_adversarial.json`
  - `paper_mmadv_h5_v1`
  - MM-BrowseComp adversarial, VLM-backed, includes `SAGEMemV2_BrowsingTrustPrior`

## Interpretation

Use the LoCoMo full runs as the primary evidence for the paper.

Use MM-BrowseComp as:
- external validation,
- a browsing-style multimodal stress test,
- and evidence that browsing-derived external text needs source-context-sensitive write priors.
