# Dataset Bundle

This folder is the publication-facing dataset bundle for the paper.

It is organized by benchmark family and separates:
- **base clean datasets** stored as files,
- **derived attacked variants** represented as generation manifests,
- **MM-BrowseComp traces / filtered clean cases** stored as concrete files.

## Layout

### `locomo/`

- `clean/locomo10.json`
  - base LoCoMo-10 dataset used by the paper
- `specs/main_untrusted_channel_suite.json`
  - manifest for the main attacked LoCoMo benchmark
- `specs/vpi_only.json`
  - manifest for the VPI-only LoCoMo benchmark
- `specs/multimodal_robustness.json`
  - manifest for the noisy/missing-modality LoCoMo benchmark

Important:
- attacked LoCoMo variants are **generated on the fly** from `locomo10.json`
- they are not stored as static duplicated corpora because attack position, seed, and suite definition are part of the experimental design

### `mm_browsecomp/`

- `official/MMBrowseComp_400.jsonl`
  - official encrypted benchmark rows
- `official/MMBrowseComp_400_decrypted.jsonl`
  - decrypted task rows used during preprocessing
- `traces/mm_browsecomp_traces_all.jsonl`
  - fetched observation traces
- `clean/mm_browsecomp_cases_filtered.jsonl`
  - quality-filtered clean cases used for MM-BrowseComp runs
  - current regenerated pool: 194 cases
  - filters: junk/blocked-page removal, duplicate removal, `min-good-obs=2`
    with one official image counted as one future VLM observation,
    `min-total-chars=300`, `min-ocr-chars=60`, `min-tool-chars=150`,
    and answer/checklist leakage dropping
- `specs/adversarial_fact_overwrite.json`
  - manifest for the current adversarial MM-BrowseComp benchmark

## Why manifests instead of duplicated attacked corpora?

For LoCoMo and MM-BrowseComp adversarial evaluation, the attacked benchmark is defined by:
- the clean source dataset,
- the attack suite,
- injection policy / position mode,
- seed set,
- and any multimodal augmentation mode.

That means the attacked dataset is a **programmatically generated benchmark configuration**, not a single immutable raw file.

## Current authoritative experiment mappings

- LoCoMo main: `paper_main_full_v1`
- LoCoMo ablations: `paper_ablations_full_v1`
- LoCoMo VPI: `paper_vpi_full_v1`
- LoCoMo multimodal robustness: `paper_mmrobust_full_v1`
- MM-BrowseComp clean: `paper_mmclean_full_v4`
- MM-BrowseComp adversarial: `paper_mmadv_full_v3`
