# Submission-Ready Analysis Summary

## Strongest Supported Results

- LoCoMo main: `SAGE-Mem v2` has `Write ASR=0.0000`, `Retrieval=0.0000`, `ASR=0.0000`.
- VPI-only: `SAGE-Mem v2` has `Write ASR=0.0000`, `ASR=0.0000`.
- Multimodal robustness: `SAGE-Mem v2` has `BCU poison=0.4367`, `Write ASR=0.0111`, `ASR=0.0050`.
- Browsing benchmark: `SAGE-Mem-BrowseGuard` has `BCU clean=0.1598`, `BCU poison=0.1512`, `Write ASR=0.0000`, `ASR=0.0000`.

## Browsing Comparison

- `MMA` collapses under attack: `BCU poison=0.0000`, `Write ASR=1.0000`, `ASR=1.0000`.
- `SAGE-Mem-Browse` reduces admission but not retrieval contamination: `Write ASR=0.3144`, `Retrieval=0.8265`, `ASR=0.8265`.
- `SAGE-Mem-BrowseGuard` is the strongest browsing defense: `Write ASR=0.0000`, `Retrieval=0.0000`, `ASR=0.0000`.

## Important Boundaries

- Focused richer-schema reruns are now available locally for the main LoCoMo suite and the adversarial browsing comparison. These support seed-aware summaries, per-attack breakdowns, and benign-write recall analysis.
- The canonical full artifact set still does not by itself provide seed-annotated rows across every benchmark family, so variance and per-attack claims should be tied explicitly to the focused reruns rather than backfilled everywhere.
- The semantic observation-group rerun is useful as a secondary browsing-mechanism ablation, but it does not replace the canonical grouped SAGE-Mem-Browse / SAGE-Mem-BrowseGuard pair.
