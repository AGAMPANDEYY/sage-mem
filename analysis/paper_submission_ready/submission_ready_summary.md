# Submission-Ready Analysis Summary

## Strongest Supported Results

- LoCoMo main: `SAGE-Mem v2` has `Write ASR=0.0000`, `Retrieval=0.0000`, `ASR=0.0000`.
- VPI-only: `SAGE-Mem v2` has `Write ASR=0.0000`, `ASR=0.0000`.
- Multimodal robustness: `SAGE-Mem v2` has `BCU poison=0.4367`, `Write ASR=0.0111`, `ASR=0.0050`.
- Browsing benchmark: `H6` has `BCU clean=0.1598`, `BCU poison=0.1512`, `Write ASR=0.0000`, `ASR=0.0000`.

## Browsing Comparison

- `MMA` collapses under attack: `BCU poison=0.0000`, `Write ASR=1.0000`, `ASR=1.0000`.
- `H5` reduces admission but not retrieval contamination: `Write ASR=0.3144`, `Retrieval=0.8265`, `ASR=0.8265`.
- `H6` is the strongest browsing defense: `Write ASR=0.0000`, `Retrieval=0.0000`, `ASR=0.0000`.

## Important Boundaries

- The current frozen canonical artifacts do not include per-row `seed` or `attack_type`, so true seed-CI and true per-attack tables still require a targeted rerun after the richer schema patch.
- The semantic observation-group rerun is useful as a secondary browsing-mechanism ablation, but it does not replace the canonical grouped H5/H6 pair.