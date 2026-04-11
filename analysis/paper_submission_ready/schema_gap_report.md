# Schema Gap Report

The current frozen canonical artifacts are sufficient for core summary tables, systems-cost tables, browsing mechanism comparisons, and Pareto figures.

The following reviewer-facing analyses cannot be computed honestly from the current frozen raws because the fields are not present in saved rows:

- `seed`
- `attack_type`
- `attack_types`
- `attack_write_attempt_count_by_type`
- `benign_write_admitted_count`

Implications:
- true per-attack breakdown requires saved attack labels per QA row or per case unit
- true mean±std / CI over seeds requires saved seed identifiers per QA row or per case unit
- benign write recall requires saved benign write admission counts

Recommended next rerun policy:
- do not rerun full suites immediately
- first stabilize the richer result schema
- then rerun only the smallest paper-critical subset needed for each missing table