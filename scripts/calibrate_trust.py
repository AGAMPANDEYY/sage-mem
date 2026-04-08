"""
scripts/calibrate_trust.py — Offline trust calibration from clean-split eval results.

Workflow:
  1. Run run_eval.py with --log-retrieved-source-types flag (adds per-item logging).
  2. This script reads the resulting JSON and computes empirical source credibility.
  3. Writes an updated config to configs/calibrated_trust_config.json.

Usage:
  python scripts/calibrate_trust.py \
    --results results/combined_locomo_mm_v2.json \
    --output configs/calibrated_trust_config.json \
    --min-count 10

The --min-count argument sets the minimum number of observations for a source_type
to be updated. Types with fewer observations keep their prior values.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trust_calibration import SourceCredibilityCalibrator


def main():
    parser = argparse.ArgumentParser(description="Calibrate source trust from eval results.")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to eval results JSON (must have raw_per_item key).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/calibrated_trust_config.json"),
        help="Output path for calibrated config JSON.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Base config to update. Defaults to configs/default_trust_config.json.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum observations per source_type to update trust value.",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"[ERROR] Results file not found: {args.results}")
        sys.exit(1)

    cal = SourceCredibilityCalibrator()
    cal.add_results(args.results)

    empirical = cal.empirical_cred(min_count=args.min_count)
    if not empirical:
        print(
            f"[WARNING] No source types with >= {args.min_count} observations found. "
            "Check that your eval results include 'raw_per_item' logging. "
            "Run run_eval.py with --log-retrieved-source-types to enable this."
        )
        sys.exit(0)

    print(f"\n[calibrator] Empirical source credibility:")
    for st, cred in sorted(empirical.items()):
        print(f"  {st:30s}: {cred:.4f}")

    cal.write_config(
        output_path=args.output,
        base_config_path=args.base_config,
        min_count=args.min_count,
    )

    print(f"\n[calibrator] Done. Use this config with:")
    print(f"  SAGEMemConfig.from_file(Path('{args.output}'))")


if __name__ == "__main__":
    main()
