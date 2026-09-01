#!/usr/bin/env python3
"""Helper script to run the transmutation pipeline for every config in analysis/configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from transmutation_pipeline import load_case_config, run_case


def parse_args(argv: tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the transmutation pipeline for every JSON config."
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path(__file__).with_name("configs"),
        help="Directory containing case JSON files (default: analysis/configs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out"),
        help="Directory for aggregated CSV outputs (default: out).",
    )
    return parser.parse_args(argv)


def main(argv: tuple[str, ...] | None = None) -> int:
    args = parse_args(argv)
    configs_dir = args.configs
    if not configs_dir.is_absolute():
        configs_dir = (Path.cwd() / configs_dir).resolve()
    if not configs_dir.exists():
        print(f"No configs directory found at {configs_dir}", file=sys.stderr)
        return 1

    config_paths = sorted(configs_dir.glob("*.json"))
    if not config_paths:
        print(f"No JSON configs found in {configs_dir}", file=sys.stderr)
        return 1

    summary_rows = []
    band_rows = []
    print(f"Running transmutation analysis for {len(config_paths)} configs…")
    for cfg in config_paths:
        print(f"  • {cfg.name}")
        case = load_case_config(cfg)
        result = run_case(case)
        print(f"    → Results written to {result.csv_path}")
        if result.summary_row:
            summary_rows.append(result.summary_row)
        if result.band_contributions:
            band_rows.extend(result.band_contributions)

    out_dir = (Path.cwd() / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        summary_path = out_dir / "transmutation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Transmutation summary written to {summary_path}")
    if band_rows:
        band_path = out_dir / "band_contributions.csv"
        pd.DataFrame(band_rows).to_csv(band_path, index=False)
        print(f"Band contributions written to {band_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
