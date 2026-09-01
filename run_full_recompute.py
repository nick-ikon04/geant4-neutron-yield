#!/usr/bin/env python3
"""One-click orchestration of the neutron-yield + analysis + transmutation pipeline.

This script performs the following steps in sequence:
  1. (Optional) wipes simulation/analysis output directories.
  2. Runs `run_scan.py` to regenerate every Geant4 configuration currently defined.
  3. Executes `analysis/analyze_yield.py`, forwarding any beam-current overrides
     so flux columns and optimum-length tables use the desired reference currents.
  4. Launches `analysis/run_all_transmutation.py` to recompute transmutation rates
     with the (updated) TALYS/EMPIRE cross sections referenced by the JSON configs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: List[str], cwd: Path, title: str) -> None:
    print(f"\n[{title}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Geant4 scan outputs, analysis tables/plots, and transmutation results."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing results/out/analysis/transmutation_results before running.",
    )
    parser.add_argument(
        "--nprimary",
        type=int,
        default=20000,
        help="Number of primaries per Geant4 run (default: 20000).",
    )
    parser.add_argument(
        "--geant4-exec",
        type=Path,
        default=Path("build/neutron_yield"),
        help="Path to the compiled Geant4 executable (default: build/neutron_yield).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for raw Geant4 outputs (default: results).",
    )
    parser.add_argument(
        "--analysis-dir",
        dest="out_dir",
        type=Path,
        default=Path("out"),
        help="Directory for aggregated analysis outputs (default: out).",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("analysis/configs"),
        help="Directory with transmutation case JSON configs (default: analysis/configs).",
    )
    parser.add_argument(
        "--electron-current",
        dest="electron_currents",
        action="append",
        default=[],
        metavar="LABEL=AMP",
        help="Override electron reference current (e.g. e2mA=0.002). "
        "Repeat to specify multiple labels.",
    )
    parser.add_argument(
        "--proton-current",
        dest="proton_currents",
        action="append",
        default=[],
        metavar="LABEL=AMP",
        help="Override proton reference current (e.g. p20mA=0.02). "
        "Repeat to specify multiple labels.",
    )
    parser.add_argument(
        "--skip-transmutation",
        action="store_true",
        help="Skip the transmutation stage (only run Geant4 + yield analysis).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    geant4_exec = args.geant4_exec
    if not geant4_exec.is_absolute():
        geant4_exec = (repo_root / geant4_exec).resolve()
    if not geant4_exec.exists():
        raise SystemExit(f"Geant4 executable not found: {geant4_exec}")

    results_dir = (repo_root / args.results_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    configs_dir = (repo_root / args.configs_dir).resolve()
    transmutation_dir = configs_dir.parent / "transmutation_results"

    if args.clean:
        for path in (results_dir, out_dir, transmutation_dir):
            if path.exists():
                print(f"[clean] Removing {path}")
                shutil.rmtree(path)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_cmd = [
        sys.executable,
        "run_scan.py",
        "--out",
        str(results_dir),
        "--nprimary",
        str(args.nprimary),
        "--executable",
        str(geant4_exec),
    ]
    run_command(scan_cmd, repo_root, "Geant4 scan")

    analysis_cmd = [
        sys.executable,
        "analysis/analyze_yield.py",
        "--in",
        str(results_dir),
        "--out",
        str(out_dir),
    ]
    for val in args.electron_currents:
        analysis_cmd.extend(["--electron-current", val])
    for val in args.proton_currents:
        analysis_cmd.extend(["--proton-current", val])
    run_command(analysis_cmd, repo_root, "Yield analysis")

    if not args.skip_transmutation:
        if not configs_dir.exists():
            raise SystemExit(f"Transmutation configs directory not found: {configs_dir}")
        trans_cmd = [
            sys.executable,
            "analysis/run_all_transmutation.py",
            "--configs",
            str(configs_dir),
            "--out",
            str(out_dir),
        ]
        run_command(trans_cmd, repo_root, "Transmutation")

    print("\nAll stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
