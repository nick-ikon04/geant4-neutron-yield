#!/usr/bin/env python3
"""CLI helper to rescale neutron energy spectra from Geant4 runs.

The tool consumes a run metadata JSON (to read beam info and N_primary) and the
matching neutron-energy histogram CSV, then emits a table with per-primary
spectral densities together with optional rescaling to:

* specific beam currents (given in amperes);
* specific particle rates (particles per second);
* total numbers of incident primaries.

Example:
    python3 analysis/neutron_spectrum_cli.py \
        --metadata results/electron/U/E200MeV/.../run_metadata.json \
        --beam-current 0.03 --beam-current 0.01 \
        --total-particles 1e12 \
        --output out/electron_200MeV_scaled_spectrum.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

E_CHARGE_C = 1.602176634e-19  # Coulombs


@dataclass
class HistogramData:
    energy_low: np.ndarray
    energy_high: np.ndarray
    energy_center: np.ndarray
    bin_width: np.ndarray
    counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale neutron spectra by beam current or particle count."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to run_metadata.json with N_primary info.",
    )
    parser.add_argument(
        "--histogram",
        type=Path,
        help="Path to neutron energy histogram CSV "
        "(defaults to output.h1_energy from metadata).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output CSV path or '-' for stdout (default: '-').",
    )
    parser.add_argument(
        "--beam-current",
        dest="beam_currents",
        type=float,
        action="append",
        default=[],
        help="Beam current in amperes. Repeat to get multiple columns.",
    )
    parser.add_argument(
        "--particle-rate",
        dest="particle_rates",
        type=float,
        action="append",
        default=[],
        help="Incident particle rate (particles/s). Repeat to add more columns.",
    )
    parser.add_argument(
        "--total-particles",
        dest="total_particles",
        type=float,
        action="append",
        default=[],
        help="Total incident primary count to scale the spectrum to. "
        "Repeat for multiple totals.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def resolve_histogram_path(metadata: dict, override: Path | None) -> Path:
    if override:
        return override.resolve()
    output_info = metadata.get("output", {})
    hist_path = output_info.get("h1_energy")
    if hist_path:
        return Path(hist_path).resolve()
    base = output_info.get("base")
    file_type = output_info.get("file_type", "csv")
    if base:
        guess = Path(f"{base}_h1_hNeutronEnergy.{file_type}")
        return guess.resolve()
    raise FileNotFoundError(
        "Histogram path not specified. Provide --histogram or ensure metadata.output has h1_energy/base."
    )


def read_histogram(path: Path) -> HistogramData:
    text = path.read_text(encoding="utf-8").splitlines()
    axis_line = next((line for line in text if line.startswith("#axis")), None)
    if axis_line is None:
        raise ValueError(f"Histogram header missing #axis in {path}")
    parts = axis_line.split()
    if len(parts) < 5:
        raise ValueError(f"Malformed axis header in {path}: {axis_line}")
    n_bins = int(parts[2])
    energy_min = float(parts[3])
    energy_max = float(parts[4])
    if not math.isfinite(energy_min) or not math.isfinite(energy_max):
        raise ValueError(f"Invalid axis bounds in {path}")

    header_idx = next(
        (idx for idx, line in enumerate(text) if line.lower().startswith("entries")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Histogram missing entries row in {path}")

    raw = np.loadtxt(
        path,
        delimiter=",",
        comments="#",
        skiprows=header_idx + 1,
    )
    if raw.ndim == 1:
        raw = raw.reshape((-1, raw.shape[0]))
    if raw.shape[0] < 3:
        raise ValueError(f"Unexpected histogram content in {path}")

    data = raw[1:-1, :]  # drop underflow/overflow
    if data.shape[0] != n_bins:
        raise ValueError(
            f"Histogram bin count mismatch: expected {n_bins}, got {data.shape[0]} in {path}"
        )

    counts = data[:, 0].astype(float)
    edges = np.linspace(energy_min, energy_max, n_bins + 1)
    energy_low = edges[:-1]
    energy_high = edges[1:]
    centers = 0.5 * (energy_low + energy_high)
    widths = energy_high - energy_low

    return HistogramData(
        energy_low=energy_low,
        energy_high=energy_high,
        energy_center=centers,
        bin_width=widths,
        counts=counts,
    )


def sanitize_value(value: float) -> str:
    text = f"{value:g}"
    return (
        text.replace(".", "p")
        .replace("+", "")
        .replace("-", "m")
    )


def build_columns(n_totals: Sequence[float], particle_rates: Sequence[float], beam_currents: Sequence[float]) -> List[str]:
    columns = [
        "energy_low_MeV",
        "energy_high_MeV",
        "energy_center_MeV",
        "bin_width_MeV",
        "counts_per_primary",
        "dNdE_per_primary_MeV_inv",
    ]
    for total in n_totals:
        token = sanitize_value(total)
        columns.append(f"counts_for_N{token}")
        columns.append(f"dNdE_for_N{token}_per_MeV")
    for rate in particle_rates:
        token = sanitize_value(rate)
        columns.append(f"counts_rate_per_s_R{token}")
        columns.append(f"dNdE_rate_per_s_R{token}_per_MeV")
    for current in beam_currents:
        token = sanitize_value(current)
        columns.append(f"counts_rate_per_s_I{token}A")
        columns.append(f"dNdE_rate_per_s_I{token}A_per_MeV")
    return columns


def build_rows(
    hist: HistogramData,
    n_primary: float,
    totals: Sequence[float],
    particle_rates: Sequence[float],
    beam_currents: Sequence[float],
) -> Iterable[dict]:
    if n_primary <= 0:
        raise ValueError("N_primary must be positive to scale spectra.")
    counts_per_primary = hist.counts / n_primary
    with np.errstate(divide="ignore", invalid="ignore"):
        density_per_primary = np.divide(
            counts_per_primary,
            hist.bin_width,
            out=np.zeros_like(counts_per_primary),
            where=hist.bin_width > 0,
        )

    for idx in range(hist.counts.size):
        row = {
            "energy_low_MeV": hist.energy_low[idx],
            "energy_high_MeV": hist.energy_high[idx],
            "energy_center_MeV": hist.energy_center[idx],
            "bin_width_MeV": hist.bin_width[idx],
            "counts_per_primary": counts_per_primary[idx],
            "dNdE_per_primary_MeV_inv": density_per_primary[idx],
        }

        for total in totals:
            scaled_counts = counts_per_primary[idx] * total
            scaled_density = density_per_primary[idx] * total
            token = sanitize_value(total)
            row[f"counts_for_N{token}"] = scaled_counts
            row[f"dNdE_for_N{token}_per_MeV"] = scaled_density

        for rate in particle_rates:
            scaled_counts = counts_per_primary[idx] * rate
            scaled_density = density_per_primary[idx] * rate
            token = sanitize_value(rate)
            row[f"counts_rate_per_s_R{token}"] = scaled_counts
            row[f"dNdE_rate_per_s_R{token}_per_MeV"] = scaled_density

        for current in beam_currents:
            rate = current / E_CHARGE_C
            scaled_counts = counts_per_primary[idx] * rate
            scaled_density = density_per_primary[idx] * rate
            token = sanitize_value(current)
            row[f"counts_rate_per_s_I{token}A"] = scaled_counts
            row[f"dNdE_rate_per_s_I{token}A_per_MeV"] = scaled_density

        yield row


def write_csv(rows: Iterable[dict], columns: Sequence[str], output: str) -> None:
    if output == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    hist_path = resolve_histogram_path(metadata, args.histogram)
    hist = read_histogram(hist_path)
    n_primary = float(metadata.get("N_primary", 0.0))

    rows = list(
        build_rows(
            hist,
            n_primary,
            totals=args.total_particles,
            particle_rates=args.particle_rates,
            beam_currents=args.beam_currents,
        )
    )
    columns = build_columns(args.total_particles, args.particle_rates, args.beam_currents)
    write_csv(rows, columns, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

