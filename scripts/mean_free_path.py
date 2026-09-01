#!/usr/bin/env python3
"""Compute mean free path from TALYS cross sections for U/W targets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

N_A = 6.02214076e23  # Avogadro constant, 1/mol
BARN_TO_CM2 = 1e-24

MATERIAL_PROPERTIES = {
    "U": {"density_g_cm3": 19.1, "atomic_mass_g_mol": 238.02891},
    "W": {"density_g_cm3": 19.3, "atomic_mass_g_mol": 183.84},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate TALYS cross sections into mean-free-path estimates."
    )
    parser.add_argument(
        "--material",
        choices=MATERIAL_PROPERTIES,
        required=True,
        help="Target material whose number density is used (U or W).",
    )
    parser.add_argument(
        "--cross-section",
        dest="cross_section",
        type=Path,
        required=True,
        help="Path to the TALYS output CSV (energy + sigma columns in barns).",
    )
    parser.add_argument(
        "--energies",
        type=float,
        nargs="*",
        help="Subset of energies (MeV) to report; default is all rows in the CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to persist the derived mean-free-path table.",
    )
    return parser.parse_args()


def load_cross_sections(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "energy_MeV" not in df.columns:
        raise ValueError("Cross section file must contain an energy_MeV column.")
    sigma_cols = [col for col in df.columns if col.startswith("sigma") and col != "energy_MeV"]
    if not sigma_cols:
        raise ValueError("No sigma_* columns detected in the CSV.")
    return df


def select_sigma_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("sigma") and col != "energy_MeV"]


def compute_mean_free_path(
    df: pd.DataFrame, material: str, energy_subset: List[float] | None
) -> pd.DataFrame:
    props = MATERIAL_PROPERTIES[material]
    density = props["density_g_cm3"]
    atomic_mass = props["atomic_mass_g_mol"]
    number_density = N_A * density / atomic_mass

    if energy_subset:
        mask = df["energy_MeV"].isin(energy_subset)
        df = df[mask]
        if df.empty:
            raise ValueError("No rows match the requested energies.")

    sigma_columns = select_sigma_columns(df)
    if not sigma_columns:
        raise ValueError("No sigma columns available after filtering.")

    records: list[dict] = []
    for _, row in df.iterrows():
        energy = float(row["energy_MeV"])
        for sigma_col in sigma_columns:
            sigma_value = float(row[sigma_col])
            if sigma_value <= 0:
                continue
            mean_free_path_cm = 1.0 / (number_density * sigma_value * BARN_TO_CM2)
            records.append(
                {
                    "material": material,
                    "channel": sigma_col,
                    "energy_MeV": energy,
                    "sigma_barn": sigma_value,
                    "mean_free_path_cm": mean_free_path_cm,
                    "mean_free_path_mm": mean_free_path_cm * 10.0,
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    df = load_cross_sections(args.cross_section)
    result = compute_mean_free_path(df, args.material, args.energies)
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Mean free path saved to {args.output}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
