#!/usr/bin/env python3
"""Convert TALYS total/elastic/reaction tables into mean-free-path CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

N_A = 6.02214076e23  # mol^-1
BARN_TO_CM2 = 1e-24

MATERIAL_PROPERTIES = {
    "W": {"density_g_cm3": 19.3, "atomic_mass_g_mol": 183.84},
    "natU": {"density_g_cm3": 19.1, "atomic_mass_g_mol": 238.02891},
}


def parse_tot_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            energy = float(parts[0])
            cross_section = float(parts[1])
        except ValueError:
            continue
        rows.append((energy, cross_section))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return pd.DataFrame(rows, columns=["energy_MeV", "cross_section_mb"])


def compute_number_density(material: str) -> float:
    props = MATERIAL_PROPERTIES[material]
    return N_A * props["density_g_cm3"] / props["atomic_mass_g_mol"]


def merge_cross_sections(
    total: pd.DataFrame, elastic: pd.DataFrame, reaction: pd.DataFrame
) -> pd.DataFrame:
    merged = total.merge(elastic, on="energy_MeV", suffixes=("_total", "_elastic"))
    merged = merged.merge(
        reaction.rename(columns={"cross_section_mb": "cross_section_mb_reaction"}),
        on="energy_MeV",
    )
    return merged.rename(
        columns={
            "cross_section_mb_total": "sigma_total_mb",
            "cross_section_mb_elastic": "sigma_elastic_mb",
            "cross_section_mb_reaction": "sigma_reaction_mb",
        }
    )


def convert_to_lambda(df: pd.DataFrame, material: str, projectile: str) -> pd.DataFrame:
    number_density = compute_number_density(material)
    df = df.copy()
    df["sigma_total_b"] = df["sigma_total_mb"] * 1.0e-3
    df["sigma_elastic_b"] = df["sigma_elastic_mb"] * 1.0e-3
    df["sigma_reaction_b"] = df["sigma_reaction_mb"] * 1.0e-3
    df["lambda_total_cm"] = 1.0 / (
        number_density * df["sigma_total_b"] * BARN_TO_CM2
    )
    df["lambda_reaction_cm"] = 1.0 / (
        number_density * df["sigma_reaction_b"] * BARN_TO_CM2
    )
    df.insert(0, "material", material)
    df.insert(1, "projectile", projectile)
    return df[
        [
            "material",
            "projectile",
            "energy_MeV",
            "sigma_total_b",
            "sigma_elastic_b",
            "sigma_reaction_b",
            "lambda_total_cm",
            "lambda_reaction_cm",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lambda table from TALYS totals.")
    parser.add_argument("--material", choices=MATERIAL_PROPERTIES.keys(), required=True)
    parser.add_argument("--projectile", required=True)
    parser.add_argument("--total", type=Path, required=True)
    parser.add_argument("--elastic", type=Path, required=True)
    parser.add_argument("--reaction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("talys_lambda.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total = parse_tot_file(args.total)
    elastic = parse_tot_file(args.elastic)
    reaction = parse_tot_file(args.reaction)
    merged = merge_cross_sections(total, elastic, reaction)
    lambda_table = convert_to_lambda(merged, args.material, args.projectile)
    lambda_table.to_csv(args.output, index=False)
    print(f"Lambda table written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
