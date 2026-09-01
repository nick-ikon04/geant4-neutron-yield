#!/usr/bin/env python3
"""Convert TALYS sigma output to proton mean-free-path tables for W/natU."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

N_A = 6.02214076e23  # mol^-1
BARN_TO_CM2 = 1e-24

MATERIAL_PROPERTIES = {
    "W": {"density_g_cm3": 19.3, "atomic_mass_g_mol": 183.84},
    "natU": {"density_g_cm3": 19.1, "atomic_mass_g_mol": 238.02891},
}

KEYWORDS = {
    "energy": ["energy", "energy_MeV"],
    "total": ["total", "sigma_total", "sigma_tot"],
    "elastic": ["elastic", "sigma_elastic"],
    "reaction": ["reaction", "sigma_reac", "sigma_reaction"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TALYS sigma columns and compute proton mean-free-paths."
    )
    parser.add_argument(
        "--material",
        choices=list(MATERIAL_PROPERTIES.keys()),
        required=True,
        help="Material label (W or natU).",
    )
    parser.add_argument(
        "--sigma-file",
        type=Path,
        default=Path("sigma"),
        help="TALYS sigma output file (default: ./sigma).",
    )
    parser.add_argument(
        "--projectile",
        type=str,
        default="p",
        help="Projectile label to store in CSV (default: p).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("talys_lambda.csv"),
        help="Output CSV path (default: talys_lambda.csv).",
    )
    return parser.parse_args()


def read_sigma_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sigma file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    if df.empty:
        raise ValueError("Empty sigma file")
    df.columns = [col.strip().lower() for col in df.columns]
    required = [KEYWORDS[k][0] for k in ("energy", "total", "elastic", "reaction")]
    # Ensure energy column exists by matching keywords
    def match(keyword_list: Sequence[str]) -> str | None:
        for col in df.columns:
            if any(kw in col for kw in keyword_list):
                return col
        return None

    energy_col = match(KEYWORDS["energy"])
    total_col = match(KEYWORDS["total"])
    elastic_col = match(KEYWORDS["elastic"])
    reaction_col = match(KEYWORDS["reaction"])

    if energy_col is None or total_col is None or reaction_col is None:
        raise ValueError("Sigma file must expose energy/total/reaction columns")

    df = df[[energy_col, total_col, elastic_col, reaction_col]].rename(
        columns={
            energy_col: "energy_MeV",
            total_col: "sigma_total_b",
            elastic_col: "sigma_elastic_b",
            reaction_col: "sigma_reaction_b",
        }
    )
    return df.astype(float)


def compute_number_density(material: str) -> float:
    props = MATERIAL_PROPERTIES[material]
    return N_A * props["density_g_cm3"] / props["atomic_mass_g_mol"]


def build_lambda_table(df: pd.DataFrame, material: str, projectile: str) -> pd.DataFrame:
    number_density = compute_number_density(material)
    df = df.copy()
    df["lambda_total_cm"] = 1.0 / (number_density * df["sigma_total_b"] * BARN_TO_CM2)
    df["lambda_reaction_cm"] = 1.0 / (number_density * df["sigma_reaction_b"] * BARN_TO_CM2)
    df.insert(0, "projectile", projectile)
    df.insert(0, "material", material)
    return df


def main() -> int:
    args = parse_args()
    sigma_df = read_sigma_table(args.sigma_file)
    result = build_lambda_table(sigma_df, args.material, args.projectile)
    result.to_csv(args.output, index=False)
    print(f"Written {len(result)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
