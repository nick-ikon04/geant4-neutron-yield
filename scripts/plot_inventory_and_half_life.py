#!/usr/bin/env python3
"""Render inventory curves and effective half-life vs current from the transmutation cases."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG_DIR = Path(__file__).resolve().parents[1] / "analysis" / "configs"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "analysis" / "transmutation_results"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "out"
    / "plots"
    / "transmutation_inventory"
)

AVOGADRO = 6.02214076e23
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

HALF_LIFE_YEARS: dict[str, float] = {
    "129I": 1.57e7,
    "135Cs": 2.3e6,
    "241Am": 432.2,
}


@dataclass
class CaseInfo:
    case_tag: str
    beam_particle: str
    target_material: str
    isotope: str
    beam_energy_MeV: float
    beam_current_A: float
    density_g_cm3: float
    volume_cm3: float
    atomic_mass_g_mol: float
    lambda_per_mA: float

    @property
    def atoms_total(self) -> float:
        return (
            self.density_g_cm3
            * self.volume_cm3
            * AVOGADRO
            / self.atomic_mass_g_mol
        )


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_lambda_per_mA(case_tag: str) -> float:
    csv_path = RESULTS_DIR / case_tag / f"{case_tag}_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results missing for {case_tag}")
    df = pd.read_csv(csv_path)
    useful = df[df["reaction_channel"] == "useful_total"]
    if useful.empty:
        raise RuntimeError(f"No useful_total row in {csv_path}")
    return float(useful["lambda_per_mA_s_inv"].iat[0])


def half_life_seconds(isotope: str) -> float:
    years = HALF_LIFE_YEARS.get(isotope)
    if years is None or years <= 0:
        raise ValueError(f"Missing half-life for {isotope}")
    return years * SECONDS_PER_YEAR


def format_years(value: float) -> str:
    if value >= 1e3:
        return f"{value/1e3:.1f}k"
    if value >= 1.0:
        return f"{value:.1f}"
    return f"{value:.2e}"


def plot_inventory(case: CaseInfo) -> None:
    inventory_dir = OUTPUT_DIR / "inventory_curves"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    lambda_decay = math.log(2.0) / half_life_seconds(case.isotope)
    current_mA = case.beam_current_A * 1e3
    lambda_trans = case.lambda_per_mA * current_mA
    lambda_eff = lambda_decay + lambda_trans

    t_years = np.logspace(-3, 7, 400)
    t_seconds = t_years * SECONDS_PER_YEAR
    base = np.exp(-lambda_decay * t_seconds)
    with_trans = np.exp(-lambda_eff * t_seconds)

    t90_decay = -math.log(0.1) / lambda_decay
    t90_eff = -math.log(0.1) / lambda_eff

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_years, base, label="Natural decay (I=0)", color="tab:gray", lw=2)
    ax.plot(
        t_years,
        with_trans,
        label=f"With {case.beam_particle} {current_mA:.1f} mA",
        color="tab:blue",
        lw=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Remaining inventory N(t)/N₀")
    ax.set_title(
        f"{case.isotope} inventory — {case.beam_particle} {case.target_material} {case.beam_energy_MeV:.0f} MeV"
    )
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.axvline(
        half_life_seconds(case.isotope) / SECONDS_PER_YEAR,
        ls=":",
        color="tab:gray",
        label="Natural T₁/₂",
    )
    ax.axvline(t90_eff / SECONDS_PER_YEAR, ls="--", color="tab:blue", label="Effective T₉₀")
    ax.legend(loc="lower left")
    ax.text(
        0.02,
        0.85,
        f"T₁/₂ natural ≈ {format_years(half_life_seconds(case.isotope)/SECONDS_PER_YEAR)} yr",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.text(
        0.02,
        0.78,
        f"T₁/₂ eff ≈ {format_years(math.log(2)/lambda_eff/SECONDS_PER_YEAR)} yr",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.text(
        0.02,
        0.71,
        f"T₉₀ natural ≈ {format_years(t90_decay/SECONDS_PER_YEAR)} yr\n"
        f"T₉₀ eff ≈ {format_years(t90_eff/SECONDS_PER_YEAR)} yr",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()
    fig_path = inventory_dir / f"inventory_{case.case_tag}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print("Saved", fig_path)


def plot_effective_half_life(lambda_map: dict[tuple[str, str], tuple[float, str]]) -> None:
    target_dir = OUTPUT_DIR / "effective_half_life"
    target_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    beam_ranges_mA = {"electron": np.linspace(0.1, 30.0, 400), "proton": np.linspace(0.01, 6.0, 400)}
    for ax, beam in zip(axes, ("electron", "proton")):
        currents = beam_ranges_mA[beam]
        for (b, isotope), (lambda_per_mA, material) in lambda_map.items():
            if b != beam:
                continue
            lambda_decay = math.log(2.0) / half_life_seconds(isotope)
            lambda_eff = lambda_decay + lambda_per_mA * currents
            half_life_years = np.log(2) / lambda_eff / SECONDS_PER_YEAR
            ax.plot(
                currents,
                half_life_years,
                label=f"{isotope} ({material})",
                lw=2,
            )
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlabel("Beam current [mA]")
        ax.set_title(f"{beam.title()} effective half-life")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.axhline(
            math.log(2) / half_life_seconds("129I") / SECONDS_PER_YEAR,
            ls=":",
            color="tab:gray",
            alpha=0.4,
        )
        ax.set_ylim(1e-3, 5e8)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("T₁/₂ₑ𝒻𝒻 [years]")

    fig.tight_layout()
    fig_path = target_dir / "effective_half_life_vs_current.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print("Saved", fig_path)


def main() -> None:
    cases: list[CaseInfo] = []
    lambda_map: dict[tuple[str, str], tuple[float, str]] = {}
    for cfg_path in sorted(CONFIG_DIR.glob("*.json")):
        cfg = load_config(cfg_path)
        case_tag = cfg["case_tag"]
        try:
            lambda_per_mA = load_lambda_per_mA(case_tag)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Skipping {case_tag}: {exc}")
            continue
        case = CaseInfo(
            case_tag=case_tag,
            beam_particle=cfg["beam_particle"],
            target_material=cfg["target_material"],
            isotope=cfg["isotope"],
            beam_energy_MeV=float(cfg["beam_energy_MeV"]),
            beam_current_A=float(cfg["beam_current_A"]),
            density_g_cm3=float(cfg["density_g_cm3"]),
            volume_cm3=float(cfg["volume_cm3"]),
            atomic_mass_g_mol=float(cfg["atomic_mass_g_mol"]),
            lambda_per_mA=lambda_per_mA,
        )
        cases.append(case)
        key = (case.beam_particle, case.isotope)
        if key not in lambda_map or lambda_per_mA > lambda_map[key][0]:
            lambda_map[key] = (lambda_per_mA, case.target_material)

    if not cases:
        raise RuntimeError("No cases available for plotting.")

    for case in cases:
        plot_inventory(case)

    plot_effective_half_life(lambda_map)


if __name__ == "__main__":
    main()
