#!/usr/bin/env python3
"""Aggregate analysis for neutron-yield simulation campaigns."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

E_CHARGE_C = 1.602176634e-19  # Coulombs
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("out")

DEFAULT_ELECTRON_CURRENTS_A = {
    "e_low": 0.010,  # 10 mA
    "e_high": 0.030,  # 30 mA
}
DEFAULT_PROTON_CURRENTS_A = {
    "p_nominal": 0.001,  # 1 mA
    "p_high": 0.002,  # 2 mA
}
ELECTRON_CURRENTS_A = dict(DEFAULT_ELECTRON_CURRENTS_A)
PROTON_CURRENTS_A = dict(DEFAULT_PROTON_CURRENTS_A)
ELECTRON_PRIORITY_LABEL = "e_high"
PROTON_PRIORITY_LABEL = "p_high"

PLOT_FIGSIZE = (7, 4.5)


@dataclass
class RunRecord:
    metadata_path: Path
    beam_type: str
    beam_energy_MeV: float
    target_material: str
    half_length_mm: float
    radius_mm: float
    scoring_thickness_mm: float
    n_primary: float
    n_neutrons: float
    yield_per_primary: float
    sigma_yield: float
    output_base: Path
    file_type: str
    hist_energy: Optional[Path] = None
    hist_costheta: Optional[Path] = None
    ntuple_path: Optional[Path] = None
    extra_metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def l_label(self) -> str:
        return format_value(self.half_length_mm)

    @property
    def r_label(self) -> str:
        return format_value(self.radius_mm)


def format_value(value: float) -> str:
    formatted = f"{value:g}"
    return formatted.replace(".", "p")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def load_metadata_file(metadata_path: Path) -> Optional[RunRecord]:
    with metadata_path.open(encoding="utf-8") as fh:
        metadata = json.load(fh)

    beam_type = metadata.get("beam_type")
    beam_energy = float(metadata.get("beam_energy_MeV", 0.0))
    material = metadata.get("target_material")
    if not beam_type or not material:
        raise ValueError("Metadata missing beam_type or target_material")
    half_length = float(metadata.get("target_half_length_mm", 0.0))
    radius = float(metadata.get("target_radius_mm", 0.0))
    scoring_thickness = float(metadata.get("scoring_shell_thickness_mm", 0.0))
    n_primary = float(metadata.get("N_primary", 0.0))
    n_neutrons = float(metadata.get("N_neutrons", 0.0))
    yield_per_primary = float(metadata.get("yield_per_primary", 0.0))
    sigma_yield = float(metadata.get("sigma_yield", 0.0))

    if yield_per_primary <= 0.0 and n_primary > 0.0:
        yield_per_primary = n_neutrons / n_primary if n_neutrons > 0.0 else 0.0
    if sigma_yield <= 0.0 and n_primary > 0.0:
        sigma_yield = (
            math.sqrt(n_neutrons) / n_primary if n_neutrons > 0.0 else 0.0
        )

    output_info = metadata.get("output", {})
    base_str = output_info.get("base")
    if not base_str:
        stem = metadata_path.stem
        if stem.endswith("_metadata"):
            base_str = str(metadata_path.with_name(stem[:-9]))  # remove suffix
        else:
            base_str = str(metadata_path.with_suffix(""))
    file_type = output_info.get("file_type", "csv")

    base_path = resolve_path(base_str)
    energy_hist = output_info.get(
        "h1_energy", f"{base_str}_h1_hNeutronEnergy.{file_type}"
    )
    cos_hist = output_info.get("h1_cosTheta")
    ntuple = output_info.get("ntuple")

    record = RunRecord(
        metadata_path=metadata_path,
        beam_type=str(beam_type),
        beam_energy_MeV=beam_energy,
        target_material=str(material),
        half_length_mm=half_length,
        radius_mm=radius,
        scoring_thickness_mm=scoring_thickness,
        n_primary=n_primary,
        n_neutrons=n_neutrons,
        yield_per_primary=yield_per_primary,
        sigma_yield=sigma_yield,
        output_base=base_path,
        file_type=file_type,
        hist_energy=resolve_path(energy_hist),
        hist_costheta=resolve_path(cos_hist) if cos_hist else None,
        ntuple_path=resolve_path(ntuple) if ntuple else None,
        extra_metadata=metadata,
    )
    return record


def find_metadata_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.rglob("*_metadata.json"))


def compute_fluxes(record: RunRecord) -> Dict[str, Tuple[float, float]]:
    fluxes: Dict[str, Tuple[float, float]] = {}
    if record.yield_per_primary <= 0.0 or record.n_primary <= 0.0:
        return fluxes

    if record.beam_type == "electron":
        for label, current in ELECTRON_CURRENTS_A.items():
            value = record.yield_per_primary * current / E_CHARGE_C
            sigma = record.sigma_yield * current / E_CHARGE_C
            fluxes[label] = (value, sigma)
    elif record.beam_type == "proton":
        for label, current in PROTON_CURRENTS_A.items():
            value = record.yield_per_primary * current / E_CHARGE_C
            sigma = record.sigma_yield * current / E_CHARGE_C
            fluxes[label] = (value, sigma)
    return fluxes


def load_ntuple(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "energy_MeV", "cosTheta", "x_exit_mm", "y_exit_mm", "z_exit_mm"
        ])
    df = pd.read_csv(path, comment="#")
    return df


def make_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def spectrum_plot(record: RunRecord, df: pd.DataFrame, plots_dir: Path) -> None:
    if df.empty:
        return
    energies = df.get("energy_MeV")
    if energies is None or energies.empty:
        return

    energies = energies.to_numpy(dtype=float)
    max_energy = max(energies.max(), 1.0)
    bins = min(200, max(60, int(np.sqrt(energies.size)) * 4))
    counts, edges = np.histogram(energies, bins=bins, range=(0.0, max_energy))
    widths = np.diff(edges)
    density = counts / (record.n_primary * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])

    make_directory(plots_dir)
    filename = (
        f"spectrum_Eneutron_{record.beam_type}_{record.target_material}_"
        f"E{format_value(record.beam_energy_MeV)}_"
        f"L{record.l_label}_R{record.r_label}.png"
    )
    out_path = plots_dir / filename

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.step(centers, np.where(density > 0, density, np.nan), where="mid")
    plt.yscale("log")
    plt.xlabel("Neutron energy (MeV)")
    plt.ylabel("dN/dE per primary (1/MeV)")
    plt.title(
        f"Neutron spectrum {record.beam_type} {record.beam_energy_MeV:g} MeV,"
        f" {record.target_material}, L={record.half_length_mm:g} mm,"
        f" R={record.radius_mm:g} mm"
    )
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def angular_plot(record: RunRecord, df: pd.DataFrame, plots_dir: Path) -> None:
    if df.empty:
        return
    cos_values = df.get("cosTheta")
    if cos_values is None or cos_values.empty:
        return

    cos_values = cos_values.to_numpy(dtype=float)
    bins = 80
    counts, edges = np.histogram(cos_values, bins=bins, range=(-1.0, 1.0))
    widths = np.diff(edges)
    density = counts / (record.n_primary * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])

    make_directory(plots_dir)
    filename = (
        f"angular_cosTheta_{record.beam_type}_{record.target_material}_"
        f"E{format_value(record.beam_energy_MeV)}_"
        f"L{record.l_label}_R{record.r_label}.png"
    )
    out_path = plots_dir / filename

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.step(centers, density, where="mid")
    plt.xlabel("cos(θ)")
    plt.ylabel("dN/dcosθ per primary")
    plt.title(
        f"Angular distribution {record.beam_type} {record.beam_energy_MeV:g} MeV,"
        f" {record.target_material}, L={record.half_length_mm:g} mm,"
        f" R={record.radius_mm:g} mm"
    )
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def yield_vs_thickness_plot(summary: pd.DataFrame, plots_dir: Path) -> None:
    make_directory(plots_dir)
    grouped = summary.groupby(
        ["beam_type", "target_material", "beam_energy_MeV", "scoring_shell_thickness_mm"]
    )
    for (beam, material, energy, scoring), group in grouped:
        plt.figure(figsize=PLOT_FIGSIZE)
        for radius, r_group in group.groupby("R_mm"):
            r_group = r_group.sort_values("L_mm")
            plt.errorbar(
                r_group["L_mm"],
                r_group["Y_n_per_primary"],
                yerr=r_group["sigma_Y"],
                marker="o",
                linestyle="-",
                label=f"R={radius:g} mm",
            )
        plt.xlabel("Target half-length L (mm)")
        plt.ylabel("Neutron yield per primary")
        plt.title(
            f"Yield vs thickness, {beam}, {material}, E={energy:g} MeV, S={scoring:g} mm"
        )
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = (
            f"yield_vs_thickness_{beam}_{material}_E{format_value(energy)}"
            f"_S{format_value(scoring)}.png"
        )
        plt.savefig(plots_dir / filename)
        plt.close()


def yield_vs_thickness_r20_variants(summary: pd.DataFrame, plots_dir: Path) -> None:
    make_directory(plots_dir)
    if summary.empty or "R_mm" not in summary.columns:
        return
    subset = summary[np.isclose(summary["R_mm"], 20.0)]
    if subset.empty:
        return
    grouped = subset.groupby(
        ["beam_type", "target_material", "beam_energy_MeV", "scoring_shell_thickness_mm"]
    )

    for (beam, material, energy, scoring), group in grouped:
        group = group.sort_values("L_mm")
        if group.empty:
            continue

        def plot_variant(data: pd.DataFrame, suffix: str, description: str) -> None:
            if data.empty:
                return
            plt.figure(figsize=PLOT_FIGSIZE)
            plt.errorbar(
                data["L_mm"],
                data["Y_n_per_primary"],
                yerr=data["sigma_Y"],
                marker="o",
                linestyle="-",
            )
            plt.xlabel("Target half-length L (mm)")
            plt.ylabel("Neutron yield per primary")
            plt.title(
                f"{description}, {beam}, {material}, E={energy:g} MeV, S={scoring:g} mm, R=20 mm"
            )
            plt.grid(True, ls="--", alpha=0.3)
            plt.tight_layout()
            filename = (
                f"yield_vs_thickness_R20_{beam}_{material}_E{format_value(energy)}"
                f"_S{format_value(scoring)}_{suffix}.png"
            )
            plt.savefig(plots_dir / filename)
            plt.close()

        plot_variant(group, "with_50_100mm", "Yield vs thickness (with 50/100 mm)")

        trimmed = group[
            (~np.isclose(group["L_mm"], 50.0)) & (~np.isclose(group["L_mm"], 100.0))
        ]
        plot_variant(trimmed, "without_50_100mm", "Yield vs thickness (without 50/100 mm)")


def yield_vs_energy_plot(summary: pd.DataFrame, plots_dir: Path) -> None:
    make_directory(plots_dir)
    grouped = summary.groupby(
        ["beam_type", "target_material", "L_mm", "scoring_shell_thickness_mm"]
    )
    for (beam, material, length, scoring), group in grouped:
        plt.figure(figsize=PLOT_FIGSIZE)
        for radius, r_group in group.groupby("R_mm"):
            r_group = r_group.sort_values("beam_energy_MeV")
            plt.errorbar(
                r_group["beam_energy_MeV"],
                r_group["Y_n_per_primary"],
                yerr=r_group["sigma_Y"],
                marker="o",
                linestyle="-",
                label=f"R={radius:g} mm",
            )
        plt.xlabel("Beam energy (MeV)")
        plt.ylabel("Neutron yield per primary")
        plt.title(
            f"Yield vs energy, {beam}, {material}, L={length:g} mm, S={scoring:g} mm"
        )
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = (
            f"yield_vs_energy_{beam}_{material}_L{format_value(length)}"
            f"_S{format_value(scoring)}.png"
        )
        plt.savefig(plots_dir / filename)
        plt.close()


def build_summary_dataframe(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        fluxes = compute_fluxes(rec)
        row = {
            "metadata_path": rec.metadata_path.as_posix(),
            "beam_type": rec.beam_type,
            "beam_energy_MeV": rec.beam_energy_MeV,
            "target_material": rec.target_material,
            "L_mm": rec.half_length_mm,
            "R_mm": rec.radius_mm,
            "scoring_shell_thickness_mm": rec.scoring_thickness_mm,
            "N_primary": rec.n_primary,
            "N_neutrons": rec.n_neutrons,
            "Y_n_per_primary": rec.yield_per_primary,
            "sigma_Y": rec.sigma_yield,
            "output_base": rec.output_base.as_posix(),
            "file_type": rec.file_type,
            "h1_energy": rec.hist_energy.as_posix() if rec.hist_energy else "",
            "h1_cosTheta": rec.hist_costheta.as_posix() if rec.hist_costheta else "",
            "ntuple": rec.ntuple_path.as_posix() if rec.ntuple_path else "",
        }
        for key, value in fluxes.items():
            row[f"neutrons_per_sec_{key}"] = value[0]
            row[f"sigma_{key}"] = value[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def _read_optional_csv(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def find_top3_by_flux(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (beam, material), group in df.groupby(["beam_type", "target_material"]):
        label = ELECTRON_PRIORITY_LABEL if beam == "electron" else PROTON_PRIORITY_LABEL
        if not label:
            continue
        metric = f"neutrons_per_sec_{label}"
        if metric not in group.columns:
            continue
        top = group.sort_values(metric, ascending=False).head(3)
        top = top.assign(rank=range(1, len(top) + 1))
        records.append(top)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def quadratic_optimum(x_vals: np.ndarray, y_vals: np.ndarray) -> Optional[Tuple[float, float]]:
    if x_vals.size < 3:
        return None
    coeffs = np.polyfit(x_vals, y_vals, 2)
    a, b, c = coeffs
    if a >= 0:
        return None
    x_opt = -b / (2 * a)
    if x_opt < x_vals.min() or x_opt > x_vals.max():
        return None
    y_opt = a * x_opt ** 2 + b * x_opt + c
    return x_opt, y_opt


def determine_optimal_sizes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (beam, material, energy, radius, scoring), group in df.groupby(
        ["beam_type", "target_material", "beam_energy_MeV", "R_mm", "scoring_shell_thickness_mm"]
    ):
        group = group.sort_values("L_mm")
        y_values = group["Y_n_per_primary"].to_numpy()
        x_values = group["L_mm"].to_numpy()
        max_idx = int(np.argmax(y_values))
        L_best = x_values[max_idx]
        Y_best = y_values[max_idx]
        sigma_best = group.iloc[max_idx]["sigma_Y"]

        if 0 < max_idx < len(x_values) - 1:
            candidate = quadratic_optimum(
                x_values[max_idx - 1 : max_idx + 2],
                y_values[max_idx - 1 : max_idx + 2],
            )
            if candidate is not None:
                L_candidate, Y_candidate = candidate
                L_best = float(L_candidate)
                Y_best = float(Y_candidate)

        record = {
            "beam_type": beam,
            "target_material": material,
            "beam_energy_MeV": energy,
            "R_mm": radius,
            "scoring_shell_thickness_mm": scoring,
            "L_opt_mm": L_best,
            "Y_opt": Y_best,
            "sigma_Y_opt": sigma_best,
        }
        fluxes = compute_fluxes(
            RunRecord(
                metadata_path=Path(),
                beam_type=beam,
                beam_energy_MeV=energy,
                target_material=material,
                half_length_mm=L_best,
                radius_mm=radius,
                scoring_thickness_mm=scoring,
                n_primary=1.0,
                n_neutrons=Y_best,
                yield_per_primary=Y_best,
                sigma_yield=sigma_best,
                output_base=Path(),
                file_type="",
            )
        )
        for key, value in fluxes.items():
            record[f"n_per_sec_{key}"] = value[0]
            record[f"sigma_{key}"] = value[1]
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    optimal_rows = []
    for (beam, material, energy), group in pd.DataFrame(rows).groupby(
        ["beam_type", "target_material", "beam_energy_MeV"]
    ):
        label = ELECTRON_PRIORITY_LABEL if beam == "electron" else PROTON_PRIORITY_LABEL
        metric = f"n_per_sec_{label}" if label else None
        if metric and metric in group.columns:
            group_sorted = group.sort_values(metric, ascending=False)
        else:
            group_sorted = group
        optimal_rows.append(group_sorted.iloc[0])
    return pd.DataFrame(optimal_rows)


RATE_BAND_ZONES = [
    ("0–0.4 eV", 0.0, 4e-7),
    ("0.4–1 eV", 4e-7, 1e-6),
    ("1–10 eV", 1e-6, 1e-5),
    ("10–100 eV", 1e-5, 1e-4),
    ("0.1–1 keV", 1e-4, 1e-3),
    ("1–10 keV", 1e-3, 1e-2),
    ("10–100 keV", 1e-2, 0.1),
    ("0.1–1 MeV", 0.1, 1.0),
    ("1–20 MeV", 1.0, 20.0),
    (">20 MeV", 20.0, float("inf")),
]


def _geometry_key(
    beam: str,
    material: str,
    energy: float,
    length: float,
    radius: float,
    scoring: float,
) -> str:
    def _fmt(value: float) -> str:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.4f}"

    return "|".join(
        [
            str(beam),
            str(material),
            f"{float(energy):.4f}",
            _fmt(length),
            _fmt(radius),
            _fmt(scoring),
        ]
    )


def time_to90_vs_current_plot(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    if trans_summary.empty or "lambda_per_mA_s_inv" not in trans_summary.columns:
        return
    data = trans_summary.copy()
    data = data[data["lambda_per_mA_s_inv"] > 0]
    if data.empty:
        return

    groups = list(data.groupby("beam_particle"))
    if not groups:
        return
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6 * len(groups), 4.5),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    ln10 = math.log(10.0)
    for ax, (beam, group) in zip(axes_flat, groups):
        max_current = max(
            group.get("case_current_mA", pd.Series([0])).max(),
            group.get("accel_ref_mA", pd.Series([0])).max(),
        )
        min_current = group.get("case_current_mA", pd.Series([0])).replace(0, np.nan).min()
        if not np.isfinite(max_current) or max_current <= 0:
            continue
        start = max(0.05, float(min_current) * 0.2) if np.isfinite(min_current) else 0.05
        currents = np.linspace(start, max_current * 1.2, 200)
        for _, row in group.iterrows():
            lam_per_mA = row["lambda_per_mA_s_inv"]
            if lam_per_mA <= 0:
                continue
            lam_eff = lam_per_mA * currents
            t90_days = (ln10 / lam_eff) / 86400.0
            label = (
                f"{row['target_material']} "
                f"L{format_value(row['geometry_L_mm'])} R{format_value(row['geometry_R_mm'])} "
                f"({row['isotope']})"
            )
            ax.plot(currents, t90_days, label=label)
        ax.set_xlabel("Beam current (mA)")
        ax.set_ylabel("t90 (days)")
        ax.set_yscale("log")
        ax.set_title(f"{beam} main configurations", fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes_flat[len(groups) :]:
        ax.axis("off")
    make_directory(plots_dir)
    fig.tight_layout()
    fig.savefig(plots_dir / "TimeTo90_vs_Current.png")
    plt.close(fig)


def lambda_eff_vs_current_plot(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    if trans_summary.empty or "lambda_per_mA_s_inv" not in trans_summary.columns:
        return
    data = trans_summary.copy()
    data = data[data["lambda_per_mA_s_inv"] > 0]
    if data.empty:
        return

    groups = list(data.groupby("beam_particle"))
    if not groups:
        return
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6 * len(groups), 4.5),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax, (beam, group) in zip(axes_flat, groups):
        max_current = max(
            group.get("case_current_mA", pd.Series([0])).max(),
            group.get("accel_ref_mA", pd.Series([0])).max(),
        )
        if not np.isfinite(max_current) or max_current <= 0:
            continue
        start = max(max_current * 0.2, 0.05)
        currents = np.linspace(start, max_current * 1.2, 200)
        for _, row in group.iterrows():
            lam_per_mA = row["lambda_per_mA_s_inv"]
            if lam_per_mA <= 0:
                continue
            values = lam_per_mA * currents
            label = (
                f"{row['target_material']} "
                f"L{format_value(row['geometry_L_mm'])} R{format_value(row['geometry_R_mm'])} "
                f"({row['isotope']})"
            )
            ax.plot(currents, values, label=label)
        ax.set_xlabel("Beam current (mA)")
        ax.set_ylabel("λ_eff (1/s)")
        ax.set_title(f"{beam} main configurations", fontsize=11)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes_flat[len(groups) :]:
        ax.axis("off")
    make_directory(plots_dir)
    fig.tight_layout()
    fig.savefig(plots_dir / "LambdaEff_vs_Current.png")
    plt.close(fig)


def _atoms_per_coulomb_label(row: pd.Series) -> str:
    parts = [
        f"{row['beam_particle']}",
        f"E{row['beam_energy_MeV']:g}MeV",
        row["target_material"],
        f"L{format_value(row['geometry_L_mm'])}",
        f"R{format_value(row['geometry_R_mm'])}",
    ]
    if "geometry_S_mm" in row and not pd.isna(row["geometry_S_mm"]):
        parts.append(f"S{format_value(row['geometry_S_mm'])}")
    return " ".join(parts)


def _atoms_per_coulomb_plot(
    df: pd.DataFrame, values: np.ndarray, title: str, filename: str, plots_dir: Path
) -> None:
    valid = df.copy()
    valid["atoms_per_coulomb"] = values
    valid = valid.replace({np.inf: np.nan, -np.inf: np.nan}).dropna(subset=["atoms_per_coulomb"])
    if valid.empty:
        return
    valid = valid.sort_values("atoms_per_coulomb", ascending=False)
    labels = valid.apply(_atoms_per_coulomb_label, axis=1)
    y_pos = np.arange(len(valid))
    colors = valid["beam_particle"].map({"electron": "tab:blue", "proton": "tab:orange"}).fillna("tab:gray")

    make_directory(plots_dir)
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(valid) * 0.25)))
    ax.barh(y_pos, valid["atoms_per_coulomb"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Atoms destroyed per Coulomb")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, axis="x", ls="--", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=4, label="electron"),
        plt.Line2D([0], [0], color="tab:orange", lw=4, label="proton"),
    ]
    ax.legend(handles=handles, title="Beam")
    fig.tight_layout()
    fig.savefig(plots_dir / filename)
    plt.close(fig)


def atoms_per_coulomb_plot(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    if trans_summary.empty:
        return
    required = {
        "beam_particle",
        "beam_energy_MeV",
        "target_material",
        "geometry_L_mm",
        "geometry_R_mm",
        "geometry_S_mm",
        "Atoms_per_C",
    }
    if not required.issubset(trans_summary.columns):
        return
    data = trans_summary.copy()
    data["atoms_per_coulomb"] = data["Atoms_per_C"]
    data = data.replace({np.inf: np.nan, -np.inf: np.nan}).dropna(subset=["atoms_per_coulomb"])
    if data.empty:
        return

    _atoms_per_coulomb_plot(
        data,
        data["atoms_per_coulomb"].to_numpy(),
        "Atoms per Coulomb (all geometries)",
        "AtomsPerCoulomb_vs_Geometry.png",
        plots_dir,
    )


def _assign_rate_zone(mid_energy: float) -> str:
    for label, e_min, e_max in RATE_BAND_ZONES:
        if e_min <= mid_energy < e_max:
            return label
    return RATE_BAND_ZONES[-1][0]


def rate_bands_stacked_plot(band_df: pd.DataFrame, plots_dir: Path) -> None:
    if band_df.empty:
        return
    required = {
        "case_tag",
        "beam_particle",
        "target_material",
        "isotope",
        "energy_min_MeV",
        "energy_max_MeV",
        "lambda_per_mA_band",
    }
    if not required.issubset(band_df.columns):
        return
    df = band_df.copy()
    df = df[df["lambda_per_mA_band"] > 0]
    if df.empty:
        return

    df["mid_energy"] = 0.5 * (df["energy_min_MeV"] + df["energy_max_MeV"])
    df["zone_label"] = df["mid_energy"].apply(_assign_rate_zone)

    band_dir = plots_dir / "rate_bands"
    make_directory(band_dir)

    for beam, beam_df in df.groupby("beam_particle"):
        pivot = (
            beam_df.pivot_table(
                index="case_tag",
                columns="zone_label",
                values="lambda_per_mA_band",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(columns=[label for label, *_ in RATE_BAND_ZONES], fill_value=0.0)
            .sort_index()
        )
        if pivot.empty:
            continue
        totals = pivot.sum(axis=1)
        pivot = pivot.loc[totals.sort_values(ascending=False).index]

        case_meta = (
            beam_df.groupby("case_tag")[["target_material", "isotope"]]
            .first()
            .reindex(pivot.index)
        )
        labels = [
            f"{row['target_material']}\n{row['isotope']}"
            for _, row in case_meta.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(pivot) * 1.2), 5))
        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))
        for zone_label, _, _ in RATE_BAND_ZONES:
            values = pivot[zone_label].to_numpy()
            if not np.any(values):
                continue
            ax.bar(x, values, bottom=bottom, label=zone_label)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("λ_eff per mA (1/s)")
        ax.set_title(f"Energy-band contributions — {beam}")
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(band_dir / f"RateBands_Stacked_{beam}.png")
        plt.close(fig)


def _record_key_tuple(
    beam: str,
    material: str,
    energy: float,
    length: float,
    radius: float,
    scoring: float,
) -> Tuple[str, str, float, float, float, float]:
    return (
        beam,
        material,
        round(float(energy), 6),
        round(float(length), 6),
        round(float(radius), 6),
        round(float(scoring), 6),
    )


def _compute_spectrum_density(record: RunRecord, df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if df.empty or record.n_primary <= 0:
        return None
    energies = df.get("energy_MeV")
    if energies is None or energies.empty:
        return None
    values = energies.to_numpy(dtype=float)
    max_energy = max(values.max(), 1.0)
    bins = min(200, max(60, int(np.sqrt(values.size)) * 4))
    counts, edges = np.histogram(values, bins=bins, range=(0.0, max_energy))
    widths = np.diff(edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = counts / (record.n_primary * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def compare_material_spectra(
    records: Sequence[RunRecord], summary: pd.DataFrame, plots_dir: Path
) -> None:
    if not records or summary.empty:
        return
    subset = summary[summary["target_material"].isin(["W", "U"])]
    if subset.empty:
        return
    record_map = {
        _record_key_tuple(
            rec.beam_type,
            rec.target_material,
            rec.beam_energy_MeV,
            rec.half_length_mm,
            rec.radius_mm,
            rec.scoring_thickness_mm,
        ): rec
        for rec in records
    }
    for beam, beam_group in subset.groupby("beam_type"):
        for energy, energy_group in beam_group.groupby("beam_energy_MeV"):
            spectra: Dict[str, Tuple[np.ndarray, np.ndarray, pd.Series]] = {}
            for material in ("W", "U"):
                mat_group = energy_group[energy_group["target_material"] == material]
                if mat_group.empty:
                    continue
                best_idx = mat_group["Y_n_per_primary"].idxmax()
                best_row = mat_group.loc[best_idx]
                key = _record_key_tuple(
                    beam,
                    material,
                    best_row["beam_energy_MeV"],
                    best_row["L_mm"],
                    best_row["R_mm"],
                    best_row["scoring_shell_thickness_mm"],
                )
                record = record_map.get(key)
                if record is None or not record.ntuple_path:
                    continue
                df_ntuple = load_ntuple(record.ntuple_path)
                spectrum = _compute_spectrum_density(record, df_ntuple)
                if spectrum is None:
                    continue
                spectra[material] = (*spectrum, best_row)
            if len(spectra) < 2:
                continue
            make_directory(plots_dir)
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            for material, (centers, density, row) in spectra.items():
                ax.step(
                    centers,
                    np.where(density > 0, density, np.nan),
                    where="mid",
                    label=(
                        f"{material}: L{format_value(row['L_mm'])} "
                        f"R{format_value(row['R_mm'])} S{format_value(row['scoring_shell_thickness_mm'])}"
                    ),
                )
            ax.set_xlabel("Neutron energy (MeV)")
            ax.set_ylabel("dN/dE per primary (1/MeV)")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_title(f"Neutron spectra W vs U — {beam}, E={energy:g} MeV")
            ax.legend()
            fig.tight_layout()
            filename = f"Spectra_W_vs_U_{beam}_E{format_value(energy)}.png"
            fig.savefig(plots_dir / filename)
            plt.close(fig)


GEOMETRY_GROUP_COLUMNS = [
    "beam_particle",
    "target_material",
    "geometry_R_mm",
    "geometry_L_mm",
    "geometry_S_mm",
]


def _iter_geometry_energy_groups(
    trans_summary: pd.DataFrame, extra_required: Sequence[str]
):
    required = {
        "beam_energy_MeV",
        "isotope",
        *GEOMETRY_GROUP_COLUMNS,
        *extra_required,
    }
    columns = set(trans_summary.columns)
    if not required.issubset(columns):
        return
    data = trans_summary.dropna(subset=["geometry_R_mm", "geometry_L_mm"])
    for key, group in data.groupby(GEOMETRY_GROUP_COLUMNS):
        group = group.sort_values("beam_energy_MeV")
        if group["beam_energy_MeV"].nunique() < 2:
            continue
        yield key, group


def _geometry_descriptor(material: str, radius: float, length: float, scoring: float) -> str:
    return (
        f"{material} R{format_value(radius)} mm "
        f"L{format_value(length)} mm S{format_value(scoring)} mm"
    )


def lambda_eff_vs_energy_plots(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    extra = ["lambda_per_mA_s_inv"]
    energy_dir = plots_dir / "energy_scans"
    for key, group in _iter_geometry_energy_groups(trans_summary, extra):
        beam, material, radius, length, scoring = key
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for isotope, iso_group in group.groupby("isotope"):
            ax.plot(
                iso_group["beam_energy_MeV"],
                iso_group["lambda_per_mA_s_inv"],
                marker="o",
                label=isotope,
            )
        ax.set_xlabel("Beam energy (MeV)")
        ax.set_ylabel("λ_eff per mA (1/s)")
        ax.set_title(f"λ_eff vs Energy — {beam} {_geometry_descriptor(material, radius, length, scoring)}")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend()
        make_directory(energy_dir)
        filename = f"LambdaEff_vs_Energy__{material}_R{format_value(radius)}_L{format_value(length)}.png"
        fig.tight_layout()
        fig.savefig(energy_dir / filename)
        plt.close(fig)


def t90_vs_energy_plots(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    extra = ["lambda_per_mA_s_inv", "accel_ref_mA"]
    energy_dir = plots_dir / "energy_scans"
    ln10 = math.log(10.0)
    for key, group in _iter_geometry_energy_groups(trans_summary, extra):
        beam, material, radius, length, scoring = key
        accel_ref = group["accel_ref_mA"].iloc[0]
        if accel_ref <= 0:
            continue
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for isotope, iso_group in group.groupby("isotope"):
            lambda_per_mA = iso_group["lambda_per_mA_s_inv"]
            lambda_accel = lambda_per_mA * accel_ref
            t90_days = (ln10 / lambda_accel) / 86400.0
            ax.plot(
                iso_group["beam_energy_MeV"],
                t90_days,
                marker="o",
                label=isotope,
            )
        for level in (30, 90, 180):
            ax.axhline(level, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Beam energy (MeV)")
        ax.set_ylabel(f"t90 at I_ref={accel_ref:g} mA (days)")
        ax.set_yscale("log")
        ax.set_title(f"t90 vs Energy — {beam} {_geometry_descriptor(material, radius, length, scoring)}")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        make_directory(energy_dir)
        filename = (
            f"t90_vs_Energy__{material}_I{format_value(accel_ref)}mA_R{format_value(radius)}_L{format_value(length)}.png"
        )
        fig.tight_layout()
        fig.savefig(energy_dir / filename)
        plt.close(fig)


def atoms_per_s_per_mA_vs_energy_plots(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    extra = ["atoms_per_s_per_mA_per_atom", "atoms_per_s_per_mA_cm3"]
    energy_dir = plots_dir / "energy_scans"
    for key, group in _iter_geometry_energy_groups(trans_summary, extra):
        beam, material, radius, length, scoring = key
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.plot(
            group["beam_energy_MeV"],
            group["atoms_per_s_per_mA_per_atom"],
            marker="o",
            label="per atom",
        )
        ax.plot(
            group["beam_energy_MeV"],
            group["atoms_per_s_per_mA_cm3"],
            marker="s",
            label="per cm³",
        )
        ax.set_xlabel("Beam energy (MeV)")
        ax.set_ylabel("Atoms/s per mA")
        ax.set_yscale("log")
        ax.set_title(f"Atoms/s per mA vs Energy — {beam} {_geometry_descriptor(material, radius, length, scoring)}")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        make_directory(energy_dir)
        filename = f"AtomsPerS_per_mA_vs_Energy__{material}_R{format_value(radius)}_L{format_value(length)}.png"
        fig.tight_layout()
        fig.savefig(energy_dir / filename)
        plt.close(fig)


def useful_per_c_vs_energy_plots(trans_summary: pd.DataFrame, plots_dir: Path) -> None:
    extra = ["Atoms_per_C"]
    energy_dir = plots_dir / "energy_scans"
    for key, group in _iter_geometry_energy_groups(trans_summary, extra):
        beam, material, radius, length, scoring = key
        energies = sorted(group["beam_energy_MeV"].unique())
        iso_list = list(group["isotope"].unique())
        width = 0.8 / max(len(iso_list), 1)
        x_positions = np.arange(len(energies))
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for idx, isotope in enumerate(iso_list):
            iso_group = group[group["isotope"] == isotope].set_index("beam_energy_MeV")
            y_vals = [iso_group["Atoms_per_C"].get(energy, 0.0) for energy in energies]
            offsets = x_positions + (idx - (len(iso_list) - 1) / 2) * width
            ax.bar(offsets, y_vals, width=width, label=isotope)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{energy:g}" for energy in energies])
        ax.set_xlabel("Beam energy (MeV)")
        ax.set_ylabel("Useful atoms per Coulomb")
        ax.set_title(f"Useful atoms/C vs Energy — {beam} {_geometry_descriptor(material, radius, length, scoring)}")
        ax.set_yscale("log")
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        ax.legend()
        make_directory(energy_dir)
        filename = f"UsefulPerC_vs_Energy__{material}_R{format_value(radius)}_L{format_value(length)}.png"
        fig.tight_layout()
        fig.savefig(energy_dir / filename)
        plt.close(fig)


def _parse_current_overrides(values: Sequence[str], prefix: str, beam: str) -> Dict[str, float]:
    if not values:
        return {}
    overrides: Dict[str, float] = {}
    for idx, raw in enumerate(values, start=1):
        if "=" in raw:
            label, val = raw.split("=", 1)
        else:
            label, val = f"{prefix}{idx}", raw
        label = label.strip() or f"{prefix}{idx}"
        try:
            amp = float(val)
        except ValueError as exc:
            raise ValueError(f"Invalid {beam} current '{raw}': {exc}") from exc
        if amp < 0:
            raise ValueError(f"{beam} current must be non-negative: '{raw}'")
        overrides[label] = amp
    if not overrides:
        raise ValueError(f"No valid {beam} currents were provided.")
    return overrides


def configure_current_overrides(
    electron_values: Sequence[str], proton_values: Sequence[str]
) -> None:
    global ELECTRON_CURRENTS_A, PROTON_CURRENTS_A
    global ELECTRON_PRIORITY_LABEL, PROTON_PRIORITY_LABEL

    electron_override = _parse_current_overrides(
        electron_values, "e_custom", "electron"
    )
    proton_override = _parse_current_overrides(proton_values, "p_custom", "proton")

    ELECTRON_CURRENTS_A = (
        electron_override if electron_override else dict(DEFAULT_ELECTRON_CURRENTS_A)
    )
    PROTON_CURRENTS_A = (
        proton_override if proton_override else dict(DEFAULT_PROTON_CURRENTS_A)
    )

    if not ELECTRON_CURRENTS_A:
        raise ValueError("At least one electron current must be defined.")
    if not PROTON_CURRENTS_A:
        raise ValueError("At least one proton current must be defined.")

    ELECTRON_PRIORITY_LABEL = next(reversed(ELECTRON_CURRENTS_A))
    PROTON_PRIORITY_LABEL = next(reversed(PROTON_CURRENTS_A))


def ensure_requirements() -> None:
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Required Python packages are missing. Install dependencies via `pip install -r requirements.txt`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse neutron-yield scan results and generate summary tables and plots."
    )
    parser.add_argument("--in", dest="input_dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing simulation results (default: results)")
    parser.add_argument("--out", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for analysis output (default: out)")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots")
    parser.add_argument(
        "--electron-current",
        dest="electron_currents",
        action="append",
        default=[],
        metavar="LABEL=AMP",
        help="Override electron reference currents used for flux projections "
        "(example: e2mA=0.002). Repeat to provide multiple values; omit to use defaults.",
    )
    parser.add_argument(
        "--proton-current",
        dest="proton_currents",
        action="append",
        default=[],
        metavar="LABEL=AMP",
        help="Override proton reference currents (example: p20mA=0.02).",
    )
    return parser.parse_args()


def main() -> int:
    ensure_requirements()
    args = parse_args()
    try:
        configure_current_overrides(args.electron_currents, args.proton_currents)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    spectra_dir = plots_dir / "spectra"
    angles_dir = plots_dir / "angles"
    thickness_dir = plots_dir / "yield_vs_thickness"
    thickness_r20_dir = plots_dir / "yield_vs_thickness_R20mm_variants"
    energy_dir = plots_dir / "yield_vs_energy"

    metadata_files = find_metadata_files(results_dir)
    if not metadata_files:
        print(f"No metadata files found under {results_dir}")
        return 1

    records: List[RunRecord] = []
    for meta_path in metadata_files:
        try:
            record = load_metadata_file(meta_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to parse metadata {meta_path}: {exc}")
            continue
        if record is None:
            continue
        records.append(record)

    if not records:
        print("No valid records were loaded.")
        return 1

    make_directory(output_dir)

    summary_df = build_summary_dataframe(records)
    summary_csv = output_dir / "summary_yield.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary written to {summary_csv}")

    trans_summary_path = output_dir / "transmutation_summary.csv"
    band_contrib_path = output_dir / "band_contributions.csv"
    trans_summary_df = _read_optional_csv(trans_summary_path)
    band_contrib_df = _read_optional_csv(band_contrib_path)
    top3_df = find_top3_by_flux(summary_df)

    if not args.no_plots:
        for record in records:
            df = load_ntuple(record.ntuple_path) if record.ntuple_path else pd.DataFrame()
            spectrum_plot(record, df, spectra_dir)
            angular_plot(record, df, angles_dir)
        yield_vs_thickness_plot(summary_df, thickness_dir)
        yield_vs_thickness_r20_variants(summary_df, thickness_r20_dir)
        yield_vs_energy_plot(summary_df, energy_dir)
        time_to90_vs_current_plot(trans_summary_df, plots_dir)
        lambda_eff_vs_current_plot(trans_summary_df, plots_dir)
        rate_bands_stacked_plot(band_contrib_df, plots_dir)
        atoms_per_coulomb_plot(trans_summary_df, plots_dir)
        compare_material_spectra(records, summary_df, plots_dir)
        lambda_eff_vs_energy_plots(trans_summary_df, plots_dir)
        t90_vs_energy_plots(trans_summary_df, plots_dir)
        atoms_per_s_per_mA_vs_energy_plots(trans_summary_df, plots_dir)
        useful_per_c_vs_energy_plots(trans_summary_df, plots_dir)

    if not top3_df.empty:
        top3_csv = output_dir / "top3_by_flux.csv"
        top3_df.to_csv(top3_csv, index=False)
        print(f"Top-3 configurations written to {top3_csv}")

    optimal_df = determine_optimal_sizes(summary_df)
    if not optimal_df.empty:
        optimal_csv = output_dir / "optimal_sizes.csv"
        optimal_df.to_csv(optimal_csv, index=False)
        print(f"Optimal sizes table written to {optimal_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
