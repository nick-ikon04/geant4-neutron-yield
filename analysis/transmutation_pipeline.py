#!/usr/bin/env python3
"""Neutron-induced transmutation analysis pipeline.

This script folds neutron flux spectra with reaction cross sections to
estimate reaction rates for long-lived isotopes and produces summary plots.

Usage:
    python3 analysis/transmutation_pipeline.py --config path/to/config.json

The configuration schema is documented in the module docstring of
`CaseConfig` below.  See `analysis/configs/` for worked examples.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E_CHARGE_C = 1.602176634e-19  # Coulombs
AVOGADRO = 6.02214076e23  # 1/mol
DEFAULT_GRID_POINTS = 2000
MILLIAMP = 1e-3
ACCEL_REF_CURRENTS_MA = {
    "electron": 20.0,
    "proton": 2.0,
}
BAND_EDGES_MEV = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
PHI_CASE_COL = "phi_n_cm2_s_MeV"
PHI_PER_MA_COL = "phi_per_mA_cm2_s_MeV"
PHI_ACCEL_REF_COL = "phi_accel_ref_cm2_s_MeV"


def _resolve_path(path_like: str, base: Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (base.parent if base.is_file() else base) / path
    return path.resolve()


@dataclass
class FluxSourceConfig:
    """Configuration for the neutron flux spectrum input.

    Supported kinds:
      * histogram: parse Geant4 CSV histogram together with run metadata.
      * csv: load precomputed spectrum with energy/flux columns.
    """

    kind: str
    histogram_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    energy_column: str = "energy_MeV"
    flux_column: str = "phi_n_per_cm2_per_s_per_MeV"


@dataclass
class CrossSectionConfig:
    path: Path
    energy_column: str = "energy_MeV"
    column_prefix: str = "sigma_"
    column_suffix: str = "_b"
    channel_map: Dict[str, str] = field(default_factory=dict)
    product_half_lives_s: Dict[str, float] = field(default_factory=dict)


@dataclass
class CaseConfig:
    """All parameters required to run a single transmutation case.

    Expected JSON keys:
      - case_tag: identifier used in filenames (optional, auto-generated if missing)
      - output_dir: directory for results (optional, defaults to analysis/transmutation_results/<case_tag>)
      - target_material, beam_particle, isotope: strings
      - beam_energy_MeV: float
      - beam_current_A: float
      - atomic_mass_g_mol: float
      - density_g_cm3: float
      - volume_cm3: float
      - irradiation_time_s: float (duration to use for saturation activity)
      - grid_points: optional int (defaults to 2000)
      - flux_source: mapping (see FluxSourceConfig)
      - cross_section: mapping (see CrossSectionConfig)
      - metadata_path: optional path to Geant4 run metadata (used if histogram flux is selected)
      - useful_channels_override: optional list of channels to treat as useful
    """

    case_tag: str
    target_material: str
    beam_particle: str
    beam_energy_MeV: float
    beam_current_A: float
    isotope: str
    atomic_mass_g_mol: float
    density_g_cm3: float
    volume_cm3: float
    irradiation_time_s: float
    flux_source: FluxSourceConfig
    cross_section: CrossSectionConfig
    grid_points: int = DEFAULT_GRID_POINTS
    output_dir: Path = Path("analysis/transmutation_results")
    useful_channels_override: Optional[List[str]] = None

    @property
    def n_nuclei(self) -> float:
        return self.density_g_cm3 * self.volume_cm3 * AVOGADRO / self.atomic_mass_g_mol


@dataclass
class GeometryInfo:
    radius_mm: float
    half_length_mm: float
    scoring_mm: float


@dataclass
class UsefulTotals:
    lambda_per_mA: float
    lambda_case: float
    lambda_accel: float
    atoms_per_s_case: float
    atoms_per_s_per_mA_sample: float
    atoms_per_s_accel_sample: float
    atoms_per_C: float
    atoms_per_s_per_mA_cm3: float
    integrand_per_mA: np.ndarray
    integrand_accel: np.ndarray
    energy_grid: np.ndarray
    case_current_mA: float
    accel_ref_mA: float


@dataclass
class CaseRunResult:
    csv_path: Path
    summary_row: Dict[str, object]
    band_contributions: List[Dict[str, object]]


def load_case_config(path: Path) -> CaseConfig:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    def _required(key: str):
        if key not in data:
            raise KeyError(f"Missing required config key '{key}' in {path}")
        return data[key]

    case_tag = data.get(
        "case_tag",
        f"{data.get('beam_particle','beam')}_{data.get('target_material','mat')}_E{data.get('beam_energy_MeV','?')}MeV_{data.get('isotope','iso')}",
    )

    output_dir = Path(
        data.get(
            "output_dir",
            Path("analysis/transmutation_results") / case_tag,
        )
    )

    flux_cfg_map = data.get("flux_source")
    if not isinstance(flux_cfg_map, Mapping):
        raise ValueError("flux_source section must be provided in config.")
    flux_kind = flux_cfg_map.get("kind", "histogram")
    flux_cfg = FluxSourceConfig(
        kind=flux_kind,
        histogram_path=_resolve_path(flux_cfg_map["histogram_path"], path)
        if flux_kind == "histogram"
        else None,
        metadata_path=_resolve_path(flux_cfg_map.get("metadata_path"), path)
        if flux_kind == "histogram" and flux_cfg_map.get("metadata_path")
        else None,
        csv_path=_resolve_path(flux_cfg_map["path"], path)
        if flux_kind == "csv"
        else None,
        energy_column=flux_cfg_map.get("energy_column", "energy_MeV"),
        flux_column=flux_cfg_map.get("flux_column", "phi_n_per_cm2_per_s_per_MeV"),
    )

    xs_cfg_map = data.get("cross_section")
    if not isinstance(xs_cfg_map, Mapping):
        raise ValueError("cross_section section must be provided in config.")
    cross_section = CrossSectionConfig(
        path=_resolve_path(_required("cross_section")["path"], path),
        energy_column=xs_cfg_map.get("energy_column", "energy_MeV"),
        column_prefix=xs_cfg_map.get("column_prefix", "sigma_"),
        column_suffix=xs_cfg_map.get("column_suffix", "_b"),
        channel_map=xs_cfg_map.get("channel_map", {}),
        product_half_lives_s=xs_cfg_map.get("product_half_lives_s", {}),
    )

    useful_override = data.get("useful_channels_override")
    if useful_override is not None and not isinstance(useful_override, list):
        raise ValueError("useful_channels_override must be a list when provided.")

    return CaseConfig(
        case_tag=case_tag,
        target_material=_required("target_material"),
        beam_particle=_required("beam_particle"),
        beam_energy_MeV=float(_required("beam_energy_MeV")),
        beam_current_A=float(_required("beam_current_A")),
        isotope=_required("isotope"),
        atomic_mass_g_mol=float(_required("atomic_mass_g_mol")),
        density_g_cm3=float(_required("density_g_cm3")),
        volume_cm3=float(_required("volume_cm3")),
        irradiation_time_s=float(_required("irradiation_time_s")),
        flux_source=flux_cfg,
        cross_section=cross_section,
        grid_points=int(data.get("grid_points", DEFAULT_GRID_POINTS)),
        output_dir=output_dir.resolve(),
        useful_channels_override=useful_override,
    )


def load_geant4_histogram_flux(
    flux_cfg: FluxSourceConfig, case: CaseConfig
) -> pd.DataFrame:
    if not flux_cfg.histogram_path or not flux_cfg.metadata_path:
        raise ValueError("Histogram flux source requires histogram_path and metadata_path.")

    metadata = json.loads(flux_cfg.metadata_path.read_text(encoding="utf-8"))
    n_primary = float(metadata.get("N_primary", 0.0))
    if n_primary <= 0:
        raise ValueError(f"N_primary not positive in metadata {flux_cfg.metadata_path}")

    radius_mm = float(metadata.get("target_radius_mm", 0.0))
    half_length_mm = float(metadata.get("target_half_length_mm", 0.0))
    scoring_thickness = float(metadata.get("scoring_shell_thickness_mm", 0.0))
    if radius_mm <= 0 or half_length_mm <= 0:
        raise ValueError("Target geometry missing or zero in metadata.")

    text = flux_cfg.histogram_path.read_text(encoding="utf-8").splitlines()
    header_idx = next(
        (idx for idx, line in enumerate(text) if line.lower().startswith("entries")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Histogram header missing entries row in {flux_cfg.histogram_path}")
    raw = np.loadtxt(
        flux_cfg.histogram_path,
        delimiter=",",
        comments="#",
        skiprows=header_idx + 1,
    )
    if raw.ndim == 1:
        raw = raw.reshape((-1, raw.shape[0]))
    if raw.shape[0] < 3:
        raise ValueError(f"Unexpected histogram content in {flux_cfg.histogram_path}")

    axis_line = next((line for line in text if line.startswith("#axis")), None)
    if axis_line is None:
        raise ValueError(f"Histogram header missing axis definition in {flux_cfg.histogram_path}")
    _, _, bins_str, xmin_str, xmax_str = axis_line.split()
    n_bins = int(bins_str)
    energy_min = float(xmin_str)
    energy_max = float(xmax_str)
    bin_width = (energy_max - energy_min) / n_bins

    # Skip underflow and overflow rows
    data = raw[1:-1, :]
    if len(data) != n_bins:
        raise ValueError(
            f"Histogram bin count mismatch: expected {n_bins}, got {len(data)} in {flux_cfg.histogram_path}"
        )

    counts = data[:, 0]
    energy_centers = energy_min + (np.arange(n_bins) + 0.5) * bin_width

    counts_per_primary = counts / n_primary
    radius_cm = radius_mm / 10.0
    length_cm = (half_length_mm * 2.0) / 10.0
    side_area = 2.0 * math.pi * radius_cm * length_cm
    end_area = 2.0 * math.pi * radius_cm**2
    surface_area_cm2 = max(side_area + end_area, 1e-12)

    phi_per_primary = np.divide(
        counts_per_primary,
        bin_width * surface_area_cm2,
        out=np.zeros_like(counts_per_primary),
        where=(bin_width > 0) & (surface_area_cm2 > 0),
    )
    particles_per_s_per_mA = MILLIAMP / E_CHARGE_C
    phi_per_mA = phi_per_primary * particles_per_s_per_mA
    case_current_mA = case.beam_current_A / MILLIAMP if case.beam_current_A > 0 else 0.0
    phi_case = phi_per_mA * case_current_mA

    accel_ref_mA = ACCEL_REF_CURRENTS_MA.get(case.beam_particle, case_current_mA)
    phi_accel_ref = phi_per_mA * accel_ref_mA

    df = pd.DataFrame(
        {
            "energy_MeV": energy_centers,
            PHI_CASE_COL: phi_case,
            PHI_PER_MA_COL: phi_per_mA,
            PHI_ACCEL_REF_COL: phi_accel_ref,
            "phi_per_primary": phi_per_primary,
            "phi_2mA_cm2_s_MeV": phi_per_mA * 2.0,
            "phi_20mA_cm2_s_MeV": phi_per_mA * 20.0,
        }
    )
    df = df[df[PHI_CASE_COL] >= 0.0].copy()
    df.sort_values("energy_MeV", inplace=True)
    df.attrs["geometry"] = GeometryInfo(
        radius_mm=radius_mm,
        half_length_mm=half_length_mm,
        scoring_mm=scoring_thickness,
    )
    return df


def load_flux_spectrum(flux_cfg: FluxSourceConfig, case: CaseConfig) -> pd.DataFrame:
    if flux_cfg.kind == "histogram":
        return load_geant4_histogram_flux(flux_cfg, case)
    if flux_cfg.kind == "csv":
        if not flux_cfg.csv_path:
            raise ValueError("CSV flux source requires 'path'.")
        df = pd.read_csv(flux_cfg.csv_path)
        df = df[[flux_cfg.energy_column, flux_cfg.flux_column]].dropna()
        df.columns = ["energy_MeV", PHI_CASE_COL]
        case_current_mA = case.beam_current_A / MILLIAMP if case.beam_current_A > 0 else 0.0
        if case_current_mA > 0:
            phi_per_mA = df[PHI_CASE_COL].to_numpy(dtype=float) / case_current_mA
        else:
            phi_per_mA = np.zeros_like(df[PHI_CASE_COL].to_numpy(dtype=float))
        accel_ref_mA = ACCEL_REF_CURRENTS_MA.get(case.beam_particle, case_current_mA)
        df[PHI_PER_MA_COL] = phi_per_mA
        df[PHI_ACCEL_REF_COL] = phi_per_mA * accel_ref_mA
        df["phi_per_primary"] = math.nan
        df["phi_2mA_cm2_s_MeV"] = phi_per_mA * 2.0
        df["phi_20mA_cm2_s_MeV"] = phi_per_mA * 20.0
        df.sort_values("energy_MeV", inplace=True)
        return df
    raise ValueError(f"Unsupported flux source kind '{flux_cfg.kind}'")


def _infer_channel_name(column: str, prefix: str, suffix: str) -> str:
    name = column
    if prefix and name.startswith(prefix):
        name = name[len(prefix) :]
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.replace("_", ",")


def _talys_target_numbers(directory: Path) -> Tuple[int, int]:
    all_tot = directory / "all.tot"
    if not all_tot.exists():
        raise ValueError(f"TALYS directory '{directory}' missing all.tot")
    z_val: Optional[int] = None
    a_val: Optional[int] = None
    with all_tot.open(encoding="utf-8") as handle:
        in_target = False
        for raw in handle:
            line = raw.strip()
            if line.startswith("# target"):
                in_target = True
                continue
            if not line.startswith("#"):
                if in_target and z_val is not None and a_val is not None:
                    break
                continue
            if not in_target:
                continue
            if line.startswith("#   Z:"):
                try:
                    z_val = int(float(line.split(":", 1)[1]))
                except ValueError:
                    continue
            elif line.startswith("#   A:"):
                try:
                    a_val = int(float(line.split(":", 1)[1]))
                except ValueError:
                    continue
            if z_val is not None and a_val is not None:
                break
    if z_val is None or a_val is None:
        raise ValueError(f"Unable to extract target Z/A from {all_tot}")
    return z_val, a_val


def _talys_channel_candidates(channel: str, z: int, a: int) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if channel == "n,gamma":
        candidates.append((f"rp{z:03d}{a + 1:03d}", "tot"))
    elif channel == "n,2n":
        candidates.append((f"rp{z:03d}{max(a - 1, 0):03d}", "tot"))
    elif channel == "n,3n":
        candidates.append((f"rp{z:03d}{max(a - 2, 0):03d}", "tot"))
    elif channel == "n,p":
        candidates.append((f"rp{max(z - 1, 0):03d}{a:03d}", "tot"))
    elif channel == "n,alpha":
        candidates.append((f"rp{max(z - 2, 0):03d}{max(a - 3, 0):03d}", "tot"))
    elif channel in {"n,f", "f"}:
        candidates.append(("fission", "tot"))
    return candidates


def _talys_load_table(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    energies: List[float] = []
    sigmas_mb: List[float] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                energy = float(parts[0])
                sigma = float(parts[1])
            except ValueError:
                continue
            energies.append(energy)
            sigmas_mb.append(sigma)
    if not energies:
        raise ValueError(f"No numeric data in {path}")
    return np.asarray(energies, dtype=float), np.asarray(sigmas_mb, dtype=float) * 1e-3


def _generate_column_name(cfg: CrossSectionConfig, channel: str) -> str:
    if cfg.channel_map and channel in cfg.channel_map:
        return cfg.channel_map[channel]
    safe = channel.replace(",", "_")
    return f"{cfg.column_prefix}{safe}{cfg.column_suffix}"


def _desired_cross_section_channels(case: CaseConfig) -> Tuple[str, ...]:
    channels: List[str] = []
    if case.cross_section.channel_map:
        channels.extend(case.cross_section.channel_map.keys())
    useful = get_useful_channels(case)
    channels.extend(ch for ch in useful if ch not in channels)
    channels.extend(
        ch for ch in case.cross_section.product_half_lives_s.keys() if ch not in channels
    )
    # Remove duplicates while preserving order
    seen: set[str] = set()
    ordered: List[str] = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return tuple(ordered)


def _load_talys_cross_sections(
    case: CaseConfig, cfg: CrossSectionConfig
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    directory = cfg.path
    if not directory.is_dir():
        raise ValueError(f"TALYS path '{directory}' is not a directory.")
    z_val, a_val = _talys_target_numbers(directory)
    desired_channels = _desired_cross_section_channels(case)
    if not desired_channels:
        raise ValueError("No reaction channels specified for TALYS cross sections.")

    energy_master: Optional[np.ndarray] = None
    columns: Dict[str, np.ndarray] = {}
    channel_map: Dict[str, str] = {}
    for channel in desired_channels:
        cand_specs = _talys_channel_candidates(channel, z_val, a_val)
        selected_file: Optional[Path] = None
        for stem, suffix in cand_specs:
            if suffix == "tot":
                matches = sorted(directory.glob(f"{stem}*.tot"))
            else:
                matches = sorted(directory.glob(f"{stem}*.{suffix}"))
            if matches:
                selected_file = matches[0]
                break
        if selected_file is None:
            print(f"[talys] Channel '{channel}' not found under {directory}")
            continue
        try:
            energies, sigma_barn = _talys_load_table(selected_file)
        except ValueError:
            continue
        if energy_master is None:
            energy_master = energies
        else:
            if energies.size != energy_master.size or not np.allclose(energies, energy_master):
                sigma_barn = np.interp(energy_master, energies, sigma_barn, left=0.0, right=0.0)
        column_name = _generate_column_name(cfg, channel)
        columns[column_name] = sigma_barn
        channel_map[channel] = column_name

    if energy_master is None or not columns:
        raise ValueError(f"Failed to load TALYS cross sections from '{directory}'")

    df = pd.DataFrame({cfg.energy_column: energy_master})
    for column_name, values in columns.items():
        df[column_name] = values
    return df, channel_map


def load_cross_sections(case: CaseConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cfg = case.cross_section
    if cfg.path.is_dir():
        return _load_talys_cross_sections(case, cfg)

    df = pd.read_csv(cfg.path)
    if cfg.energy_column not in df.columns:
        raise ValueError(f"Energy column '{cfg.energy_column}' not found in {cfg.path}")
    df = df.dropna(subset=[cfg.energy_column]).sort_values(cfg.energy_column)
    channel_columns: Dict[str, str] = {}
    if cfg.channel_map:
        for channel, column in cfg.channel_map.items():
            if column in df.columns:
                channel_columns[channel] = column
    else:
        for column in df.columns:
            if column == cfg.energy_column:
                continue
            if cfg.column_prefix and not column.startswith(cfg.column_prefix):
                continue
            if cfg.column_suffix and not column.endswith(cfg.column_suffix):
                continue
            channel = _infer_channel_name(column, cfg.column_prefix, cfg.column_suffix)
            channel_columns[channel] = column
    if not channel_columns:
        raise ValueError(f"No cross-section columns found in {cfg.path}")
    return df, channel_columns


USEFUL_CHANNELS_DEFAULT: Dict[str, Sequence[str]] = {
    "129I": ("n,gamma", "n,2n", "n,p", "n,alpha", "n,3n"),
    "135Cs": ("n,gamma", "n,2n", "n,p", "n,alpha", "n,3n"),
    "241Am": ("n,f", "n,2n", "n,p", "n,alpha", "n,3n"),
}


def get_useful_channels(case: CaseConfig) -> Tuple[str, ...]:
    if case.useful_channels_override is not None:
        return tuple(case.useful_channels_override)
    return tuple(USEFUL_CHANNELS_DEFAULT.get(case.isotope, ()))


@dataclass
class ChannelResult:
    channel: str
    energy_grid: np.ndarray
    phi_interp: np.ndarray
    phi_per_mA_interp: np.ndarray
    sigma_interp_cm2: np.ndarray
    integrand: np.ndarray
    integrand_per_mA: np.ndarray
    cumulative_rate: np.ndarray
    R_nuc: float
    lambda_per_mA: float
    R_vol: float
    atoms_per_s: float
    atoms_per_C: float
    A_sat_Bq: float
    is_useful: bool


def make_energy_grid(
    flux_df: pd.DataFrame, xs_df: pd.DataFrame, energy_column: str, n_points: int
) -> Tuple[np.ndarray, float, float]:
    emin = max(flux_df["energy_MeV"].min(), xs_df[energy_column].min())
    emax = min(flux_df["energy_MeV"].max(), xs_df[energy_column].max())
    if not math.isfinite(emin) or not math.isfinite(emax) or emin >= emax:
        raise ValueError("Flux and cross-section energy ranges do not overlap.")
    grid = np.linspace(emin, emax, n_points)
    return grid, emin, emax


def compute_channel_results(
    case: CaseConfig,
    flux_df: pd.DataFrame,
    xs_df: pd.DataFrame,
    channel_columns: Dict[str, str],
) -> List[ChannelResult]:
    E_grid, _, _ = make_energy_grid(flux_df, xs_df, case.cross_section.energy_column, case.grid_points)
    useful_channels = get_useful_channels(case)
    flux_energy = flux_df["energy_MeV"].to_numpy()
    flux_values = flux_df[PHI_CASE_COL].to_numpy(dtype=float)
    if PHI_PER_MA_COL in flux_df.columns:
        flux_per_mA_values = flux_df[PHI_PER_MA_COL].to_numpy(dtype=float)
    else:
        current_mA = case.beam_current_A / MILLIAMP if case.beam_current_A > 0 else 0.0
        if current_mA > 0:
            flux_per_mA_values = flux_values / current_mA
        else:
            flux_per_mA_values = np.zeros_like(flux_values)
    xs_energy = xs_df[case.cross_section.energy_column].to_numpy()

    results: List[ChannelResult] = []
    for channel, column in channel_columns.items():
        sigma_values_barn = xs_df[column].to_numpy(dtype=float)
        phi_interp = np.interp(E_grid, flux_energy, flux_values, left=0.0, right=0.0)
        phi_per_mA_interp = np.interp(E_grid, flux_energy, flux_per_mA_values, left=0.0, right=0.0)
        sigma_interp = np.interp(E_grid, xs_energy, sigma_values_barn, left=0.0, right=0.0)
        sigma_cm2 = sigma_interp * 1e-24
        integrand = phi_interp * sigma_cm2
        integrand_per_mA = phi_per_mA_interp * sigma_cm2
        R_nuc = float(np.trapezoid(integrand, E_grid))
        lambda_per_mA = float(np.trapezoid(integrand_per_mA, E_grid))
        R_vol = R_nuc * case.n_nuclei
        atoms_per_s = R_vol
        atoms_per_C = atoms_per_s / case.beam_current_A if case.beam_current_A > 0 else math.nan

        half_life = case.cross_section.product_half_lives_s.get(channel, 0.0)
        if half_life and half_life > 0:
            lambda_decay = math.log(2.0) / half_life
            A_sat = atoms_per_s * (1.0 - math.exp(-lambda_decay * case.irradiation_time_s))
        else:
            A_sat = 0.0

        trapezoids = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(E_grid)
        cumulative = np.concatenate(([0.0], np.cumsum(trapezoids)))

        results.append(
            ChannelResult(
                channel=channel,
                energy_grid=E_grid,
                phi_interp=phi_interp,
                phi_per_mA_interp=phi_per_mA_interp,
                sigma_interp_cm2=sigma_cm2,
                integrand=integrand,
                integrand_per_mA=integrand_per_mA,
                cumulative_rate=cumulative,
                R_nuc=R_nuc,
                lambda_per_mA=lambda_per_mA,
                R_vol=R_vol,
                atoms_per_s=atoms_per_s,
                atoms_per_C=atoms_per_C,
                A_sat_Bq=A_sat,
                is_useful=channel in useful_channels,
            )
        )
    return results


def plot_flux_and_cross_section(case: CaseConfig, result: ChannelResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    eps = 1e-40
    ax1.plot(result.energy_grid, np.clip(result.phi_interp, eps, None), label="Flux", color="tab:blue")
    ax1.set_xlabel("Energy (MeV)")
    ax1.set_ylabel("Flux [n cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]", color="tab:blue")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        result.energy_grid,
        np.clip(result.sigma_interp_cm2, eps, None),
        label="Cross section",
        color="tab:orange",
    )
    ax2.set_ylabel("Cross section [cm$^{2}$]", color="tab:orange")
    ax2.set_yscale("log")

    plt.title(
        f"Flux vs σ — {case.case_tag} channel {result.channel}",
        fontsize=11,
    )
    fig.tight_layout()
    channel_label = result.channel.replace(",", "_")
    plt.savefig(out_dir / f"Flux_and_CrossSection_{case.case_tag}_{channel_label}.png")
    plt.close(fig)


def plot_integrand(case: CaseConfig, result: ChannelResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eps = 1e-40
    ax.plot(
        result.energy_grid,
        np.clip(result.integrand, eps, None),
        color="tab:green",
    )
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("φ·σ [s$^{-1}$ MeV$^{-1}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(f"phi*sigma vs Energy — {case.case_tag} channel {result.channel}")
    fig.tight_layout()
    channel_label = result.channel.replace(",", "_")
    plt.savefig(out_dir / f"Integrand_{case.case_tag}_{channel_label}.png")
    plt.close(fig)


def plot_cumulative(case: CaseConfig, result: ChannelResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.energy_grid, result.cumulative_rate, color="tab:red")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Cumulative reaction rate [s$^{-1}$]")
    ax.grid(True, ls="--", alpha=0.3)
    plt.title(f"Cumulative rate — {case.case_tag} channel {result.channel}")
    fig.tight_layout()
    channel_label = result.channel.replace(",", "_")
    plt.savefig(out_dir / f"Cumulative_{case.case_tag}_{channel_label}.png")
    plt.close(fig)


def plot_efficiency_bars(case: CaseConfig, results: Sequence[ChannelResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_results = [res for res in results if res.channel != "n,3n"]
    if not filtered_results:
        return

    channels = [res.channel for res in filtered_results]
    atoms_per_c = [max(res.atoms_per_C, 1e-40) for res in filtered_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(channels, atoms_per_c, color="tab:purple")
    ax.set_xlabel("Reaction channel")
    ax.set_ylabel("Atoms per Coulomb")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.title(f"Transmutation efficiency per Coulomb — {case.case_tag}")
    fig.tight_layout()
    plt.savefig(out_dir / f"EfficiencyBars_{case.case_tag}.png")
    plt.close(fig)


def channel_results_to_rows(case: CaseConfig, results: Sequence[ChannelResult]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    case_current_mA = case.beam_current_A / MILLIAMP if case.beam_current_A > 0 else 0.0
    accel_ref_mA = ACCEL_REF_CURRENTS_MA.get(case.beam_particle, case_current_mA)
    atoms_total = case.n_nuclei
    for res in results:
        lambda_case = res.lambda_per_mA * case_current_mA
        lambda_accel = res.lambda_per_mA * accel_ref_mA
        atoms_per_s_per_mA_sample = res.lambda_per_mA * atoms_total
        atoms_per_s_case = lambda_case * atoms_total
        atoms_per_s_accel = lambda_accel * atoms_total
        t12_case_days = _safe_days(lambda_case, math.log(2.0))
        t90_case_days = _safe_days(lambda_case, math.log(10.0))
        t12_accel_days = _safe_days(lambda_accel, math.log(2.0))
        t90_accel_days = _safe_days(lambda_accel, math.log(10.0))
        rows.append(
            {
                "target_material": case.target_material,
                "beam_particle": case.beam_particle,
                "beam_energy_MeV": case.beam_energy_MeV,
                "beam_current_A": case.beam_current_A,
                "isotope": case.isotope,
                "reaction_channel": res.channel,
                "R_nuc_1_per_s": res.R_nuc,
                "lambda_per_mA_s_inv": res.lambda_per_mA,
                "lambda_case_s_inv": lambda_case,
                "lambda_accel_s_inv": lambda_accel,
                "R_vol_1_per_s_cm3": res.R_vol,
                "P_atoms_per_s_cm3": res.atoms_per_s,
                "atoms_per_s_per_mA_sample": atoms_per_s_per_mA_sample,
                "atoms_per_s_case_sample": atoms_per_s_case,
                "atoms_per_s_accel_sample": atoms_per_s_accel,
                "t12_case_days": t12_case_days,
                "t12_accel_days": t12_accel_days,
                "t90_case_days": t90_case_days,
                "t90_accel_days": t90_accel_days,
                "Atoms_per_C": res.atoms_per_C,
                "A_sat_Bq": res.A_sat_Bq,
                "note_useful_for_transmutation": "yes" if res.is_useful else "no",
            }
        )
    return rows


def add_useful_totals(
    case: CaseConfig, results: Sequence[ChannelResult], rows: List[Dict[str, object]]
) -> Optional[UsefulTotals]:
    useful = [res for res in results if res.is_useful]
    if not useful:
        return None
    lambda_per_mA = sum(res.lambda_per_mA for res in useful)
    case_current_mA = case.beam_current_A / MILLIAMP if case.beam_current_A > 0 else 0.0
    accel_ref_mA = ACCEL_REF_CURRENTS_MA.get(case.beam_particle, case_current_mA)
    lambda_case = lambda_per_mA * case_current_mA
    lambda_accel = lambda_per_mA * accel_ref_mA
    R_vol = sum(res.R_vol for res in useful)
    atoms_per_s = sum(res.atoms_per_s for res in useful)
    atoms_per_C = sum(res.atoms_per_C for res in useful)
    A_sat = sum(res.A_sat_Bq for res in useful)
    atoms_total = case.n_nuclei
    atoms_per_s_per_mA_sample = lambda_per_mA * atoms_total
    atoms_per_s_case = lambda_case * atoms_total
    atoms_per_s_accel = lambda_accel * atoms_total
    volume_cm3 = case.volume_cm3 if case.volume_cm3 > 0 else 1.0
    n_density = atoms_total / volume_cm3
    atoms_per_s_per_mA_cm3 = n_density * lambda_per_mA
    t12_case_days = _safe_days(lambda_case, math.log(2.0))
    t90_case_days = _safe_days(lambda_case, math.log(10.0))
    t12_accel_days = _safe_days(lambda_accel, math.log(2.0))
    t90_accel_days = _safe_days(lambda_accel, math.log(10.0))
    row = {
        "target_material": case.target_material,
        "beam_particle": case.beam_particle,
        "beam_energy_MeV": case.beam_energy_MeV,
        "beam_current_A": case.beam_current_A,
        "isotope": case.isotope,
        "reaction_channel": "useful_total",
        "R_nuc_1_per_s": lambda_case,
        "lambda_per_mA_s_inv": lambda_per_mA,
        "lambda_case_s_inv": lambda_case,
        "lambda_accel_s_inv": lambda_accel,
        "R_vol_1_per_s_cm3": R_vol,
        "P_atoms_per_s_cm3": atoms_per_s,
        "atoms_per_s_per_mA_sample": atoms_per_s_per_mA_sample,
        "atoms_per_s_case_sample": atoms_per_s_case,
        "atoms_per_s_accel_sample": atoms_per_s_accel,
        "atoms_per_s_per_mA_cm3": atoms_per_s_per_mA_cm3,
        "atoms_per_s_per_mA_per_atom": lambda_per_mA,
        "t12_case_days": t12_case_days,
        "t12_accel_days": t12_accel_days,
        "t90_case_days": t90_case_days,
        "t90_accel_days": t90_accel_days,
        "Atoms_per_C": atoms_per_C,
        "A_sat_Bq": A_sat,
        "note_useful_for_transmutation": "yes",
    }
    rows.append(row)
    integrand_per_mA = np.sum(np.stack([res.integrand_per_mA for res in useful], axis=0), axis=0)
    integrand_accel = integrand_per_mA * accel_ref_mA
    return UsefulTotals(
        lambda_per_mA=lambda_per_mA,
        lambda_case=lambda_case,
        lambda_accel=lambda_accel,
        atoms_per_s_case=atoms_per_s_case,
        atoms_per_s_per_mA_sample=atoms_per_s_per_mA_sample,
        atoms_per_s_accel_sample=atoms_per_s_accel,
        atoms_per_C=atoms_per_C,
        atoms_per_s_per_mA_cm3=atoms_per_s_per_mA_cm3,
        integrand_per_mA=integrand_per_mA,
        integrand_accel=integrand_accel,
        energy_grid=useful[0].energy_grid,
        case_current_mA=case_current_mA,
        accel_ref_mA=accel_ref_mA,
    )


def _safe_days(rate: float, numerator: float) -> float:
    if rate <= 0:
        return math.inf
    return numerator / rate / 86400.0


def build_transmutation_summary_row(
    case: CaseConfig, totals: UsefulTotals, geometry: Optional[GeometryInfo]
) -> Dict[str, object]:
    geometry = geometry or GeometryInfo(radius_mm=math.nan, half_length_mm=math.nan, scoring_mm=math.nan)
    return {
        "case_tag": case.case_tag,
        "beam_particle": case.beam_particle,
        "beam_energy_MeV": case.beam_energy_MeV,
        "target_material": case.target_material,
        "isotope": case.isotope,
        "geometry_L_mm": geometry.half_length_mm,
        "geometry_R_mm": geometry.radius_mm,
        "geometry_S_mm": geometry.scoring_mm,
        "beam_current_A": case.beam_current_A,
        "case_current_mA": totals.case_current_mA,
        "accel_ref_mA": totals.accel_ref_mA,
        "lambda_per_mA_s_inv": totals.lambda_per_mA,
        "lambda_case_s_inv": totals.lambda_case,
        "lambda_accel_s_inv": totals.lambda_accel,
        "t90_case_days": _safe_days(totals.lambda_case, math.log(10.0)),
        "t90_accel_days": _safe_days(totals.lambda_accel, math.log(10.0)),
        "t12_case_days": _safe_days(totals.lambda_case, math.log(2.0)),
        "t12_accel_days": _safe_days(totals.lambda_accel, math.log(2.0)),
        "atoms_per_s_case_sample": totals.atoms_per_s_case,
        "atoms_per_s_accel_sample": totals.atoms_per_s_accel_sample,
        "atoms_per_s_per_mA_sample": totals.atoms_per_s_per_mA_sample,
        "atoms_per_s_per_mA_cm3": totals.atoms_per_s_per_mA_cm3,
        "atoms_per_s_per_mA_per_atom": totals.lambda_per_mA,
        "Atoms_per_C": totals.atoms_per_C,
    }


def compute_band_contributions(case: CaseConfig, totals: UsefulTotals) -> List[Dict[str, object]]:
    energy = totals.energy_grid
    integrand = totals.integrand_per_mA
    total_lambda = totals.lambda_per_mA
    rows: List[Dict[str, object]] = []
    if energy.size == 0:
        return rows
    for e_min, e_max in zip(BAND_EDGES_MEV[:-1], BAND_EDGES_MEV[1:]):
        mask = (energy >= e_min) & (energy <= e_max)
        if mask.sum() < 2:
            continue
        lam_band = float(np.trapezoid(integrand[mask], energy[mask]))
        fraction = lam_band / total_lambda if total_lambda > 0 else 0.0
        rows.append(
            {
                "case_tag": case.case_tag,
                "beam_particle": case.beam_particle,
                "beam_energy_MeV": case.beam_energy_MeV,
                "target_material": case.target_material,
                "isotope": case.isotope,
                "energy_min_MeV": e_min,
                "energy_max_MeV": e_max,
                "lambda_per_mA_band": lam_band,
                "lambda_case_band": lam_band * totals.case_current_mA,
                "lambda_accel_band": lam_band * totals.accel_ref_mA,
                "fraction_of_total": fraction,
                "atoms_per_s_per_mA_band_sample": lam_band * case.n_nuclei,
            }
        )
    return rows


def find_resonance_peaks(energy: np.ndarray, sigma_values: np.ndarray, max_peaks: int = 8) -> List[Tuple[float, float]]:
    peaks: List[Tuple[float, float, float]] = []
    for idx in range(1, len(sigma_values) - 1):
        left = sigma_values[idx - 1]
        mid = sigma_values[idx]
        right = sigma_values[idx + 1]
        if mid > left and mid > right:
            prominence = mid - max(left, right)
            peaks.append((mid, energy[idx], prominence))
    if not peaks:
        return []
    peaks.sort(key=lambda item: (item[0], item[2]), reverse=True)
    selected = peaks[:max_peaks]
    selected.sort(key=lambda item: item[1])
    return [(val, eng) for val, eng, _ in selected]


def _case_descriptor(case: CaseConfig, geometry: Optional[GeometryInfo], i_ref_mA: float) -> str:
    parts = [
        f"{case.beam_particle}",
        f"E={case.beam_energy_MeV:g} MeV",
        case.target_material,
    ]
    if geometry:
        parts.append(f"L={geometry.half_length_mm:g} mm")
        parts.append(f"R={geometry.radius_mm:g} mm")
    parts.append(f"I_ref={i_ref_mA:g} mA")
    return ", ".join(parts)


def plot_resonance_figures(
    case: CaseConfig,
    flux_df: pd.DataFrame,
    xs_df: pd.DataFrame,
    channel_columns: Dict[str, str],
    useful_channels: Tuple[str, ...],
    totals: UsefulTotals,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    energy = totals.energy_grid
    if energy.size == 0:
        return
    xs_energy = xs_df[case.cross_section.energy_column].to_numpy()
    sigma_sum_barn = np.zeros_like(energy)
    for channel in useful_channels or channel_columns.keys():
        column = channel_columns.get(channel)
        if not column:
            continue
        sigma_values = xs_df[column].to_numpy(dtype=float)
        sigma_sum_barn += np.interp(energy, xs_energy, sigma_values, left=0.0, right=0.0)
    sigma_sum_cm2 = sigma_sum_barn * 1e-24
    flux_energy = flux_df["energy_MeV"].to_numpy()
    phi_accel_array = flux_df.get(PHI_ACCEL_REF_COL, flux_df[PHI_CASE_COL]).to_numpy(dtype=float)
    phi_accel_interp = np.interp(energy, flux_energy, phi_accel_array, left=0.0, right=0.0)

    geom: Optional[GeometryInfo] = flux_df.attrs.get("geometry")
    descriptor = _case_descriptor(case, geom, totals.accel_ref_mA)
    peaks = find_resonance_peaks(energy, sigma_sum_barn)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    eps = 1e-40
    ax1.plot(
        energy,
        np.clip(phi_accel_interp, eps, None),
        label=f"ϕ @ I_ref={totals.accel_ref_mA:g} mA",
        color="tab:cyan",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Energy (MeV)")
    ax1.set_ylabel("Flux [n cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(energy, np.clip(sigma_sum_barn, eps, None), color="tab:orange", label="Σσ useful")
    ax2.set_yscale("log")
    ax2.set_ylabel("Cross section [barns]")

    for idx, (sigma_val, e_val) in enumerate(peaks, start=1):
        ax2.axvline(e_val, color="tab:red", linestyle=":", alpha=0.4)
        ax2.text(
            e_val,
            sigma_val,
            f"{idx}:{e_val:.2f} MeV",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="tab:red",
        )

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, title=descriptor)
    fig.tight_layout()
    fig.savefig(plots_dir / f"Flux_and_CrossSection_{case.case_tag}_useful_total.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        energy,
        np.clip(totals.integrand_accel, eps, None),
        label=f"ϕσ @ I_ref={totals.accel_ref_mA:g} mA",
        color="tab:olive",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("ϕ·σ [s$^{-1}$ MeV$^{-1}$]")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    clipped_integrand = np.clip(totals.integrand_accel, eps, None)
    for idx, (_, e_val) in enumerate(peaks, start=1):
        ax.axvline(e_val, color="tab:red", linestyle=":", alpha=0.4)
        y_val = np.interp(e_val, energy, clipped_integrand)
        ax.text(
            e_val,
            y_val,
            f"{idx}",
            rotation=0,
            fontsize=8,
            color="tab:red",
            ha="left",
            va="bottom",
        )
    ax.legend(title=descriptor)
    fig2.tight_layout()
    fig2.savefig(plots_dir / f"Integrand_{case.case_tag}__phi_sigma_useful_total.png", bbox_inches="tight")
    plt.close(fig2)


def run_case(case: CaseConfig) -> CaseRunResult:
    output_dir = case.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    flux_df = load_flux_spectrum(case.flux_source, case)
    xs_df, channel_columns = load_cross_sections(case)
    results = compute_channel_results(case, flux_df, xs_df, channel_columns)

    channel_rows = channel_results_to_rows(case, results)
    useful_totals = add_useful_totals(case, results, channel_rows)

    csv_path = output_dir / f"{case.case_tag}_results.csv"
    pd.DataFrame(channel_rows).to_csv(csv_path, index=False)

    for res in results:
        plot_flux_and_cross_section(case, res, plots_dir)
        plot_integrand(case, res, plots_dir)
        plot_cumulative(case, res, plots_dir)
    plot_efficiency_bars(case, results, plots_dir)

    summary_row: Dict[str, object] = {}
    band_rows: List[Dict[str, object]] = []
    if useful_totals is not None:
        geometry: Optional[GeometryInfo] = flux_df.attrs.get("geometry")
        summary_row = build_transmutation_summary_row(case, useful_totals, geometry)
        band_rows = compute_band_contributions(case, useful_totals)
        useful_channels = get_useful_channels(case)
        plot_resonance_figures(
            case,
            flux_df,
            xs_df,
            channel_columns,
            useful_channels,
            useful_totals,
            plots_dir,
        )

    return CaseRunResult(csv_path=csv_path, summary_row=summary_row, band_contributions=band_rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold neutron spectra with TALYS cross sections to evaluate transmutation rates."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to JSON configuration describing a single analysis case.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case = load_case_config(args.config.resolve())
    result = run_case(case)
    print(f"Results written to {result.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
