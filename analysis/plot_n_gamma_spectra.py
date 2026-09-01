"""Produce n,γ cross-section spectra for I-129, Cs-135 and Am-241 from TALYS outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "analysis" / "data" / "talys"
OUTPUT_DIR = REPO_ROOT / "out" / "n_gamma_spectra"
PER_ISOTOPE_DIR = OUTPUT_DIR / "per_isotope"
PER_ISOTOPE_FULL_DIR = PER_ISOTOPE_DIR / "full_range"
PER_ISOTOPE_RESONANCE_DIR = PER_ISOTOPE_DIR / "resonance_region"
COMPARISON_DIR = OUTPUT_DIR / "comparison_with_endf"
ENDF_ONLY_DIR = COMPARISON_DIR / "endf_only"

ISOTOPE_FILES: Dict[str, Path] = {
    "I-129": DATA_ROOT / "129I" / "rp053130.tot",
    "Cs-135": DATA_ROOT / "135Cs" / "rp055136.tot",
    "Am-241": DATA_ROOT / "241Am" / "rp095242.tot",
}

ENDF_ROOT = REPO_ROOT / "analysis" / "data" / "endf"
ENDF_FILES: Dict[str, Path] = {
    "I-129": ENDF_ROOT / "E4R7187_e4.zvd.dat.txt",
    "Cs-135": ENDF_ROOT / "E4R7188_e4.zvd.dat.txt",
    "Am-241": ENDF_ROOT / "E4R7189_e4.zvd.dat.txt",
}


def load_tot_table(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read an ENDF/TALYS style .tot file and return energies & σ in barns."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    energies: List[float] = []
    sigmas: List[float] = []
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
            sigmas.append(sigma * 1e-3)  # convert mb → barns
    if not energies:
        raise ValueError(f"No numeric data read from {path}")
    return np.array(energies, dtype=float), np.array(sigmas, dtype=float)


def _clip_positive(values: np.ndarray) -> np.ndarray:
    """Keep a positive floor so log plots do not break on zeros."""
    return np.clip(values, 1e-12, None)


def _safe_filename(isotope: str) -> str:
    """Convert isotope names to filesystem-friendly tokens."""
    return isotope.replace(" ", "_").replace("-", "_")


def load_endf_ascii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a simple ASCII ENDF-like file exported from ZVView."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    energies: List[float] = []
    sigmas: List[float] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
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
            sigmas.append(sigma)
    if not energies:
        raise ValueError(f"No numeric data read from {path}")
    return np.array(energies, dtype=float), np.array(sigmas, dtype=float)


def save_spectra_csv(data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    csv_dir = OUTPUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for isotope, (energy, sigma) in data.items():
        out_path = csv_dir / f"{isotope.replace('-', '_')}_n_gamma.csv"
        np.savetxt(
            out_path,
            np.column_stack((energy, sigma)),
            delimiter=",",
            header="energy_MeV,cross_section_barn",
            comments="",
            fmt="%.9e",
        )


def plot_general_spectrum(data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for isotope, (energy, sigma) in data.items():
        ax.plot(energy, _clip_positive(sigma), label=isotope, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(
        min(energy[0] for energy, _ in data.values()),
        max(energy[-1] for energy, _ in data.values()),
    )
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("σₙ,γ [barn]")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n_gamma_general_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def plot_resonance_region(data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for isotope, (energy, sigma) in data.items():
        mask = energy <= 12
        ax.plot(energy[mask], _clip_positive(sigma[mask]), label=isotope, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    min_energy = min(energy[0] for energy, _ in data.values())
    ax.set_xlim(min_energy, 12)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("σₙ,γ [barn]")
    ax.set_title("Resonance region (E ≤ 12 MeV)")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n_gamma_resonance_region.png", bbox_inches="tight")
    plt.close(fig)


def plot_isotope_full_range(isotope: str, energy: np.ndarray, sigma: np.ndarray) -> None:
    PER_ISOTOPE_FULL_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(isotope)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(energy, _clip_positive(sigma), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(energy[0], energy[-1])
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("σₙ,γ [barn]")
    ax.set_title(f"{isotope} n,γ {energy[0]:.3g}-{energy[-1]:.3g} MeV")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PER_ISOTOPE_FULL_DIR / f"{safe}_full_range.png", bbox_inches="tight")
    plt.close(fig)


def plot_isotope_resonance(isotope: str, energy: np.ndarray, sigma: np.ndarray) -> None:
    mask = energy <= 12
    if not np.any(mask):
        return
    PER_ISOTOPE_RESONANCE_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(isotope)
    energy_masked = energy[mask]
    sigma_masked = sigma[mask]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(energy_masked, _clip_positive(sigma_masked), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(energy_masked[0], energy_masked[-1])
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("σₙ,γ [barn]")
    ax.set_title(f"{isotope} resonance window (≤12 MeV)")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PER_ISOTOPE_RESONANCE_DIR / f"{safe}_resonance_region.png", bbox_inches="tight")
    plt.close(fig)


def plot_endf_comparison(
    isotope: str,
    talys_energy: np.ndarray,
    talys_sigma: np.ndarray,
    endf_energy: np.ndarray,
    endf_sigma: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot TALYS σₙ,γ data versus ENDF/B-VIII.1 over their overlapping range."""
    start = max(talys_energy.min(), endf_energy.min())
    end = min(talys_energy.max(), endf_energy.max())
    mask_talys = (talys_energy >= start) & (talys_energy <= end)
    mask_endf = (endf_energy >= start) & (endf_energy <= end)
    if not mask_talys.any() or not mask_endf.any():
        return

    talys_clip = _clip_positive(talys_sigma[mask_talys])
    endf_clip = _clip_positive(endf_sigma[mask_endf])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        talys_energy[mask_talys],
        talys_clip,
        label="TALYS σₙ,γ",
        linewidth=1.4,
    )
    ax.plot(
        endf_energy[mask_endf],
        endf_clip,
        label="ENDF/B-VIII.1",
        linewidth=1.2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(start, end)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("σₙ,γ [barn]")
    ax.set_title(f"{isotope}: TALYS vs ENDF resonance overlap")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    safe = _safe_filename(isotope)
    fig.savefig(output_dir / f"{safe}_talys_vs_endf.png", bbox_inches="tight")
    plt.close(fig)


def plot_endf_only(
    isotope: str,
    endf_energy: np.ndarray,
    endf_sigma: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a standalone ENDF/B-VIII.1 n,γ spectrum for reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(isotope)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        endf_energy,
        _clip_positive(endf_sigma),
        label="ENDF/B-VIII.1",
        linewidth=1.3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("σₙ,γ [barn]")
    ax.set_title(f"{isotope}: ENDF/B-VIII.1 n,γ")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe}_endf.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for isotope, path in ISOTOPE_FILES.items():
        energy, sigma = load_tot_table(path)
        spectra[isotope] = (energy, sigma)

    save_spectra_csv(spectra)
    plot_general_spectrum(spectra)
    plot_resonance_region(spectra)

    for isotope, (energy, sigma) in spectra.items():
        plot_isotope_full_range(isotope, energy, sigma)
        plot_isotope_resonance(isotope, energy, sigma)

    for isotope, (energy, sigma) in spectra.items():
        peak = sigma.max()
        peak_energy = float(energy[np.argmax(sigma)])
        print(f"{isotope}: peak sigma(n,gamma) ~ {peak:.3e} barn at {peak_energy:.3f} MeV")

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    ENDF_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    for isotope, (energy, sigma) in spectra.items():
        endf_path = ENDF_FILES.get(isotope)
        if not endf_path or not endf_path.exists():
            continue
        endf_energy, endf_sigma = load_endf_ascii(endf_path)
        plot_endf_comparison(isotope, energy, sigma, endf_energy, endf_sigma, COMPARISON_DIR)
        plot_endf_only(isotope, endf_energy, endf_sigma, ENDF_ONLY_DIR)


if __name__ == "__main__":
    main()
