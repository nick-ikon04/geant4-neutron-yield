#!/usr/bin/env python3
"""Parameter sweep driver for the neutron yield Geant4 simulation."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_THICKNESSES_MM = {
    "electron": [2.0, 4.0, 8.0, 12.0, 16.0, 50.0, 100.0],
    "proton": [2.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0],
}
DEFAULT_RADII_MM = [20.0]
DEFAULT_SCORING_THICKNESSES_MM = [0.5]
ELECTRON_ENERGIES = [50.0, 100.0, 200.0]
PROTON_ENERGIES = [100.0, 200.0, 400.0, 800.0]
MATERIAL_OPTIONS = {"U": "natU", "W": "W"}
BEAM_PARTICLE = {"electron": "e-", "proton": "proton"}


def format_float_for_path(value: float) -> str:
    formatted = f"{value:g}"
    return formatted.replace(".", "p")


def build_output_paths(root: Path, beam: str, material: str, energy: float,
                       thickness: float, radius: float,
                       scoring_thickness: float) -> Tuple[Path, Path]:
    segment = (
        f"L{format_float_for_path(thickness)}mm_"
        f"R{format_float_for_path(radius)}mm_"
        f"S{format_float_for_path(scoring_thickness)}mm"
    )
    dir_path = (
        root
        / beam
        / material
        / f"E{format_float_for_path(energy)}MeV"
        / segment
    )
    macro_path = dir_path / "run.mac"
    base_path = dir_path / "run"
    return macro_path, base_path


def write_macro_file(path: Path, base: Path, beam: str, energy: float,
                     material_key: str, thickness: float, radius: float,
                     scoring_thickness: float, nprimary: int) -> None:
    material_cmd = MATERIAL_OPTIONS[material_key]
    particle = BEAM_PARTICLE[beam]
    base_name = base.as_posix()

    lines = [
        "/control/verbose 0",
        "/run/verbose 0",
        "/analysis/fileType csv",
        f"/analysis/fileName {base_name}",
        f"/detector/material {material_cmd}",
        f"/detector/halfLength {thickness} mm",
        f"/detector/radius {radius} mm",
        f"/detector/scoringThickness {scoring_thickness} mm",
        f"/beam/particle {particle}",
        f"/beam/energy {energy} MeV",
        f"/beam/targetHalfLength {thickness} mm",
        "/run/initialize",
        f"/run/beamOn {nprimary}",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_run_config(base: Path, metadata: dict) -> None:
    config_path = base.parent / "run_config.json"
    with config_path.open("w", encoding="utf-8") as cfg:
        json.dump(metadata, cfg, indent=2)


def run_simulation(executable: Path, macro_path: Path, cwd: Path,
                   dry_run: bool) -> int:
    cmd = [str(executable), str(macro_path)]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def collect_tasks(beams: Iterable[str]) -> List[Tuple[str, float]]:
    tasks: List[Tuple[str, float]] = []
    for beam in beams:
        energies = ELECTRON_ENERGIES if beam == "electron" else PROTON_ENERGIES
        tasks.extend((beam, energy) for energy in energies)
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter scan for neutron yield simulation.")
    parser.add_argument("--out", dest="output", type=Path, default=Path("results"),
                        help="Output directory root for simulation results (default: results)")
    parser.add_argument("--executable", type=Path, default=Path("build/neutron_yield"),
                        help="Path to the Geant4 executable (default: build/neutron_yield)")
    parser.add_argument("--nprimary", type=int, default=10000,
                        help="Number of primary particles per run (default: 10000)")
    parser.add_argument("--thickness", nargs="*", type=float,
                        help="Override target half-length grid in mm")
    parser.add_argument("--radius", nargs="*", type=float,
                        help="Override target radius grid in mm")
    parser.add_argument("--scoring-thickness", nargs="*", type=float,
                        help="Override scoring shell thickness grid in mm (default: 0.5)")
    parser.add_argument("--beam", nargs="*", choices=["electron", "proton"],
                        help="Limit scan to specified beam types")
    parser.add_argument("--material", nargs="*", choices=list(MATERIAL_OPTIONS.keys()),
                        help="Limit scan to specified target materials (U/W)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose metadata already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing Geant4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    cwd = Path.cwd()
    executable = args.executable.resolve()

    if not args.dry_run and not executable.exists():
        print(f"Executable not found: {executable}")
        return 1

    thickness_override = args.thickness
    radius_grid = args.radius if args.radius else DEFAULT_RADII_MM
    scoring_grid = (
        args.scoring_thickness if args.scoring_thickness else DEFAULT_SCORING_THICKNESSES_MM
    )
    beams = args.beam if args.beam else ["electron", "proton"]
    materials = args.material if args.material else list(MATERIAL_OPTIONS.keys())

    tasks = collect_tasks(beams)

    for beam, energy in tasks:
        thickness_grid = thickness_override if thickness_override else DEFAULT_THICKNESSES_MM[beam]
        for material in materials:
            for thickness, radius, scoring in itertools.product(
                thickness_grid, radius_grid, scoring_grid
            ):
                macro_path, base_path = build_output_paths(
                    output_root,
                    beam,
                    material,
                    energy,
                    thickness,
                    radius,
                    scoring,
                )
                metadata_path = base_path.parent / "run_metadata.json"

                if args.skip_existing and metadata_path.exists():
                    print(f"Skipping existing run: {metadata_path}")
                    continue

                write_macro_file(
                    macro_path,
                    base_path,
                    beam,
                    energy,
                    material,
                    thickness,
                    radius,
                    scoring,
                    args.nprimary,
                )

                run_metadata = {
                    "beam_type": beam,
                    "beam_energy_MeV": energy,
                    "target_material": material,
                    "target_half_length_mm": thickness,
                    "target_radius_mm": radius,
                    "scoring_shell_thickness_mm": scoring,
                    "N_primary": args.nprimary,
                    "macro": macro_path.as_posix(),
                }
                write_run_config(base_path, run_metadata)

                print(
                    f"Running beam={beam} E={energy} MeV material={material} "
                    f"L={thickness} mm R={radius} mm S={scoring} mm"
                )
                code = run_simulation(executable, macro_path, cwd, args.dry_run)
                if code != 0:
                    print(f"Simulation failed with exit code {code} for {macro_path}")
                    return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
