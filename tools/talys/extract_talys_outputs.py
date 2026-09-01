#!/usr/bin/env python3

"""Aggregate TALYS outputs into CSV reports.

The script scans subdirectories for TALYS default output files (`sigma`,
`residual`) and assembles:

* A wide `sigma` CSV with every reaction channel as a column.
* A residual summary CSV that aggregates production cross sections by the
  residual nucleus (and optional reaction channel) across the incident energy
  grid.

By default the processed artefacts are organised under an `output` directory
split into `output/sigma`, `output/residual`, and `output/out`.

Run `python extract_talys_outputs.py --help` for usage details.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class FileContext:
    isotope: str
    variant: str
    path: Path


@dataclass
class RunContext:
    isotope: str
    variant: str
    run_path: Path
    sigma_dir: Path
    residual_dir: Path
    log_dir: Path


SIGMA_HEADER_KEYWORDS = {"energy"}
RESIDUAL_HEADER_KEYWORDS = {"energy", "residual"}

SIGMA_SELECTED_CHANNELS = [
    ("total_mb", ("total", "sigma_total")),
    ("elastic_mb", ("elastic", "sigma_elastic")),
    ("nonelastic_mb", ("nonelastic", "non_elastic", "reaction")),
    ("compound_elastic_mb", ("compound_elast",)),
    ("shape_elastic_mb", ("shape_elastic",)),
    ("compound_nonelastic_mb", ("compound_nonel",)),
    ("direct_mb", ("direct", "direct_reaction")),
    ("preequilibrium_mb", ("preequilibrium", "pre_equilibrium")),
    ("capture_mb", ("capture", "radiative_capture", "gamma_capture", "n_gamma", "direct_capture")),
    ("fission_mb", ("fission", "total_fission")),
    ("n2n_mb", ("n2n", "n_2n", "two_neutron")),
    ("n3n_mb", ("n3n", "n_3n", "three_neutron")),
    ("np_mb", ("np", "n_p", "proton_emission")),
    ("nd_mb", ("nd", "n_d", "deuteron_emission")),
    ("nt_mb", ("nt", "n_t", "triton_emission")),
    ("nhe3_mb", ("n_3he", "nhe3", "helion_emission")),
    ("na_mb", ("n_alpha", "na", "alpha_emission")),
    ("gamma_prod_mb", ("gamma", "gamma_prod", "gamma_production", "gprod")),
    ("proton_prod_mb", ("proton", "pprod", "proton_production")),
    ("deuteron_prod_mb", ("deuteron", "dprod", "deuteron_production")),
    ("triton_prod_mb", ("triton", "tprod", "triton_production")),
    ("helion_prod_mb", ("helion", "hprod", "helion_production")),
    ("alpha_prod_mb", ("alpha", "aprod", "alpha_production")),
]


def normalize_header_token(token: str) -> str:
    token = token.strip()
    token = token.replace("(mb)", "")
    token = token.replace("(mev)", "")
    token = token.replace("[mb]", "")
    token = token.replace("[mev]", "")
    token = re.sub(r"[^0-9a-zA-Z]+", "_", token)
    token = token.strip("_")
    return token.lower()


def parse_yandf_table(
    path: Path,
) -> Tuple[List[str], List[List[str]], Dict[str, str], List[str], Optional[str], Optional[str]]:
    columns: List[str] = []
    rows: List[List[str]] = []
    metadata: Dict[str, str] = {}
    units: List[str] = []
    header_line: Optional[str] = None
    first_data_line: Optional[str] = None
    current_section: Optional[str] = None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("##"):
                clean = stripped.lstrip("#").strip()
                if not clean:
                    continue
                if "[" in clean:
                    units = clean.split()
                    continue
                columns = clean.split()
                header_line = raw_line[2:].rstrip("\n")
            elif stripped.startswith("#"):
                clean = stripped.lstrip("#").strip()
                if not clean:
                    continue
                if clean.endswith(":"):
                    current_section = clean[:-1].lower()
                    continue
                if ":" in clean and current_section:
                    key, value = [part.strip() for part in clean.split(":", 1)]
                    metadata[f"{current_section}.{key.lower()}"] = value
                else:
                    current_section = None
            else:
                rows.append(stripped.split())
                if first_data_line is None:
                    first_data_line = raw_line.rstrip("\n")

    return columns, rows, metadata, units, header_line, first_data_line


def parse_table(path: Path, required_keywords: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", "%", "!", "@")):
                stripped = stripped.lstrip("#%!@ ")
                if not stripped:
                    continue
            tokens = stripped.split()
            if not tokens:
                continue
            lowered = {t.lower() for t in tokens}
            if header is None and all(any(keyword in lower for lower in lowered) for keyword in required_keywords):
                header = [normalize_header_token(token) for token in tokens]
                continue
            if header is None:
                continue
            if len(tokens) < len(header):
                continue
            rows.append(tokens[: len(header)])
    if header is None:
        raise ValueError(f"Failed to locate header in {path}")
    return header, rows


def derive_column_labels(
    fallback: Sequence[str],
    header_line: Optional[str],
    data_line: Optional[str],
    column_count: int,
) -> List[str]:
    if data_line:
        spans = [match.span() for match in re.finditer(r"\S+", data_line)]
        if spans:
            spans = spans[:column_count]
            labels: List[str] = []
            header_reference = header_line or ""
            if len(header_reference) < len(data_line):
                header_reference = header_reference.ljust(len(data_line))
            for index, (start, end) in enumerate(spans):
                raw_segment = header_reference[start:end]
                segment = raw_segment.strip()
                if (not segment or segment == raw_segment.strip()) and header_reference:
                    left = start
                    while left > 0 and header_reference[left - 1] not in {" ", "\t"}:
                        left -= 1
                    if left < start:
                        segment = header_reference[left:end].strip()
                if not segment and index < len(fallback):
                    segment = fallback[index]
                if not segment:
                    segment = f"col_{index}"
                labels.append(segment)
            if len(labels) == len(spans):
                return labels

    return list(fallback[:column_count])


def find_sigma_files(root: Path) -> Iterable[FileContext]:
    for sigma_path in sorted(root.glob("**/sigma")):
        variant = sigma_path.parent.name
        isotope = sigma_path.parent.parent.name if sigma_path.parent.parent != root else variant
        yield FileContext(isotope=isotope, variant=variant, path=sigma_path)


def find_residual_files(root: Path) -> Iterable[FileContext]:
    for residual_path in sorted(root.glob("**/residual")):
        variant = residual_path.parent.name
        isotope = residual_path.parent.parent.name if residual_path.parent.parent != root else variant
        yield FileContext(isotope=isotope, variant=variant, path=residual_path)


def parse_sigma_file(ctx: FileContext) -> Tuple[List[str], List[Dict[str, float]]]:
    header, rows = parse_table(ctx.path, SIGMA_HEADER_KEYWORDS)
    numeric_rows: List[Dict[str, float]] = []
    for parts in rows:
        record: Dict[str, float] = {}
        for key, value in zip(header, parts):
            try:
                record[key] = float(value)
            except ValueError:
                record[key] = float("nan")
        numeric_rows.append(record)
    return header, numeric_rows


def parse_residual_file(ctx: FileContext) -> Tuple[List[str], List[Dict[str, object]]]:
    header, rows = parse_table(ctx.path, RESIDUAL_HEADER_KEYWORDS)
    numeric_rows: List[Dict[str, object]] = []
    for parts in rows:
        record: Dict[str, object] = {}
        for key, value in zip(header, parts):
            normalized_key = key
            try:
                record[normalized_key] = float(value)
            except ValueError:
                record[normalized_key] = value
        numeric_rows.append(record)
    return header, numeric_rows


def iter_run_contexts(
    runs_root: Path,
    sigma_root: Path,
    residual_root: Path,
    log_root: Path,
) -> Iterable[RunContext]:
    if not runs_root.exists():
        return
    for isotope_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        isotope = isotope_dir.name
        for variant_dir in sorted(p for p in isotope_dir.iterdir() if p.is_dir()):
            if not any(variant_dir.glob("*.inp")) and not (variant_dir / "all.tot").exists():
                continue
            variant = variant_dir.name
            yield RunContext(
                isotope=isotope,
                variant=variant,
                run_path=variant_dir,
                sigma_dir=sigma_root / isotope / variant,
                residual_dir=residual_root / isotope / variant,
                log_dir=log_root / isotope / variant,
            )


def generate_sigma_file(ctx: RunContext) -> bool:
    source = ctx.run_path / "all.tot"
    if not source.exists():
        print(f"Skipping sigma for {ctx.isotope}/{ctx.variant}: '{source.name}' not found")
        return False

    columns, rows, metadata, units, header_line, first_data_line = parse_yandf_table(source)
    if not columns or not rows:
        print(f"Skipping sigma for {ctx.isotope}/{ctx.variant}: '{source.name}' is empty")
        return False

    try:
        column_count = int(float(metadata.get("datablock.columns", str(len(columns)))))
    except ValueError:
        column_count = len(columns)

    if units and len(units) == column_count:
        column_units = units
    else:
        column_units = [""] * column_count

    if column_count and len(columns) > column_count:
        columns = columns[:column_count]

    column_labels = derive_column_labels(columns, header_line, first_data_line, column_count)

    header: List[str] = []
    for index, token in enumerate(column_labels):
        normalized = normalize_header_token(token)
        if index == 0:
            header.append("energy")
        elif normalized:
            header.append(normalized)
        else:
            header.append(f"col_{index}")

    # Append unit suffixes when available to disambiguate columns
    for idx, unit in enumerate(column_units[: len(header)]):
        if not unit or idx >= len(header):
            continue
        normalized_unit = normalize_header_token(unit).replace("_", "")
        if normalized_unit:
            header[idx] = f"{header[idx]}_{normalized_unit}"

    output_path = ctx.sigma_dir / "sigma"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# sigma table generated from {source.name}\n")
        handle.write(" ".join(header) + "\n")
        for row in rows:
            handle.write(" ".join(row[: len(header)]) + "\n")

    return True


def generate_residual_file(ctx: RunContext) -> bool:
    entries: List[Tuple[str, str, str, str, str, str, str]] = []
    residual_files = sorted(p for p in ctx.run_path.glob("rp*") if p.is_file())
    if not residual_files:
        print(f"No residual files found for {ctx.isotope}/{ctx.variant}")
        return False

    for residual_path in residual_files:
        columns, rows, metadata, units, header_line, first_data_line = parse_yandf_table(residual_path)
        if not rows:
            continue

        try:
            column_count = int(float(metadata.get("datablock.columns", str(len(columns)))))
        except ValueError:
            column_count = len(columns)

        column_labels = derive_column_labels(columns, header_line, first_data_line, column_count)
        normalized_columns = [normalize_header_token(token) for token in column_labels]
        if not normalized_columns:
            continue

        try:
            energy_index = 0
            sigma_index = next(
                idx
                for idx, name in enumerate(normalized_columns)
                if name in {"sigma", "xs", "cross_section"}
            )
        except StopIteration:
            continue

        residual_name = metadata.get("residual.nuclide") or residual_path.stem
        reaction_name = metadata.get("reaction.type") or residual_path.stem
        z_value = metadata.get("residual.z", "nan")
        a_value = metadata.get("residual.a", "nan")
        m_value = metadata.get("residual.level.number", metadata.get("residual.isomer", "nan"))

        for row in rows:
            if len(row) <= max(energy_index, sigma_index):
                continue
            energy_value = row[energy_index]
            sigma_value = row[sigma_index]
            entries.append((energy_value, residual_name, reaction_name, sigma_value, z_value, a_value, m_value))

    if not entries:
        print(f"No residual data extracted for {ctx.isotope}/{ctx.variant}")
        return False

    header = "# residual production cross sections compiled from TALYS outputs\n"
    column_labels = "energy residual reaction sigma z a m\n"
    rows = [
        f"{energy_value} {residual_name} {reaction_name} {sigma_value} {z_value} {a_value} {m_value}\n"
        for energy_value, residual_name, reaction_name, sigma_value, z_value, a_value, m_value in entries
    ]

    output_path = ctx.residual_dir / "residual"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(column_labels)
        for row in rows:
            handle.write(row)

    run_output_path = ctx.run_path / "residual"
    try:
        run_output_path.parent.mkdir(parents=True, exist_ok=True)
        with run_output_path.open("w", encoding="utf-8") as handle:
            handle.write(header)
            handle.write(column_labels)
            for row in rows:
                handle.write(row)
    except OSError as exc:
        print(
            f"Unable to write residual file to {run_output_path}: {exc}"
        )

    return True


def copy_run_log(ctx: RunContext) -> Optional[Path]:
    preferred = ctx.run_path / f"{ctx.isotope}_{ctx.variant}.out"
    log_source: Optional[Path] = None
    if preferred.exists():
        log_source = preferred
    else:
        candidates = list(ctx.run_path.glob("*.out")) or list(ctx.run_path.glob("talys*.log"))
        if candidates:
            log_source = candidates[0]

    if log_source is None:
        print(f"No log file found for {ctx.isotope}/{ctx.variant}")
        return None

    destination = ctx.log_dir / f"{ctx.isotope}_{ctx.variant}.out"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_source, destination)
    return destination


def materialise_outputs(
    runs_root: Path,
    sigma_root: Path,
    residual_root: Path,
    log_root: Path,
) -> None:
    generated_any = False
    for ctx in iter_run_contexts(runs_root, sigma_root, residual_root, log_root):
        for directory in (ctx.sigma_dir, ctx.residual_dir, ctx.log_dir):
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)

        sigma_built = generate_sigma_file(ctx)
        sigma_selected_built = False
        if sigma_built:
            sigma_path = ctx.sigma_dir / "sigma"
            if sigma_path.exists():
                sigma_selected_built = write_sigma_selected(ctx, sigma_path) is not None

        residual_built = generate_residual_file(ctx)
        residual_summary_built = False
        if residual_built:
            residual_path = ctx.residual_dir / "residual"
            if residual_path.exists():
                residual_summary_built = write_residual_summary(ctx, residual_path) is not None

        log_copied = copy_run_log(ctx)

        if (
            sigma_built
            or residual_built
            or sigma_selected_built
            or residual_summary_built
            or log_copied is not None
        ):
            generated_any = True

    if not generated_any:
        print("No outputs materialised; nothing to process")


def select_sigma_columns(header: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    used: set[str] = set()
    normalized_header = [normalize_header_token(token) for token in header]
    header_lookup = {name: name for name in normalized_header}

    for label, candidates in SIGMA_SELECTED_CHANNELS:
        for candidate in candidates:
            normalized_candidate = normalize_header_token(candidate)
            if (
                normalized_candidate in header_lookup
                and header_lookup[normalized_candidate] not in used
            ):
                mapping[label] = header_lookup[normalized_candidate]
                used.add(header_lookup[normalized_candidate])
                break

            stripped_candidate = normalized_candidate.replace("_", "")
            for column in normalized_header:
                if column in used:
                    continue
                if column == normalized_candidate or column.replace("_", "") == stripped_candidate:
                    mapping[label] = column
                    used.add(column)
                    break
            if label in mapping:
                break

    return mapping


def write_sigma_selected(ctx: RunContext, sigma_path: Path) -> Optional[Path]:
    file_ctx = FileContext(isotope=ctx.isotope, variant=ctx.variant, path=sigma_path)
    try:
        header, rows = parse_sigma_file(file_ctx)
    except ValueError as exc:
        print(f"Skipping sigma_selected for {ctx.isotope}/{ctx.variant}: {exc}")
        return None

    energy_field = select_field(header, "energy", "energy_mev", "mev", "e")
    if energy_field is None:
        energy_field = header[0]

    selected_columns = select_sigma_columns(header)
    if not selected_columns:
        print(f"No recognised sigma channels for {ctx.isotope}/{ctx.variant}; skipping sigma_selected")
        return None

    ordered_labels = [label for label, _ in SIGMA_SELECTED_CHANNELS if label in selected_columns]
    fieldnames = ["isotope", "variant", "energy_mev", *ordered_labels]

    output_path = ctx.sigma_dir / f"{ctx.isotope}_{ctx.variant}_sigma_selected.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record: Dict[str, object] = {
                "isotope": ctx.isotope,
                "variant": ctx.variant,
                "energy_mev": row.get(energy_field),
            }
            for label in ordered_labels:
                source_field = selected_columns[label]
                record[label] = row.get(source_field)
            writer.writerow(record)

    return output_path


def compute_residual_summary_rows(
    isotope: str,
    variant: str,
    header: Sequence[str],
    rows: Sequence[Dict[str, object]],
    source: Optional[str] = None,
) -> List[Dict[str, object]]:
    if not rows:
        return []

    sigma_field = select_field(header, "sigma", "xs", "cross_section", "production", "cs")
    if sigma_field is None:
        if source:
            print(f"Skipping {source}: unable to identify sigma column")
        return []

    energy_field = select_field(header, "energy", "energy_mev", "einc")
    residual_field = select_field(header, "residual", "nuclide", "product")
    reaction_field = select_field(header, "reaction", "channel", "process")
    z_field = select_field(header, "z", "zres")
    a_field = select_field(header, "a", "ares")
    m_field = select_field(header, "m", "isomer")

    accumulator: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        raw_sigma = row.get(sigma_field)
        if raw_sigma is None:
            continue
        try:
            sigma = float(raw_sigma)
        except (TypeError, ValueError):
            continue

        energy = float("nan")
        if energy_field:
            raw_energy = row.get(energy_field)
            try:
                energy = float(raw_energy)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                energy = float("nan")

        residual_name = ""
        if residual_field:
            value = row.get(residual_field)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                residual_name = str(value)
        else:
            parts = []
            if a_field:
                a_value = row.get(a_field)
                try:
                    parts.append(str(int(float(a_value))))
                except (TypeError, ValueError):
                    pass
            if z_field:
                z_value = row.get(z_field)
                try:
                    parts.append(f"Z{int(float(z_value))}")
                except (TypeError, ValueError):
                    pass
            if m_field:
                m_value = row.get(m_field)
                try:
                    m_int = int(float(m_value))
                except (TypeError, ValueError):
                    m_int = 0
                if m_int:
                    parts.append(f"m{m_int}")
            residual_name = "/".join(parts) if parts else "unknown"

        reaction_name = ""
        if reaction_field:
            reaction_value = row.get(reaction_field)
            if reaction_value is not None and not (
                isinstance(reaction_value, float) and math.isnan(reaction_value)
            ):
                reaction_name = str(reaction_value)

        accumulator[(residual_name, reaction_name)].append((energy, sigma))

    summary_rows: List[Dict[str, object]] = []
    for (residual_name, reaction_name), samples in accumulator.items():
        sigmas = [sigma for _, sigma in samples if sigma == sigma]
        energies = [energy for energy, sigma in samples if energy == energy and sigma == sigma]
        if not sigmas:
            continue
        sum_sigma = float(sum(sigmas))
        max_sigma = max(sigmas)
        energy_at_max = energies[sigmas.index(max_sigma)] if energies else float("nan")
        mean_sigma = statistics.fmean(sigmas) if len(sigmas) > 1 else sigmas[0]
        summary_rows.append(
            {
                "isotope": isotope,
                "variant": variant,
                "residual": residual_name,
                "reaction": reaction_name or "",
                "sigma_sum_mb": sum_sigma,
                "sigma_mean_mb": mean_sigma,
                "sigma_max_mb": max_sigma,
                "energy_at_sigma_max_mev": energy_at_max,
                "samples": len(sigmas),
            }
        )

    return summary_rows


def write_residual_summary(ctx: RunContext, residual_path: Path) -> Optional[Path]:
    file_ctx = FileContext(isotope=ctx.isotope, variant=ctx.variant, path=residual_path)
    try:
        header, rows = parse_residual_file(file_ctx)
    except ValueError as exc:
        print(f"Skipping residual summary for {ctx.isotope}/{ctx.variant}: {exc}")
        return None

    summary_rows = compute_residual_summary_rows(
        ctx.isotope,
        ctx.variant,
        header,
        rows,
        source=f"{ctx.isotope}/{ctx.variant}",
    )
    if not summary_rows:
        return None

    output_path = ctx.residual_dir / f"{ctx.isotope}_{ctx.variant}_residual_summary.csv"

    fieldnames = [
        "isotope",
        "variant",
        "residual",
        "reaction",
        "sigma_sum_mb",
        "sigma_mean_mb",
        "sigma_max_mb",
        "energy_at_sigma_max_mev",
        "samples",
    ]

    def write_summary(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    write_summary(output_path)

    run_output_path = ctx.run_path / f"{ctx.isotope}_{ctx.variant}_residual_summary.csv"
    try:
        write_summary(run_output_path)
    except OSError as exc:
        print(f"Unable to write residual summary to {run_output_path}: {exc}")

    return output_path

def select_field(header: Sequence[str], *candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate in header:
            return candidate
    return None


def build_sigma_report(root: Path, output: Path) -> None:
    fieldnames: List[str] = ["isotope", "variant"]
    data_rows: List[Dict[str, object]] = []

    for ctx in find_sigma_files(root):
        try:
            header, rows = parse_sigma_file(ctx)
        except ValueError as exc:
            print(exc)
            continue
        for column in header:
            if column not in fieldnames:
                fieldnames.append(column)
        for row in rows:
            enriched: Dict[str, object] = dict(row)
            enriched["isotope"] = ctx.isotope
            enriched["variant"] = ctx.variant
            data_rows.append(enriched)

    if not data_rows:
        print("No sigma files found; skipping sigma report")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)


def summarise_residuals(root: Path, output: Path) -> None:
    summary_rows: List[Dict[str, object]] = []

    for ctx in find_residual_files(root):
        try:
            header, rows = parse_residual_file(ctx)
        except ValueError as exc:
            print(exc)
            continue

        summary_rows.extend(
            compute_residual_summary_rows(
                ctx.isotope,
                ctx.variant,
                header,
                rows,
                source=str(ctx.path),
            )
        )

    if not summary_rows:
        print("No residual files found; skipping residual summary")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "isotope",
        "variant",
        "residual",
        "reaction",
        "sigma_sum_mb",
        "sigma_mean_mb",
        "sigma_max_mb",
        "energy_at_sigma_max_mev",
        "samples",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("."),
        help="Directory with raw TALYS outputs organised by isotope/variant",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("output/out"),
        help="Destination directory for run-specific outputs",
    )
    parser.add_argument(
        "--sigma-out",
        type=Path,
        default=Path("output/sigma/sigma_combined.csv"),
        help="Destination CSV for merged sigma table",
    )
    parser.add_argument(
        "--residual-out",
        type=Path,
        default=Path("output/residual/residual_summary.csv"),
        help="Destination CSV for residual summary",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    sigma_root = args.sigma_out.parent
    residual_root = args.residual_out.parent
    out_root = args.out_root

    for directory in (sigma_root, residual_root, out_root):
        directory.mkdir(parents=True, exist_ok=True)

    materialise_outputs(args.runs_root, sigma_root, residual_root, out_root)
    build_sigma_report(sigma_root, args.sigma_out)
    summarise_residuals(residual_root, args.residual_out)


if __name__ == "__main__":
    main()
