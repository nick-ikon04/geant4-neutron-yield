#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Configuration -------------------------------------------------------------
TEMPLATE_PATH="${TEMPLATE_PATH:-${SCRIPT_DIR}/template.inp}"
RUNS_ROOT="${RUNS_ROOT:-${SCRIPT_DIR}/runs}"

ISOTOPES=(
  "I129 I 129 I129"
  "Cs135 Cs 135 Cs135"
  "Am241 Am 241 Am241"
)

ENERGY_VARIANTS=(
  "energy"
)

# ---------------------------------------------------------------------------

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "Template '${TEMPLATE_PATH}' not found" >&2
  exit 1
fi

if ! command -v talys >/dev/null 2>&1; then
  echo "talys executable not found in PATH" >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}"
OUTPUT_SIGMA="${OUTPUT_ROOT}/sigma"
OUTPUT_RESIDUAL="${OUTPUT_ROOT}/residual"
OUTPUT_OUT="${OUTPUT_ROOT}/out"

mkdir -p "${OUTPUT_SIGMA}" "${OUTPUT_RESIDUAL}" "${OUTPUT_OUT}"

create_input() {
  local template_path="$1"
  local element="$2"
  local mass="$3"
  local tag="$4"
  local variant="$5"
  local output_path="$6"
  local out_basename="$7"

  python3 - "$template_path" "$element" "$mass" "$tag" "$variant" "$output_path" "$out_basename" <<'PYCODE'
import sys
import math
from pathlib import Path

tpl_path, element, mass, tag, variant, output_path, out_basename = sys.argv[1:8]

text = Path(tpl_path).read_text(encoding="utf-8")
text = text.replace("__ELEMENT__", element)
text = text.replace("__MASS__", mass)
text = text.replace("__TAG__", tag)

lines = text.splitlines()
processed = []


def ensure_directive(lines, directive, replacement):
    target = directive.lower()
    updated = []
    inserted = False
    for entry in lines:
        stripped = entry.strip()
        if stripped.lower().startswith(target) and not stripped.startswith("#"):
            if not inserted:
                updated.append(replacement)
                inserted = True
            continue
        updated.append(entry)
    if not inserted:
        updated.append(replacement)
    return updated

VARIANT_CONFIG = {
    "energy": {"mode": "log", "emin": 1.0e-5, "emax": 20.0, "count": 400},
}

config = VARIANT_CONFIG.get(variant, VARIANT_CONFIG["energy"])


def generate_grid(cfg):
    count = max(int(cfg.get("count", 2)), 2)
    emin = float(cfg.get("emin", 1.0e-5))
    emax = float(cfg.get("emax", 20.0))
    mode = cfg.get("mode", "log")

    if count <= 1 or math.isclose(emin, emax):
        return [emin, emax]

    if mode == "lin":
        step = (emax - emin) / (count - 1)
        return [emin + step * idx for idx in range(count)]

    start = math.log10(emin)
    stop = math.log10(emax)
    return [10 ** (start + (stop - start) * idx / (count - 1)) for idx in range(count)]


energy_written = False

SELECTED_CHANNEL_COMMENT = (
    "# channels filtered downstream by extract_talys_outputs.py"
)
OUTFILENAME_COMMENT = "# outfilename disabled (not supported by TALYS version)"
OUTPUT_COMMENT = "# output selection handled downstream"
PRINTLEVEL_COMMENT = "# printlevel disabled (not supported by TALYS version)"

for line in lines:
    normalized = line.lstrip("#").strip().lower()
    if normalized.startswith("energy"):
        if not energy_written:
            processed.append("energy energies")
            processed.append(
                f"# energy grid generated automatically ({variant}): {config['mode']} {config['emin']} {config['emax']} {config['count']}"
            )
            energy_written = True
        else:
            cleaned = line.lstrip("#").strip()
            processed.append(f"# {cleaned}")
        continue
    if normalized.startswith("outfilename"):
        tokens = normalized.split()
        if len(tokens) >= 2:
            processed.append(OUTFILENAME_COMMENT)
            continue
    if normalized.startswith("output"):
        tokens = normalized.split()
        if len(tokens) >= 2:
            processed.append(OUTPUT_COMMENT)
            continue
    if normalized.startswith("printlevel"):
        tokens = normalized.split()
        if len(tokens) >= 2:
            processed.append(PRINTLEVEL_COMMENT)
            continue
    if normalized.startswith("channels"):
        tokens = normalized.split()
        if len(tokens) > 2:
            processed.append(SELECTED_CHANNEL_COMMENT)
            continue
    processed.append(line)

if not energy_written:
    processed.append("energy energies")
    processed.append(
        f"# energy grid generated automatically ({variant}): {config['mode']} {config['emin']} {config['emax']} {config['count']}"
    )

processed = ensure_directive(processed, "resonance", "resonance y")
processed = ensure_directive(processed, "channels", "channels y")
processed = ensure_directive(processed, "filechannels", "filechannels y")

Path(output_path).write_text("\n".join(processed) + "\n", encoding="utf-8")

energy_path = Path(output_path).with_name("energies")
values = generate_grid(config)
with energy_path.open("w", encoding="utf-8") as handle:
    for value in values:
        handle.write(f"  {value: .6E}\n")
PYCODE
}

for entry in "${ISOTOPES[@]}"; do
  read -r tag element mass target_dir <<<"${entry}"
  if [[ -z "${tag}" || -z "${element}" || -z "${mass}" || -z "${target_dir}" ]]; then
    echo "Malformed isotope entry '${entry}'" >&2
    exit 1
  fi

  run_base="${RUNS_ROOT}/${target_dir}"
  mkdir -p "${run_base}"

  for variant in "${ENERGY_VARIANTS[@]}"; do
    variant_dir="${run_base}/${variant}"
    if [[ -e "${variant_dir}" ]]; then
      echo "Run directory already exists: ${variant_dir}" >&2
      echo "Move or remove it explicitly before starting a fresh TALYS run." >&2
      exit 1
    fi
    mkdir -p "${variant_dir}"

    input_name="${tag}_${variant}.inp"
    input_path="${variant_dir}/${input_name}"
    out_basename="${tag}_${variant}"

    create_input "${TEMPLATE_PATH}" "${element}" "${mass}" "${tag}" "${variant}" "${input_path}" "${out_basename}"

    (
      cd "${variant_dir}"
      talys <"${input_name}" >"${out_basename}.out" 2>&1
    )
  done
done
