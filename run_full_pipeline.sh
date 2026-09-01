#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/out}"
NPRIMARY="${NPRIMARY:-10000}"

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

python3 "$PROJECT_ROOT/run_scan.py" --out "$RESULTS_DIR" --nprimary "$NPRIMARY"
python3 "$PROJECT_ROOT/analysis/analyze_yield.py" --in "$RESULTS_DIR" --out "$OUTPUT_DIR"
