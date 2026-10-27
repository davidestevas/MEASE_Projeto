#!/usr/bin/env bash
set -e

PYTHON_BIN="/opt/anaconda3/bin/python"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Cleaning temporary files ==="
rm -f "$BASE_DIR/data/processed/temp"

echo "=== 1/7 Running process_flame.py (Axis measurement + Flame Tomo) ==="
"$PYTHON_BIN" "$BASE_DIR/src/process_flame.py"

echo "=== 2/7 Running process_iso.py (ISO Tomo with same axis shift) ==="
"$PYTHON_BIN" "$BASE_DIR/src/process_iso.py"

echo "=== 3/7 Running analysis_uncertainty.py (Noise debiasing, ACF N_eff & Uncertainty Budget) ==="
"$PYTHON_BIN" "$BASE_DIR/src/analysis_uncertainty.py"

echo "=== 4/7 Running visualize.py (Flame 2x2 maps) ==="
"$PYTHON_BIN" "$BASE_DIR/src/visualize.py"

echo "=== 5/7 Running visualize_iso.py (ISO 2x2 maps on Flame scale) ==="
"$PYTHON_BIN" "$BASE_DIR/src/visualize_iso.py"

echo "=== 6/7 Running analysis_flame_front.py (Band center + Outer 50% RMS boundary) ==="
"$PYTHON_BIN" "$BASE_DIR/src/analysis_flame_front.py"

echo "=== 7/7 Running analysis_clipping_bias.py (Zero-clipping bias with shifted axis) ==="
"$PYTHON_BIN" "$BASE_DIR/src/analysis_clipping_bias.py"

echo "=== ALL SCRIPTS COMPLETED SUCCESSFULLY! ==="
