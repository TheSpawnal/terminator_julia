#!/usr/bin/env bash
# render.sh - generate all benchmark plots from DAS-5 CSV results.
# Wraps the existing visualize_benchmarks.py scripts. No new plotting logic.
# Usage:
#   ./render.sh                 # render everything into ./plots
#   ./render.sh -o /path/plots  # custom output dir
set -euo pipefail

OUT="./plots"
[ "${1:-}" = "-o" ] && { OUT="$2"; shift 2; }

JL_DIR="julia_polybench_refactored"
OMP_DIR="openmp_polybench_refactored"
JL_RES="$JL_DIR/results"
OMP_RES="$OMP_DIR/results"
VIZ_JL="$JL_DIR/visualize_benchmarks.py"
VIZ_OMP="$OMP_DIR/scripts/visualize_benchmarks.py"

export MPLBACKEND=Agg   # headless: DAS-5 nodes have no display

python3 -c "import pandas, numpy, matplotlib" 2>/dev/null \
  || { echo "missing deps. run: pip install pandas numpy matplotlib"; exit 1; }

mkdir -p "$OUT/julia" "$OUT/openmp" "$OUT/compare"

echo "[1/3] julia plots"
python3 "$VIZ_JL" "$JL_RES"/*.csv -o "$OUT/julia" -t "DAS-5 Julia"

echo "[2/3] openmp plots"
python3 "$VIZ_JL" "$OMP_RES"/*.csv -o "$OUT/openmp" -t "DAS-5 OpenMP"

echo "[3/3] cross-language compare"
# stage with language prefixes so detect_language is deterministic
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for f in "$JL_RES"/*.csv;  do cp "$f" "$STAGE/julia_$(basename "$f")";  done
for f in "$OMP_RES"/*.csv; do cp "$f" "$STAGE/openmp_$(basename "$f")"; done
python3 "$VIZ_OMP" --compare "$STAGE"/*.csv -o "$OUT/compare" -t "DAS-5"

echo "done. plots in: $OUT/{julia,openmp,compare}"
