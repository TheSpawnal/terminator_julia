#!/usr/bin/env bash
# profile_flame.sh - perf-based flame graph for an OpenMP/C kernel.
#
# Rebuilds the target kernel with frame pointers + debug symbols (does not touch
# the production binaries), records with perf, and renders an SVG via Brendan
# Gregg's FlameGraph. perf must be permitted for the user on the node.
#
# Usage (from openmp_polybench_refactored/):
#   ./profile_flame.sh <kernel> [dataset=LARGE] [threads=16] [strategy=threads_static]
#   kernel in: 2mm 3mm cholesky correlation jacobi2d nussinov
#
# Output: results/flame/omp_flame_<kernel>_<dataset>_<strategy>.svg
set -euo pipefail

K="${1:?usage: ./profile_flame.sh <kernel> [dataset] [threads] [strategy]}"
DATASET="${2:-LARGE}"
THREADS="${3:-16}"
STRAT="${4:-threads_static}"

SRC="src/benchmark_${K}.c"
[ -f "$SRC" ] || { echo "no source $SRC"; exit 1; }
[ -f obj/metrics.o ] || make obj/metrics.o >/dev/null

OUT="results/flame"; mkdir -p "$OUT"
PROF_BIN="benchmark_${K}_prof"

# 1. profiling build: keep -O3 -march=haswell but add frame pointers + symbols
echo "[flame] building $PROF_BIN"
gcc -O3 -march=haswell -mavx2 -mfma -fopenmp -g -fno-omit-frame-pointer \
    -Iinclude "$SRC" obj/metrics.o -o "$PROF_BIN" -lm

# 2. locate FlameGraph toolkit (clone once on the head node if absent)
FG="${FLAMEGRAPH_DIR:-./FlameGraph}"
if [ ! -x "$FG/flamegraph.pl" ]; then
  echo "[flame] fetching FlameGraph -> $FG"
  git clone --depth 1 https://github.com/brendangregg/FlameGraph "$FG"
fi

# 3. record. dwarf call-graph handles -O3 inlining better than fp here.
DATA="$OUT/perf_${K}_${DATASET}_${STRAT}.data"
echo "[flame] perf record ($K $DATASET t=$THREADS $STRAT)"
OMP_NUM_THREADS="$THREADS" \
perf record -F 999 -g --call-graph dwarf -o "$DATA" -- \
  ./"$PROF_BIN" --dataset "$DATASET" -t "$THREADS" -s "$STRAT" >/dev/null

# 4. fold + render
SVG="$OUT/omp_flame_${K}_${DATASET}_${STRAT}.svg"
perf script -i "$DATA" | "$FG/stackcollapse-perf.pl" \
  | "$FG/flamegraph.pl" --title "OpenMP $K $DATASET $STRAT (t=$THREADS)" > "$SVG"

rm -f "$DATA" "$PROF_BIN"
echo "[flame] wrote $SVG"
