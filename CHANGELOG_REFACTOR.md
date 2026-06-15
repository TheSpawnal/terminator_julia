# Changelog - Refactor

## Added
- **OpenMP `jacobi2d` benchmark** (`openmp_polybench_refactored/src/benchmark_jacobi2d.c`):
  2D 5-point Jacobi stencil, the common stencil kernel paired with the Julia `jacobi2d`.
  Strategies: `sequential, threads_static, threads_dynamic, tiled, simd, red_black`.
  Dataset sizes, stencil coefficient (0.2), and strategy names match the Julia runner.
  All strategies verify PASS (red-black against a dedicated red-black reference).
- **`DAS5_LAUNCH_PROTOCOL.md`**: prun-based launch protocol - one benchmark per exclusive
  node, backgrounded for concurrency, no batch queue.

## Fixed
- **Data-integrity defect (critical).** Every Julia `run_*.jl` CSV writer emitted the literal
  `"correlation"` in the `benchmark` column (copy-paste artifact), mislabeling the
  2mm/3mm/cholesky/jacobi2d/nussinov outputs as correlation. Each runner now writes its own
  kernel name. Regenerate all Julia CSVs after pulling this change; treat any pre-fix
  top-level Julia results as suspect.

## Changed
- **Strategy-name canonicalization for `jacobi2d`** (Julia side): `threads -> threads_static`,
  `redblack -> red_black`, so the Julia and OpenMP stencils align by exact name in the
  comparison plots. Internal kernel function names are unchanged; only the public strategy
  strings and the `STRATEGIES_JACOBI2D` registry were renamed.
- **Makefile**: `jacobi2d` added to `BENCHMARKS` with an explicit build target; the `run`,
  `test`, and `scaling` targets pick it up automatically.
- **`slurm/das5_run_all.slurm`** and **`das5_openmp_benchmarks_deploy.md`**: benchmark lists
  now include `jacobi2d` as part of the common set; `heat3d` documented as OpenMP-only bonus.
- **READMEs** (top-level and OpenMP) updated: common kernel set, jacobi2d/heat3d split,
  red_black strategy, and a pointer to the launch protocol.

## Notes / documented follow-ups (from the state report, not done here)
- Strategy-name canonicalization for the other five Julia kernels (correlation/cholesky use
  `threads`, `colmajor`, `blas` where OpenMP uses `threads_static`, `collapsed`, `simd`).
- Routing Julia runs through `PolyBenchJulia`/`BenchCore`/`Metrics` so confidence intervals,
  GC accounting, and allocation counts reach the CSVs (the run scripts are currently
  standalone and bypass the statistical engine).
- Dual-speedup reporting (parallel vs algorithmic) and a load-time filename-vs-column
  benchmark guard in the visualizer.
