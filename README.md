# Julia vs OpenMP - Parallelism & Multithreading

A rigorous comparative benchmarking framework that runs the same PolyBench kernels in
**Julia** and **OpenMP/C** under multiple parallelization strategies, then measures and
compares time, speedup, efficiency, and scaling. Target platform of record is the
**DAS-5** cluster (VU68: dual 8-core Xeon E5-2630-v3 Haswell, 16 physical cores, 64 GB,
FDR InfiniBand, Rocky Linux + OpenHPC, SLURM + prun).

## Layout

    .
    |- README.md                     this file
    |- DAS5_LAUNCH_PROTOCOL.md        prun CLI loops: one benchmark per node, no queueing
    |- openmp_polybench_refactored/   OpenMP/C suite (Makefile, src/, include/, slurm/, scripts/)
    |- julia_polybench_refactored/    Julia suite (src/, scripts/, slurm/)

## Common kernel set (both languages)

    2mm   3mm   cholesky   correlation   nussinov   jacobi2d

These six kernels are implemented in both languages with matching dataset sizes and a
canonical strategy vocabulary, so the central comparative question (Julia vs OpenMP across
compute-bound and memory-bound workloads) can be answered fairly.

- `jacobi2d` is the **common stencil** (2D 5-point, coefficient 0.2). The OpenMP twin was
  added in this refactor to pair with the existing Julia `jacobi2d`; both expose
  `sequential, threads_static, threads_dynamic, tiled, simd, red_black`.
- `heat3d` (3D 7-point) remains an **OpenMP-only bonus** with no Julia twin and is excluded
  from the common set.

## What changed in this refactor

1. **OpenMP `jacobi2d` added** (`src/benchmark_jacobi2d.c`) as the common stencil kernel,
   modeled on `heat3d` but 2D, matching the Julia sizes, coefficient, and strategy names.
   Wired into the `Makefile`, `slurm/das5_run_all.slurm`, and the deploy docs.
2. **Data-integrity defect fixed.** Every Julia `run_*.jl` CSV writer hardcoded
   `"correlation"` as the `benchmark` column (a copy-paste artifact), which mislabeled
   the 2mm/3mm/cholesky/jacobi2d/nussinov outputs. Each script now writes its own kernel
   name. Re-run the suite to regenerate clean CSVs; quarantine any pre-fix results.
3. **Strategy names canonicalized for `jacobi2d`** (`threads -> threads_static`,
   `redblack -> red_black`) so the Julia and OpenMP stencils line up by exact name in the
   comparison plots.
4. **Docs improved**: per-suite READMEs updated, and a focused DAS-5 launch protocol
   (`DAS5_LAUNCH_PROTOCOL.md`) built on simple `prun` loops.

## Metrics and CSV schema

Both suites emit the same schema:

    benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,
    gflops,speedup,efficiency_pct,verified,max_error,allocations

`efficiency_pct` is reported only for genuinely parallel strategies (`is_parallel=true`);
it is blank for `sequential`, `simd`, and `blas`, where parallel efficiency is meaningless.
Every strategy is verified numerically against a sequential reference (`PASS`/`FAIL` plus
`max_error`); red-black strategies verify against a dedicated red-black reference.

## Quick start (local)

OpenMP:

    cd openmp_polybench_refactored
    make                 # portable build; use 'make native' locally or 'make das5' on DAS-5
    ./benchmark_jacobi2d --dataset MEDIUM --threads 4 --output csv
    make scaling         # 1,2,4,8,16-thread sweep over the whole suite

Julia:

    cd julia_polybench_refactored
    julia -t 4 scripts/run_jacobi2d.jl --dataset MEDIUM --output csv

## DAS-5 deployment

See **`DAS5_LAUNCH_PROTOCOL.md`** for the prun-based protocol that puts one benchmark on
each exclusive node with no batch queue. The per-suite `slurm/` directories and the
OpenMP `das5_openmp_benchmarks_deploy.md` contain the sbatch-based alternatives (including
deferred off-hours dispatch for EXTRALARGE).

## Design principles

- Manual, explicit timing (warmup / GC control / state reset) for HPC predictability.
- Honest baselines: distinguish parallel scaling from algorithmic/locality wins; cache-aware
  strategies can beat a naive sequential baseline by more than the thread count, which is a
  locality effect, not super-linear parallelism.
- 16 physical cores per node; 32-thread SMT is counterproductive for compute-bound kernels.
- Pin threads (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`) and use identical hardware
  (`-C cpunode`) for every run.
