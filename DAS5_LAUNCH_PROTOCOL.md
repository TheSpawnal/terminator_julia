# DAS-5 Launch Protocol - sbatch, one benchmark per node, no tmux

Target cluster: `fs0.das5.cs.vu.nl` (VU68). Partition `defq`, constraint `cpunode`
(dual 8-core Xeon E5-2630-v3 Haswell, 16 physical cores, 64 GB).

Dispatch mechanism: **`sbatch --wrap`**. Each benchmark is submitted as one
independent batch job that claims a full exclusive cpunode. Submitting returns
immediately, the scheduler runs the jobs as nodes free up, and the jobs run
independently of your shell. You can submit and log out; results land in files.

Why no `tmux`, no backgrounding, no `wait`: those were only needed for the older
synchronous `prun` approach, which dies when your SSH session drops. `sbatch`
hands the job to SLURM and detaches it from your terminal entirely. There is
nothing to keep alive.

Why no copy-paste worries: each loop below is `for b in ...; do sbatch ...; done`.
Every iteration submits a job and returns instantly. Paste the whole block at once.

---

## 0. Policy (read once)

- Edit and compile on `fs0`. Never execute on `fs0` - jobs run only on compute nodes.
- Daytime cap is 15 minutes of walltime per job. LARGE fits. EXTRALARGE needs hours,
  so it is deferred to off-hours with `--begin=` (see section 5). Long off-hours jobs
  may require your supervisor's permission per DAS-5 policy.
- `--exclusive -C cpunode` gives you a whole node of identical Haswell hardware,
  mandatory for valid cross-benchmark and cross-language comparison.
- The job environment does not source your `.bashrc`/`.bash_profile`, so every wrap
  body sources `/etc/bashrc` and `/etc/profile.d/lmod.sh` and loads its modules itself.

---

## 1. The common benchmark set

After the refactor, both languages share six identical kernels:

    2mm  3mm  cholesky  correlation  nussinov  jacobi2d

`jacobi2d` is the common stencil (2D 5-point, coefficient 0.2, canonical strategies
`sequential, threads_static, threads_dynamic, tiled, simd, red_black`). `heat3d`
remains an OpenMP-only bonus and is not part of the common set.

---

## 2. Setup and preflight (on fs0, once per session)

Set the roots. The doubled `terminator_julia` is intentional (the suites live one
level inside the cloned tree).

    export OMP_DIR="$HOME/terminator_julia/terminator_julia/openmp_polybench_refactored"
    export JL_DIR="$HOME/terminator_julia/terminator_julia/julia_polybench_refactored"
    module load prun        # makes sbatch/squeue/sinfo available

Build the OpenMP binaries (Julia needs no build):

    cd "$OMP_DIR"
    make clean && make das5          # -march=haswell -mavx2 -mfma (DAS-5 safe, NOT native)
    ls -lh benchmark_{2mm,3mm,cholesky,correlation,nussinov,jacobi2d,heat3d}

Confirm the Julia module resolves:

    module load julia/1.11.4 2>/dev/null || module load julia
    julia --version

Check how many cpunodes are idle right now:

    sinfo -p defq -t idle -o "%N %t %f" | grep cpunode

Each loop below uses the six common kernels:

    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

---

## 3. OpenMP - LARGE, thread scaling 4/8/16

Six jobs, one benchmark each, each on its own exclusive node, each sweeping threads
`4 8 16` in an inner loop so you get a scaling curve per benchmark in one job.

    OMP_DIR="$HOME/terminator_julia/terminator_julia/openmp_polybench_refactored"
    mkdir -p "$OMP_DIR/results"
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

    for b in $BENCHES; do
        sbatch --job-name="scale_omp_${b}" \
               --output="${OMP_DIR}/results/scale_omp_${b}_%j.out" \
               --error="${OMP_DIR}/results/scale_omp_${b}_%j.err" \
               --time=00:15:00 \
               -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
               --partition=defq -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                       module load prun; \
                       export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false; \
                       cd ${OMP_DIR}; \
                       for t in 4 8 16; do \
                           echo \"[\$(hostname)] ${b} LARGE t=\$t\"; \
                           OMP_NUM_THREADS=\$t ./benchmark_${b} --dataset LARGE \
                               --threads \$t --iterations 10 --warmup 3 --output csv; \
                       done"
    done
    squeue -u $USER

What expands when:
- `${OMP_DIR}`, `${b}` expand on fs0 at submit time (the loop variable). Intended.
- `\$t`, `\$(hostname)` are escaped, so they expand on the compute node when the job
  runs. The log then records which node ran which benchmark at which thread count.

---

## 4. Julia - LARGE, thread scaling 4/8/16

Same shape. The Julia thread count comes from `-t`/`JULIA_NUM_THREADS` (read via
`Threads.nthreads()`); there is no `--threads` flag and no `--warmup` flag (warmup is
fixed at 5 inside the script). BLAS is pinned to 1 thread so it does not contend with
the kernel threads.

    JL_DIR="$HOME/terminator_julia/terminator_julia/julia_polybench_refactored"
    mkdir -p "$JL_DIR/results"
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

    for b in $BENCHES; do
        sbatch --job-name="scale_jl_${b}" \
               --output="${JL_DIR}/results/scale_jl_${b}_%j.out" \
               --error="${JL_DIR}/results/scale_jl_${b}_%j.err" \
               --time=00:15:00 \
               -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
               --partition=defq -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                       module load prun julia/1.11.4; \
                       export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
                       cd ${JL_DIR}; \
                       for t in 4 8 16; do \
                           echo \"[\$(hostname)] julia ${b} LARGE t=\$t\"; \
                           JULIA_NUM_THREADS=\$t julia -t \$t scripts/run_${b}.jl \
                               --dataset LARGE --iterations 10 --output csv; \
                       done"
    done
    squeue -u $USER

---

## 5. EXTRALARGE - both languages, deferred to off-hours

EXTRALARGE exceeds the 15-min daytime cap, so it must start off-hours. `--begin`
defers the start; the job sits in the queue (state `PD`, reason `(BeginTime)`) until
then, then runs unattended. No tmux, no open session required. Deferral options:

    --begin=22:00              # tonight at 22:00
    --begin=saturday           # next Saturday 00:00
    --begin=now+6hours         # six hours from submit
    --begin=2026-06-20T18:00   # explicit timestamp

`--time=04:00:00` covers the three thread points 4/8/16 including the slower t=4 runs
of the heaviest kernels (nussinov, cholesky). The node is released the moment its
sweep finishes; `--time` is only the ceiling.

OpenMP EXTRALARGE:

    OMP_DIR="$HOME/terminator_julia/terminator_julia/openmp_polybench_refactored"
    mkdir -p "$OMP_DIR/results"
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

    for b in $BENCHES; do
        sbatch --begin=05:22 \
               --job-name="scaleXL_omp_${b}" \
               --output="${OMP_DIR}/results/scaleXL_omp_${b}_%j.out" \
               --error="${OMP_DIR}/results/scaleXL_omp_${b}_%j.err" \
               --time=04:00:00 \
               -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
               --partition=defq -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                       module load prun; \
                       export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false; \
                       cd ${OMP_DIR}; \
                       for t in 2 4 8 16; do \
                           echo \"[\$(hostname)] ${b} EXTRALARGE t=\$t\"; \
                           OMP_NUM_THREADS=\$t ./benchmark_${b} --dataset EXTRALARGE \
                               --threads \$t --iterations 5 --warmup 2 --output csv; \
                       done"
    done
    squeue -u $USER

Julia EXTRALARGE (iterations trimmed to 3; the script's fixed 5 warmups are heavy at XL):

    JL_DIR="$HOME/terminator_julia/terminator_julia/julia_polybench_refactored"
    mkdir -p "$JL_DIR/results"
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

    for b in $BENCHES; do
        sbatch --begin=05:22 \
               --job-name="scaleXL_jl_${b}" \
               --output="${JL_DIR}/results/scaleXL_jl_${b}_%j.out" \
               --error="${JL_DIR}/results/scaleXL_jl_${b}_%j.err" \
               --time=04:00:00 \
               -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
               --partition=defq -C cpunode \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                       module load prun julia/1.11.4; \
                       export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
                       cd ${JL_DIR}; \
                       for t in 2 4 8 16; do \
                           echo \"[\$(hostname)] julia ${b} EXTRALARGE t=\$t\"; \
                           JULIA_NUM_THREADS=\$t julia -t \$t scripts/run_${b}.jl \
                               --dataset EXTRALARGE --iterations 3 --output csv; \
                       done"
    done
    squeue -u $USER

---

## 6. Single-benchmark smoke test (MEDIUM, 16 threads)

Before committing a full sweep, validate one kernel end to end in well under a minute.

    OMP_DIR="$HOME/terminator_julia/terminator_julia/openmp_polybench_refactored"
    sbatch --job-name="smoke_omp_2mm" \
           --output="${OMP_DIR}/results/smoke_omp_2mm_%j.out" \
           --error="${OMP_DIR}/results/smoke_omp_2mm_%j.err" \
           --time=00:05:00 \
           -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
           --partition=defq -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun; \
                   cd ${OMP_DIR}; \
                   OMP_NUM_THREADS=16 ./benchmark_2mm --dataset MEDIUM \
                       --threads 16 --iterations 3 --warmup 1 --output csv"
    squeue -u $USER

---

## 7. Skip known-broken strategies

Both the C binaries and the Julia runners accept `--strategies` (comma-separated). To
avoid wasting a long EXTRALARGE reservation on a strategy with a known failure (e.g.
cholesky `tiled`/`threads_static` historically FAILed), pass only verified ones:

    ./benchmark_cholesky --dataset EXTRALARGE --threads 16 \
        --iterations 5 --warmup 2 --output csv \
        --strategies sequential,threads_dynamic,simd

Drop that `--strategies ...` into the wrap body for that one benchmark.

---

## 8. Monitor, inspect, collect, visualize

Watch the queue (LARGE jobs show `R` running / `PD (Resources)` queued; XL jobs show
`PD (BeginTime)` until their `--begin` time):

    squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %.20R"

Live-tail one job's console output:

    tail -f "$OMP_DIR"/results/scale_omp_2mm_*.out

After completion, inspect on fs0. Check the `.err` files first if a CSV is missing:

    cd "$OMP_DIR"
    ls -lh results/*LARGE*.csv results/*EXTRALARGE*.csv
    wc -l results/*.csv
    grep -l . results/*.err            # non-empty error files = something to read
    awk -F, 'NR>1 && $13=="FAIL" {print FILENAME": "$3" "$4"T"}' results/*.csv

Pull results to your laptop. The remote path is written in full because a local
`$OMP_DIR` would expand on your laptop, not on DAS-5. Replace `das5` with your SSH
host alias for `fs0.das5.cs.vu.nl` if it differs:

    RR="terminator_julia/terminator_julia"   # relative to ade910's home on das5
    rsync -avz --include='*.csv' --include='*.out' --include='*.err' --exclude='*' \
          das5:"$RR/openmp_polybench_refactored/results/"  ./das5_openmp_results/
    rsync -avz --include='*.csv' --include='*.out' --include='*.err' --exclude='*' \
          das5:"$RR/julia_polybench_refactored/results/"   ./das5_julia_results/

Plot and compare:

    cd openmp_polybench_refactored
    python3 scripts/visualize_benchmarks.py ../das5_openmp_results/*.csv -o ./plots -t "DAS-5 VU68"
    python3 scripts/visualize_benchmarks.py --scaling ../das5_openmp_results/*LARGE*.csv -o ./plots_scaling
    python3 scripts/compare_benchmarks.py ../das5_julia_results/*.csv ../das5_openmp_results/*.csv \
        --report das5_julia_vs_openmp.md

---

## 9. CSV schema and file naming

OpenMP binaries emit:

    benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,
    std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations

Files are named `results/<bench>_<DATASET>_<yyyymmdd_hhmmss>.csv`. Per-second
timestamps prevent collisions across the concurrent jobs, and the dataset in the
filename keeps LARGE and EXTRALARGE separate. The `benchmark` column now self-labels
correctly on the Julia side too (the prior "correlation" mislabel is fixed).

---

## 10. Resource sizing reference (EXTRALARGE)

| Benchmark   | XL size              | Seq time (1T) | 16T target | Memory peak |
|-------------|----------------------|---------------|------------|-------------|
| 2mm         | NI/J/K/L=4000        | ~4 min        | ~15-30 s   | ~500 MB     |
| 3mm         | NI/J/K/L/M=4000      | ~8 min        | ~30-60 s   | ~900 MB     |
| cholesky    | n=4000               | ~7 min        | ~20-40 s   | ~130 MB     |
| correlation | M/N=4000             | ~8 min        | ~25-45 s   | ~250 MB     |
| nussinov    | n=5500               | ~15 min       | dep-limited| ~250 MB     |
| jacobi2d    | n=2800, tsteps=1000  | bandwidth-bound | bus-limited | ~125 MB  |
| heat3d (OMP-only) | n=120, tsteps=1000 | ~10 min | ~3-5 min   | ~14 MB      |

The 4 h `--time` for the 4/8/16 sweep leaves margin for the t=4 runs of the heaviest
kernels. jacobi2d is memory-bandwidth-bound, so it saturates the bus before the cores;
expect its scaling to flatten past 8 threads on a dual-socket node.

---

## 11. Security and operational notes

- No credentials in any command; only `$USER` and `$HOME`.
- `--exclusive` blocks co-tenancy on the node. Correct for timing, and it minimises
  information leakage via `/proc`, `/sys`, and shared memory while you run.
- `-C cpunode` pins every run to identical Haswell hardware. Mandatory for valid
  comparison across benchmarks, thread counts, and languages.
- Expansion discipline: loop variables (`${b}`, `${OMP_DIR}`) expand at submit time on
  fs0; runtime variables (`\$t`, `\$(hostname)`) are escaped and expand inside the job.
- Optionally restrict the results directory: `chmod 700 "$OMP_DIR/results"`. Logs
  carry only hostnames and job IDs, but least-privilege is good hygiene.
- Never build an `sbatch --wrap=` string from untrusted input. The wrap body is
  executed verbatim by bash inside the job.

---

## 12. Why this satisfies "one benchmark per node, no queueing churn"

- Each `sbatch` submits one job requesting `-N 1 --exclusive -C cpunode`: one whole
  node, yours alone, for one benchmark.
- Six submissions enter the queue at once and run concurrently as nodes are free. No
  single shell blocks; you are not holding the jobs open.
- Jobs are detached from your terminal, so no tmux and no `wait`. Submit, then log out
  if you like; SLURM runs them and writes the results to `results/`.
- When a job's sweep finishes, its node is released immediately for the next user.
