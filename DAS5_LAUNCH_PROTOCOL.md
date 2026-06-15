# DAS-5 Launch Protocol - prun, one benchmark per node, no queueing

Target cluster: `fs0.das5.cs.vu.nl` (VU68). Partition `defq`, constraint `cpunode`
(dual 8-core Xeon E5-2630-v3 Haswell, 16 physical cores, 64 GB).

Design goal of this protocol: **each compute node runs exactly one benchmark**,
launched **interactively via `prun`** (synchronous, shell-like), **backgrounded**
so the whole suite runs concurrently across distinct nodes. No `sbatch` batch queue
accumulates, and `--exclusive` guarantees no other work shares a node you are timing.

Everything below is a plain terminal CLI loop. There are no wrapper `.slurm` or `.sh`
files to maintain - the existing per-benchmark binaries and run scripts do the work.

---

## 0. Policy (read once)

- Edit and compile on `fs0`. **Never execute on `fs0`** - run only on compute nodes.
- Daytime cap is 15 minutes per reservation. `LARGE` fits; `EXTRALARGE` is off-hours only
  (after 18:00, weekends) and needs explicit permission for long runs.
- `prun` reserves through SLURM, so prun- and sbatch-initiated jobs never interfere.
- `module load prun` once per login session before any prun call.

---

## 1. The common benchmark set

After the refactor, both languages share six identical kernels:

    2mm  3mm  cholesky  correlation  nussinov  jacobi2d

`jacobi2d` is the common stencil (2D 5-point, coefficient 0.2, canonical strategies
`sequential, threads_static, threads_dynamic, tiled, simd, red_black`). `heat3d`
remains an OpenMP-only bonus and is **not** in the common set.

---

## 2. Build on fs0 (once)

OpenMP:

    export OMP_DIR="$HOME/Julia_vs_OpenMP_Parallelism_Multithreading/openmp_polybench_refactored"
    cd "$OMP_DIR"
    make clean && make das5          # -march=haswell -mavx2 -mfma (DAS-5 safe, NOT native)
    ls -lh benchmark_{2mm,3mm,cholesky,correlation,nussinov,jacobi2d,heat3d}

Julia needs no compile step; just confirm the module resolves:

    export JL_DIR="$HOME/Julia_vs_OpenMP_Parallelism_Multithreading/julia_polybench_refactored"
    module load julia 2>/dev/null || module load julia/1.11.4
    julia --version

Check idle cpunodes before launching (you want at least 6 idle for the common set):

    module load prun
    sinfo -p defq -o "%20N %6t %12f" | grep cpunode | grep idle

---

## 3. OpenMP - LARGE, 16 threads, one benchmark per node (daytime)

This is the core loop. Six `prun` reservations, each claims one exclusive cpunode,
each runs a different benchmark, all concurrent. `wait` blocks until all finish.

    cd "$OMP_DIR"; mkdir -p results
    export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"
    TS=$(date +%Y%m%d_%H%M%S)

    for b in $BENCHES; do
        prun -np 1 -native '-C cpunode --exclusive --time=00:15:00' \
            bash -c "cd ${OMP_DIR}; \
                     export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores; \
                     echo \"[\$(hostname)] ${b} LARGE 16T\"; \
                     ./benchmark_${b} --dataset LARGE --threads 16 \
                         --iterations 10 --warmup 3 --output csv" \
            > results/prun_${b}_${TS}.log 2>&1 &
    done
    wait
    echo "All OpenMP LARGE runs complete."

What expands when:
- `${OMP_DIR}`, `${b}` expand on `fs0` at submit time (the loop variable). Intended.
- `\$(hostname)` is escaped, so it expands on the compute node and the log records
  which node ran which benchmark.

CSV outputs land in `results/<bench>_LARGE_<timestamp>.csv` (per-second timestamps avoid
collisions across the concurrent jobs). Logs are `results/prun_<bench>_<TS>.log`.

---

## 4. OpenMP - thread-scaling sweep, one benchmark per node

Each node runs its single benchmark across `1 2 4 8 16` threads in an inner loop, so
you get a full Amdahl curve per benchmark without re-reserving nodes (no queueing churn).

    cd "$OMP_DIR"; mkdir -p results
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"
    TS=$(date +%Y%m%d_%H%M%S)

    for b in $BENCHES; do
        prun -np 1 -native '-C cpunode --exclusive --time=00:15:00' \
            bash -c "cd ${OMP_DIR}; \
                     export OMP_PROC_BIND=close OMP_PLACES=cores; \
                     for t in 1 2 4 8 16; do \
                         echo \"[\$(hostname)] ${b} LARGE t=\$t\"; \
                         OMP_NUM_THREADS=\$t ./benchmark_${b} --dataset LARGE \
                             --threads \$t --iterations 10 --warmup 3 --output csv; \
                     done" \
            > results/prun_scale_${b}_${TS}.log 2>&1 &
    done
    wait

`\$t` is escaped so the thread loop runs on the node. Loading every per-thread CSV
together gives the visualizer its scaling axis.

---

## 5. Julia - same shape, one benchmark per node

Julia binaries are JIT-compiled per run, so the compute-side shell must source lmod and
load the Julia module. BLAS is pinned to 1 thread so it does not fight the kernel threads.

    cd "$JL_DIR"; mkdir -p results
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"
    TS=$(date +%Y%m%d_%H%M%S)

    for b in $BENCHES; do
        prun -np 1 -native '-C cpunode --exclusive --time=00:15:00' \
            bash -c "source /etc/bashrc; source /etc/profile.d/lmod.sh; \
                     module load julia 2>/dev/null || module load julia/1.11.4; \
                     cd ${JL_DIR}; \
                     export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                            JULIA_NUM_THREADS=16 JULIA_DEPOT_PATH=\$HOME/.julia; \
                     echo \"[\$(hostname)] julia ${b} LARGE 16T\"; \
                     julia -t 16 scripts/run_${b}.jl --dataset LARGE \
                         --iterations 10 --warmup 5 --output csv" \
            > results/prun_julia_${b}_${TS}.log 2>&1 &
    done
    wait

For the Julia scaling sweep, replace `julia -t 16 ... --threads 16` with an inner
`for t in 1 2 4 8 16; do JULIA_NUM_THREADS=\$t julia -t \$t scripts/run_${b}.jl ...; done`.

---

## 6. Both languages at once (12 nodes)

Because each `prun` is independent, you can fire the OpenMP loop (Section 3) and the
Julia loop (Section 5) back to back in the same terminal. Twelve nodes light up, six
running OpenMP and six running Julia, all distinct, all exclusive. Run the OpenMP loop,
then immediately the Julia loop; both sets of `&` background jobs coexist. One shared
`wait` at the end joins everything. This is the fastest path to a full cross-language
LARGE matrix in a single sitting.

---

## 7. EXTRALARGE (off-hours only)

`EXTRALARGE` exceeds the 15-minute daytime cap. Run it after hours, and keep the session
alive with `tmux` so a disconnect does not kill your prun reservations.

    tmux new -s xl
    cd "$OMP_DIR"; mkdir -p results
    export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
    BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"
    TS=$(date +%Y%m%d_%H%M%S)

    for b in $BENCHES; do
        prun -np 1 -native '-C cpunode --exclusive --time=03:00:00' \
            bash -c "cd ${OMP_DIR}; export OMP_PROC_BIND=close OMP_PLACES=cores; \
                     for t in 8 16; do \
                         echo \"[\$(hostname)] ${b} XL t=\$t\"; \
                         OMP_NUM_THREADS=\$t ./benchmark_${b} --dataset EXTRALARGE \
                             --threads \$t --iterations 5 --warmup 2 --output csv; \
                     done" \
            > results/prun_xl_${b}_${TS}.log 2>&1 &
    done
    wait

Detach with `Ctrl-b d`, reattach later with `tmux attach -t xl`. For a fully unattended
overnight launch, the deferred `sbatch --begin=22:00` form in
`openmp_polybench_refactored/das5_openmp_benchmarks_deploy.md` (Section 7) is the
alternative - it queues the same one-node-per-benchmark dispatch to start at night.

---

## 8. Skip known-broken strategies

The binaries accept `--strategies` (comma-separated). To run only verified strategies for
a kernel with a known failure (e.g. cholesky `tiled`/`threads_static` historically FAILed):

    ./benchmark_cholesky --dataset EXTRALARGE --threads 16 \
        --iterations 5 --warmup 2 --output csv \
        --strategies sequential,threads_dynamic,simd

Drop that into the prun `bash -c` body for that benchmark only.

---

## 9. Monitor, collect, visualize

While running:

    squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %.20R"   # your reservations
    tail -f "$OMP_DIR"/results/prun_2mm_*.log                # one benchmark's log

Sanity-check for verification failures across all CSVs:

    cd "$OMP_DIR"
    awk -F, 'NR>1 && $13=="FAIL" {print FILENAME": "$3" "$4"T"}' results/*.csv

Pull results to your laptop:

    rsync -avz --include='*.csv' --include='*.log' --exclude='*' \
        das5:"$OMP_DIR"/results/  ./das5_openmp_results/
    rsync -avz --include='*.csv' --include='*.log' --exclude='*' \
        das5:"$JL_DIR"/results/   ./das5_julia_results/

Plot and compare:

    cd openmp_polybench_refactored
    python3 scripts/visualize_benchmarks.py ../das5_openmp_results/*.csv -o ./plots -t "DAS-5 VU68"
    python3 scripts/visualize_benchmarks.py --scaling ../das5_openmp_results/*LARGE*.csv -o ./plots_scaling
    python3 scripts/compare_benchmarks.py ../das5_julia_results/*.csv ../das5_openmp_results/*.csv \
        --report das5_julia_vs_openmp.md

---

## 10. Why this satisfies "one benchmark per node, no queueing"

- `prun -np 1` reserves a single full node and runs one process on it. With six
  benchmarks backgrounded, six distinct nodes are claimed at once.
- `--exclusive` blocks co-tenancy, so nothing else shares the node while you time it.
- `prun` is synchronous per call; backgrounding gives concurrency without a growing
  batch queue. When a benchmark finishes, its node is released immediately.
- `-C cpunode` pins every run to identical Haswell hardware - mandatory for valid
  cross-benchmark and cross-language comparison.



One prerequisite: confirm the OpenMP binaries exist (ls -lh "$OMP_DIR"/benchmark_2mm). If not, cd "$OMP_DIR" && make das5 first. Julia needs no build.
OpenMP — LARGE, scaling 4/8/16
bashOMP_DIR="$HOME/terminator_julia/terminator_julia/openmp_polybench_refactored"
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
                       OMP_NUM_THREADS=\$t ./benchmark_${b} --dataset LARGE --threads \$t --iterations 10 --warmup 3 --output csv; \
                   done"
done
squeue -u $USER
Julia — LARGE, scaling 4/8/16
bashJL_DIR="$HOME/terminator_julia/terminator_julia/julia_polybench_refactored"
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
                       JULIA_NUM_THREADS=\$t julia -t \$t scripts/run_${b}.jl --dataset LARGE --iterations 10 --output csv; \
                   done"
done
squeue -u $USER
EXTRALARGE — both languages, deferred to off-hours (no tmux)
EXTRALARGE exceeds the 15-min daytime cap, so --begin=22:00 defers the start to tonight (use --begin=saturday for the weekend). The job waits in the queue, then runs unattended. Per DAS-5 policy, long off-hours jobs may need your supervisor's nod.
OpenMP:
bashOMP_DIR="$HOME/terminator_julia/terminator_julia/openmp_polybench_refactored"
mkdir -p "$OMP_DIR/results"
BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

for b in $BENCHES; do
    sbatch --job-name="scaleXL_omp_${b}" \
           --output="${OMP_DIR}/results/scaleXL_omp_${b}_%j.out" \
           --error="${OMP_DIR}/results/scaleXL_omp_${b}_%j.err" \
           --time=04:00:00 --begin=22:00 \
           -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
           --partition=defq -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                   module load prun; \
                   export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false; \
                   cd ${OMP_DIR}; \
                   for t in 4 8 16; do \
                       echo \"[\$(hostname)] ${b} EXTRALARGE t=\$t\"; \
                       OMP_NUM_THREADS=\$t ./benchmark_${b} --dataset EXTRALARGE --threads \$t --iterations 5 --warmup 2 --output csv; \
                   done"
done
squeue -u $USER
Julia (iterations trimmed to 3; the script's fixed 5 warmups are heavy at XL):
bashJL_DIR="$HOME/terminator_julia/terminator_julia/julia_polybench_refactored"
mkdir -p "$JL_DIR/results"
BENCHES="2mm 3mm cholesky correlation nussinov jacobi2d"

for b in $BENCHES; do
    sbatch --job-name="scaleXL_jl_${b}" \
           --output="${JL_DIR}/results/scaleXL_jl_${b}_%j.out" \
           --error="${JL_DIR}/results/scaleXL_jl_${b}_%j.err" \
           --time=04:00:00 --begin=22:00 \
           -N 1 --ntasks=1 --cpus-per-task=16 --exclusive \
           --partition=defq -C cpunode \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; \
                   module load prun julia/1.11.4; \
                   export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
                   cd ${JL_DIR}; \
                   for t in 4 8 16; do \
                       echo \"[\$(hostname)] julia ${b} EXTRALARGE t=\$t\"; \
                       JULIA_NUM_THREADS=\$t julia -t \$t scripts/run_${b}.jl --dataset EXTRALARGE --iterations 3 --output csv; \
                   done"
done
squeue -u $USER
After submitting, watch with squeue -u $USER; each job shows R when running, PD while queued (XL jobs sit PD with reason (BeginTime) until 22:00). Per-run console goes to results/scale_*_%j.out, errors to results/scale_*_%j.err — check the .err files first if a CSV is missing. The CSVs self-name with dataset and timestamp as before.
