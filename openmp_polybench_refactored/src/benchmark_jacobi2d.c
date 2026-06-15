/*
 * Jacobi-2D Benchmark - 2D Jacobi Stencil (PolyBench)
 * 5-point stencil, double-buffered Jacobi iteration.
 *
 * This is the COMMON stencil kernel shared with the Julia suite
 * (julia_polybench_refactored/scripts/run_jacobi2d.jl). Dataset sizes,
 * stencil coefficient (0.2), and strategy names are kept identical so the
 * two languages can be compared like-for-like (RQ3, memory-bound axis).
 *
 * Update rule: u'[i,j] = 0.2 * (u[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
 *
 * Memory-bound kernel. Arithmetic intensity ~0.1 FLOP/byte
 * (5 FLOPs/point, 5 reads + 1 write). Performance limited by memory bandwidth.
 *
 * Strategies (names canonicalized to match the Julia suite):
 *   sequential       - Baseline Jacobi iteration (double-buffered)
 *   threads_static   - parallel for over j with static scheduling
 *   threads_dynamic  - parallel for over j with dynamic scheduling
 *   tiled            - cache-blocked i,j tiles, thread-parallel over tile rows
 *   simd             - #pragma omp simd inner loop + static thread split
 *   red_black        - red-black Gauss-Seidel ordering (in-place, more parallel)
 *
 * Verification:
 *   Jacobi-family strategies compare against the sequential Jacobi reference.
 *   red_black follows a different numerical trajectory than Jacobi, so it is
 *   compared against a dedicated sequential red-black reference with a relaxed
 *   tolerance, mirroring the Julia runner's treatment of red-black.
 *
 * Layout note:
 *   PolyBench/C jacobi-2d is row-major; the inner loop runs over j so that the
 *   contiguous stride-1 access (i fixed) is vectorizable. This is the C/row-major
 *   counterpart of the Julia column-major version, which threads over j and runs
 *   simd over i. The numerics are identical; only the contiguous axis differs.
 *
 * References:
 *   PolyBench/C 4.2.1 - jacobi-2d kernel
 *   "Structured Parallel Programming" - McCool, Robison, Reinders
 */

#include "benchmark_common.h"
#include "metrics.h"
#include <getopt.h>
#include <strings.h>   /* strcasecmp */

/* ------------------------------------------------------------------ */
/*  Dataset configurations (identical to the Julia jacobi2d runner)   */
/* ------------------------------------------------------------------ */

typedef struct {
    int n;
    int tsteps;
} DatasetJacobi2D;

static const DatasetJacobi2D DATASETS[] = {
    {30,    20},    /* MINI       */
    {90,    40},    /* SMALL      */
    {250,   100},   /* MEDIUM     */
    {1300,  500},   /* LARGE      */
    {2800,  1000}   /* EXTRALARGE */
};

/* Stencil coefficient */
#define COEFF 0.2

/* Tile size for spatial cache blocking */
#define TILE 64

/* ------------------------------------------------------------------ */
/*  5-point stencil (row-major: contiguous along j)                   */
/* ------------------------------------------------------------------ */

static inline double stencil_5pt(const double* restrict src, int i, int j, int n) {
    return COEFF * (src[IDX2(i,   j,   n)] +
                    src[IDX2(i-1, j,   n)] +
                    src[IDX2(i+1, j,   n)] +
                    src[IDX2(i,   j-1, n)] +
                    src[IDX2(i,   j+1, n)]);
}

/* ------------------------------------------------------------------ */
/*  Initialization (matches the Julia init_arrays!)                   */
/* ------------------------------------------------------------------ */

static void init_array(int n, double* A, double* B) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double val = (double)(i * (n - i) + j * (n - j)) / n;
            A[IDX2(i, j, n)] = val;
            B[IDX2(i, j, n)] = val;
        }
}

/* ------------------------------------------------------------------ */
/*  Verification: L-infinity relative error vs reference              */
/* ------------------------------------------------------------------ */

static double verify_result(int n, const double* ref, const double* out) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int idx = IDX2(i, j, n);
            double err = fabs(ref[idx] - out[idx]);
            if (fabs(ref[idx]) > 1e-15) err /= fabs(ref[idx]);
            if (err > max_err) max_err = err;
        }
    return max_err;
}

/* ------------------------------------------------------------------ */
/*  Strategy 1: Sequential baseline (Jacobi double-buffer)            */
/* ------------------------------------------------------------------ */

static void kernel_jacobi2d_sequential(int tsteps, int n,
                                       double* restrict A,
                                       double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                B[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);

        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                A[IDX2(i,j,n)] = stencil_5pt(B, i, j, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 2: threads_static                                        */
/* ------------------------------------------------------------------ */

static void kernel_jacobi2d_threads_static(int tsteps, int n,
                                           double* restrict A,
                                           double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                B[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);

        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                A[IDX2(i,j,n)] = stencil_5pt(B, i, j, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 3: threads_dynamic                                       */
/* ------------------------------------------------------------------ */

static void kernel_jacobi2d_threads_dynamic(int tsteps, int n,
                                            double* restrict A,
                                            double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                B[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 1; i < n-1; i++)
            for (int j = 1; j < n-1; j++)
                A[IDX2(i,j,n)] = stencil_5pt(B, i, j, n);
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 4: Tiled (spatial cache blocking on i,j)                 */
/* ------------------------------------------------------------------ */

static void kernel_jacobi2d_tiled(int tsteps, int n,
                                  double* restrict A,
                                  double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for schedule(static)
        for (int ii = 1; ii < n-1; ii += TILE) {
            int i_end = MIN(ii + TILE, n-1);
            for (int jj = 1; jj < n-1; jj += TILE) {
                int j_end = MIN(jj + TILE, n-1);
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        B[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int ii = 1; ii < n-1; ii += TILE) {
            int i_end = MIN(ii + TILE, n-1);
            for (int jj = 1; jj < n-1; jj += TILE) {
                int j_end = MIN(jj + TILE, n-1);
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        A[IDX2(i,j,n)] = stencil_5pt(B, i, j, n);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy 5: SIMD (vectorized contiguous j-loop + thread split)    */
/* ------------------------------------------------------------------ */

static void kernel_jacobi2d_simd(int tsteps, int n,
                                 double* restrict A,
                                 double* restrict B) {
    for (int t = 0; t < tsteps; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n-1; i++) {
            #pragma omp simd
            for (int j = 1; j < n-1; j++)
                B[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n-1; i++) {
            #pragma omp simd
            for (int j = 1; j < n-1; j++)
                A[IDX2(i,j,n)] = stencil_5pt(B, i, j, n);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Red-Black Gauss-Seidel                                            */
/*                                                                    */
/*  In a 5-point stencil, red cells (i+j even) depend only on black   */
/*  neighbours and vice versa, so within a colour phase the updates   */
/*  are race-free and the parallel version matches the sequential     */
/*  red-black reference within tolerance. Runs 2*tsteps sweeps to     */
/*  match the Jacobi double-sweep FLOP count.                         */
/* ------------------------------------------------------------------ */

/* Sequential red-black reference (ground truth; not parallelized) */
static void kernel_jacobi2d_redblack_seq(int tsteps, int n, double* restrict A) {
    int total_iters = 2 * tsteps;
    for (int t = 0; t < total_iters; t++) {
        /* Red phase: (i+j) even */
        for (int i = 1; i < n-1; i++) {
            int j0 = 1 + ((i + 1) & 1);
            for (int j = j0; j < n-1; j += 2)
                A[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);
        }
        /* Black phase: (i+j) odd */
        for (int i = 1; i < n-1; i++) {
            int j0 = 1 + (i & 1);
            for (int j = j0; j < n-1; j += 2)
                A[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);
        }
    }
}

/* Strategy 6: parallel red-black (verified against sequential red-black) */
static void kernel_jacobi2d_red_black(int tsteps, int n,
                                      double* restrict A,
                                      double* restrict B) {
    int total_iters = 2 * tsteps;
    for (int t = 0; t < total_iters; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n-1; i++) {
            int j0 = 1 + ((i + 1) & 1);
            for (int j = j0; j < n-1; j += 2)
                A[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n-1; i++) {
            int j0 = 1 + (i & 1);
            for (int j = j0; j < n-1; j += 2)
                A[IDX2(i,j,n)] = stencil_5pt(A, i, j, n);
        }
    }

    /* Keep B in sync with A for interface consistency with the driver */
    memcpy(B, A, (size_t)n * n * sizeof(double));
}

/* ------------------------------------------------------------------ */
/*  Strategy dispatch table                                           */
/* ------------------------------------------------------------------ */

typedef void (*KernelFunc)(int, int, double*, double*);

typedef enum {
    REF_JACOBI   = 0,
    REF_REDBLACK = 1
} RefKind;

typedef struct {
    const char* name;
    KernelFunc  func;
    RefKind     ref;
} Strategy;

static const Strategy STRATEGIES[] = {
    {"sequential",      kernel_jacobi2d_sequential,      REF_JACOBI},
    {"threads_static",  kernel_jacobi2d_threads_static,  REF_JACOBI},
    {"threads_dynamic", kernel_jacobi2d_threads_dynamic, REF_JACOBI},
    {"tiled",           kernel_jacobi2d_tiled,           REF_JACOBI},
    {"simd",            kernel_jacobi2d_simd,            REF_JACOBI},
    {"red_black",       kernel_jacobi2d_red_black,       REF_REDBLACK}
};

static const int NUM_STRATEGIES = sizeof(STRATEGIES) / sizeof(STRATEGIES[0]);

/* Returns 1 if the strategy matching `name` should run given the CLI arg */
static int strategy_selected(const char* requested, const char* name) {
    if (!requested || strcmp(requested, "all") == 0) return 1;
    return strstr(requested, name) != NULL;
}

/* ------------------------------------------------------------------ */
/*  CLI                                                               */
/* ------------------------------------------------------------------ */

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --dataset SIZE     MINI, SMALL, MEDIUM, LARGE, EXTRALARGE (default: LARGE)\n");
    printf("  --iterations N     Timed iterations (default: 10)\n");
    printf("  --warmup N         Warmup iterations (default: 3)\n");
    printf("  --threads N        OpenMP thread count (default: all)\n");
    printf("  --output csv       Export results to CSV\n");
    printf("  --strategies LIST  Comma-separated or 'all' (default: all)\n");
    printf("  --help             Show this help\n");
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    DatasetSize dataset_size = DATASET_LARGE;
    int iterations = 10;
    int warmup     = 3;
    int threads    = omp_get_max_threads();
    int output_csv = 0;
    char* strategies_arg = NULL;

    static struct option long_options[] = {
        {"dataset",    required_argument, 0, 'd'},
        {"iterations", required_argument, 0, 'i'},
        {"warmup",     required_argument, 0, 'w'},
        {"threads",    required_argument, 0, 't'},
        {"output",     required_argument, 0, 'o'},
        {"strategies", required_argument, 0, 's'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:i:w:t:o:s:h",
                              long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                for (int i = 0; i <= DATASET_EXTRALARGE; i++) {
                    if (strcasecmp(optarg, DATASET_NAMES[i]) == 0) {
                        dataset_size = (DatasetSize)i;
                        break;
                    }
                }
                break;
            case 'i': iterations = atoi(optarg); break;
            case 'w': warmup     = atoi(optarg); break;
            case 't': threads    = atoi(optarg); break;
            case 'o': output_csv = (strcmp(optarg, "csv") == 0); break;
            case 's': strategies_arg = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
        }
    }

    setup_openmp_env();
    omp_set_num_threads(threads);

    const DatasetJacobi2D* ds = &DATASETS[dataset_size];
    int n      = ds->n;
    int tsteps = ds->tsteps;
    double flops = flops_jacobi2d(tsteps, n);

    size_t array_bytes = (size_t)n * n * sizeof(double);

    printf("Jacobi-2D Benchmark\n");
    printf("Dataset: %s (N=%d, TSTEPS=%d)\n",
           DATASET_NAMES[dataset_size], n, tsteps);
    printf("Threads: %d | Iterations: %d | Warmup: %d\n",
           threads, iterations, warmup);
    printf("FLOPS: %.2e | Memory: %.1f MB (per array)\n",
           flops, array_bytes / (1024.0 * 1024.0));
    printf("NOTE: memory-bound, AI ~0.1 FLOP/byte (bandwidth-limited)\n\n");

    /* Allocate arrays (flat 1D, cache-aligned) */
    double* A        = ALLOC_2D(double, n, n);
    double* B        = ALLOC_2D(double, n, n);
    double* A_jacobi = ALLOC_2D(double, n, n);
    double* A_rb     = NULL;

    /* Decide whether the red-black reference is needed */
    int need_rb_ref = 0;
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        if (STRATEGIES[s].ref == REF_REDBLACK &&
            strategy_selected(strategies_arg, STRATEGIES[s].name)) {
            need_rb_ref = 1;
            break;
        }
    }

    /* Jacobi reference */
    init_array(n, A, B);
    kernel_jacobi2d_sequential(tsteps, n, A, B);
    memcpy(A_jacobi, A, array_bytes);

    /* Red-black reference (only if used) */
    if (need_rb_ref) {
        A_rb = ALLOC_2D(double, n, n);
        init_array(n, A, B);
        kernel_jacobi2d_redblack_seq(tsteps, n, A);
        memcpy(A_rb, A, array_bytes);
    }

    /* Metrics collector */
    MetricsCollector mc;
    metrics_init(&mc, "jacobi2d", DATASET_NAMES[dataset_size], threads);
    metrics_print_header();

    /* Run strategies */
    for (int s = 0; s < NUM_STRATEGIES; s++) {
        if (!strategy_selected(strategies_arg, STRATEGIES[s].name))
            continue;

        TimingData timing;
        timing_init(&timing);

        /* Warmup */
        for (int w = 0; w < warmup; w++) {
            init_array(n, A, B);
            STRATEGIES[s].func(tsteps, n, A, B);
        }

        /* Timed iterations */
        for (int iter = 0; iter < iterations; iter++) {
            init_array(n, A, B);

            double t0 = omp_get_wtime();
            STRATEGIES[s].func(tsteps, n, A, B);
            double t1 = omp_get_wtime();

            timing_record(&timing, (t1 - t0) * 1000.0);
        }

        /* Verification run */
        init_array(n, A, B);
        STRATEGIES[s].func(tsteps, n, A, B);

        const double* ref = (STRATEGIES[s].ref == REF_REDBLACK) ? A_rb : A_jacobi;
        double max_err = verify_result(n, ref, A);
        /* red-black uses a relaxed tolerance, mirroring the Julia runner */
        double tol = (STRATEGIES[s].ref == REF_REDBLACK) ? 1e-4 : VERIFY_TOLERANCE;
        int verified = (max_err < tol);

        metrics_record(&mc, STRATEGIES[s].name, &timing, flops,
                       verified, max_err);
        metrics_print_result(&mc.results[mc.num_results - 1]);
    }

    /* CSV export */
    if (output_csv) {
        char filename[256];
        char ts[64];
        get_timestamp(ts, sizeof(ts));
        snprintf(filename, sizeof(filename),
                 "results/jacobi2d_%s_%s.csv",
                 DATASET_NAMES[dataset_size], ts);
        metrics_export_csv(&mc, filename);
    }

    /* Cleanup */
    FREE_ARRAY(A);
    FREE_ARRAY(B);
    FREE_ARRAY(A_jacobi);
    if (A_rb) FREE_ARRAY(A_rb);

    return 0;
}
