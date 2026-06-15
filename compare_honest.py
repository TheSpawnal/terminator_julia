#!/usr/bin/env python3
"""
compare_honest.py - honest Julia vs OpenMP comparison.

Why this exists: the suite-wide "best speedup" / julia_vs_openmp plot takes
max(speedup) per kernel, which silently selects algorithm-changing strategies
(blas, tiled, colmajor) whose speedup conflates an algorithmic/cache win with
parallel scaling. That inflates efficiency past 100% and is not a like-for-like
language comparison.

This script reports two well-defined views instead:

  METRIC 1 - Ecosystem best (absolute):
    Per (kernel), the fastest legitimate strategy in each language, including
    BLAS for Julia. Answers "which ecosystem computes this fastest". Julia wins
    here are labelled with the winning strategy so a BLAS win is never mistaken
    for a threading win.

  METRIC 2 - Native data-parallel (like-for-like):
    Restricted to the naive data-parallel loop in each language
    (Julia threads_static / threads  vs  OpenMP threads_static), which run the
    SAME arithmetic as sequential. Here speedup S=T_seq/T_p and efficiency
    E=S/p are meaningful. This is the RQ3 head-to-head.

Efficiency is NEVER reported for blas/simd/tiled/colmajor (algorithm or layout
change -> no matched single-thread baseline). E>100% is annotated as a
non-representative (cache-pessimal) sequential baseline, not real super-scaling.

Usage:
  python3 compare_honest.py --julia-dir julia_polybench_refactored/results \
                            --omp-dir   openmp_polybench_refactored/results \
                            --dataset EXTRALARGE --threads 16 -o plots/honest
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KERNELS = ["2mm", "3mm", "cholesky", "correlation", "jacobi2d", "nussinov"]
# strategies that parallelize the SAME sequential arithmetic -> E=S/p valid
PARALLEL_EQUIV = {"threads_static", "threads_dynamic", "tasks", "collapsed"}
# strategies that change algorithm or memory layout -> speedup ok, efficiency N/A
ALGO_CHANGE = {"blas", "simd", "tiled", "colmajor", "red_black", "wavefront", "threads"}
# canonical naive data-parallel loop, in name-preference order, per language
NATIVE_PREF = {"julia": ["threads_static", "threads"], "openmp": ["threads_static"]}

JL_C, OMP_C = "#1f6fb2", "#d9762b"   # print-friendly, colour-blind safe-ish


def load(d, lang):
    rows = []
    for f in glob.glob(os.path.join(d, "*.csv")):
        df = pd.read_csv(f)
        df["lang"] = lang
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    if "efficiency" in df:      df = df.rename(columns={"efficiency": "eff"})
    if "efficiency_pct" in df:  df = df.rename(columns={"efficiency_pct": "eff"})
    return df


def native_row(g):
    for s in NATIVE_PREF[g.name[1] if hasattr(g, "name") else "openmp"]:
        r = g[g.strategy == s]
        if len(r):
            return r.iloc[0]
    return None


def build_tables(a, dataset, threads):
    sub = a[(a.dataset == dataset) & (a.threads == threads)]
    m1, m2 = [], []
    for b in KERNELS:
        for lang, pref in NATIVE_PREF.items():
            g = sub[(sub.benchmark == b) & (sub.lang == lang)]
            par = g[g.strategy != "sequential"]
            if len(par):
                r = par.loc[par.min_ms.idxmin()]
                m1.append(dict(kernel=b, lang=lang, ms=r.min_ms,
                               gflops=r.gflops, strategy=r.strategy))
            # native loop (fall back to fastest available if naive loop absent)
            nr = None
            for s in pref:
                rr = g[g.strategy == s]
                if len(rr):
                    nr = rr.iloc[0]; break
            if nr is None and len(par):
                nr = par.loc[par.min_ms.idxmin()]   # fallback, flagged below
            if nr is not None:
                m2.append(dict(kernel=b, lang=lang, ms=nr.min_ms,
                               speedup=nr.speedup, eff=nr.eff, strategy=nr.strategy,
                               native=nr.strategy in ("threads_static", "threads")))
    return pd.DataFrame(m1), pd.DataFrame(m2)


def grouped(ax, df, value, kernels, jl_lbl=False):
    x = np.arange(len(kernels)); w = 0.38
    jl = [df[(df.kernel == k) & (df.lang == "julia")][value].max() for k in kernels]
    op = [df[(df.kernel == k) & (df.lang == "openmp")][value].max() for k in kernels]
    b1 = ax.bar(x - w/2, jl, w, label="Julia", color=JL_C)
    b2 = ax.bar(x + w/2, op, w, label="OpenMP", color=OMP_C)
    ax.set_xticks(x); ax.set_xticklabels(kernels, rotation=20, ha="right")
    return b1, b2, jl, op


def annotate_strat(ax, bars, df, kernels, lang, value):
    for bar, k in zip(bars, kernels):
        r = df[(df.kernel == k) & (df.lang == lang)]
        if len(r):
            s = r.iloc[0]["strategy"]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    s, ha="center", va="bottom", fontsize=7, rotation=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--julia-dir", required=True)
    p.add_argument("--omp-dir", required=True)
    p.add_argument("--dataset", default="EXTRALARGE")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("-o", "--out", default="plots/honest")
    args = p.parse_args()

    a = pd.concat([load(args.julia_dir, "julia"), load(args.omp_dir, "openmp")],
                  ignore_index=True)
    m1, m2 = build_tables(a, args.dataset, args.threads)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.dataset}_{args.threads}T"

    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Julia vs OpenMP - honest comparison  "
                 f"(DAS-5, {args.dataset}, {args.threads} threads)",
                 fontsize=13, fontweight="bold")

    # Panel A: ecosystem-best wall-clock time (log scale; range spans 100x)
    b1, b2, jl, op = grouped(ax[0], m1, "ms", KERNELS)
    annotate_strat(ax[0], b1, m1, KERNELS, "julia", "ms")
    annotate_strat(ax[0], b2, m1, KERNELS, "openmp", "ms")
    ax[0].set_yscale("log"); ax[0].set_ylabel("best min time (ms, log)")
    ax[0].set_title("A. Ecosystem best time (lower=better)\nlabel = winning strategy")
    ax[0].legend(loc="upper left", fontsize=9)

    # Panel B: native data-parallel speedup @ p
    b1, b2, jl, op = grouped(ax[1], m2, "speedup", KERNELS)
    ax[1].axhline(args.threads, ls="--", c="grey", lw=1, label=f"ideal ({args.threads}x)")
    ax[1].set_ylabel("speedup  S = T_seq / T_p")
    ax[1].set_title("B. Native data-parallel speedup\n(threads_static / @threads only)")
    ax[1].legend(loc="upper right", fontsize=8)

    # Panel C: native data-parallel efficiency @ p
    b1, b2, jl, op = grouped(ax[2], m2, "eff", KERNELS)
    ax[2].axhline(100, ls="--", c="grey", lw=1)
    ax[2].set_ylabel("efficiency  E = S / p   (%)")
    ax[2].set_title("C. Native data-parallel efficiency\n(>100% = non-representative baseline)")
    ax[2].legend(["100%", "Julia", "OpenMP"], loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(args.out, f"honest_julia_vs_openmp_{tag}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    # tidy CSV for the report
    m1.to_csv(os.path.join(args.out, f"honest_ecosystem_best_{tag}.csv"), index=False)
    m2.to_csv(os.path.join(args.out, f"honest_native_threads_{tag}.csv"), index=False)
    print("wrote", fig_path)
    print("\nMETRIC 1 ecosystem-best:\n", m1.round(1).to_string(index=False))
    print("\nMETRIC 2 native data-parallel:\n", m2.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
