#!/usr/bin/env julia
#=
profile_flame.jl - flame graph + flat hot-frame report for a Julia kernel.

Approach (non-invasive: no edits to run_*.jl):
  1. include("scripts/run_<kernel>.jl") once. This defines all functions AND
     runs main() once, which warms the JIT (compilation is excluded from the
     profile this way).
  2. Profile.clear(); @profile main()  -> profiles a second, JIT-warm run.
  3. Save an SVG flame graph (ProfileSVG) and a flat textual hot-frame report.

The run scripts execute every strategy, so the flame graph shows where total
time concentrates across the kernel's strategies - useful for spotting a hot
or regressing path (e.g. an allocating tiled loop).

Usage (from julia_polybench_refactored/):
  julia -t 16 profile_flame.jl <kernel> [dataset=LARGE] [iters=15]
  # kernel in: 2mm 3mm cholesky correlation jacobi2d nussinov

Requires: ProfileSVG (]add ProfileSVG). Profile is stdlib.
Output: results/flame/julia_flame_<kernel>_<dataset>.svg
        results/flame/julia_flat_<kernel>_<dataset>.txt
=#

using Profile

kernel  = length(ARGS) >= 1 ? ARGS[1] : error("usage: profile_flame.jl <kernel> [dataset] [iters]")
dataset = length(ARGS) >= 2 ? ARGS[2] : "LARGE"
iters   = length(ARGS) >= 3 ? ARGS[3] : "15"

script = joinpath(@__DIR__, "scripts", "run_$(kernel).jl")
isfile(script) || error("no run script for kernel '$kernel' at $script")

outdir = joinpath(@__DIR__, "results", "flame")
mkpath(outdir)

# child ARGS consumed by the run script's main()
empty!(ARGS)
append!(ARGS, ["--dataset", dataset, "--warmup", "5", "--iterations", iters])

println("[profile] warming JIT: include $script")
Base.include(Main, script)          # defines functions + runs main() once (warm)

@isdefined(main) || error("run script did not define main()")

println("[profile] profiling warm run")
Profile.clear()
Profile.init(; n = 10_000_000, delay = 0.0005)
@profile main()

flat_path = joinpath(outdir, "julia_flat_$(kernel)_$(dataset).txt")
open(flat_path, "w") do io
    Profile.print(IO = io, format = :flat, sortedby = :count, mincount = 5)
end
println("[profile] wrote $flat_path")

try
    @eval using ProfileSVG
    svg_path = joinpath(outdir, "julia_flame_$(kernel)_$(dataset).svg")
    Base.invokelatest(getfield(Main, :ProfileSVG).save, svg_path)
    println("[profile] wrote $svg_path")
catch e
    println("[profile] ProfileSVG unavailable ($(e)); flat report written.")
    println("[profile] install with:  julia -e 'using Pkg; Pkg.add(\"ProfileSVG\")'")
end
