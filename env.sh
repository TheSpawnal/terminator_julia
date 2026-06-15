# DAS-5 project environment for terminator_julia
# Source once per login session on fs0:  source env.sh
# Do NOT add this to .bashrc; module loads in .bashrc can break non-interactive shells.
 
export ROOT="$HOME/terminator_julia/terminator_julia"
export OMP_DIR="$ROOT/openmp_polybench_refactored"
export JL_DIR="$ROOT/julia_polybench_refactored"
 
# OpenMP runtime defaults (harmless to non-OMP shells)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
 
module load prun 2>/dev/null
 
echo "env loaded: ROOT=$ROOT"
 
