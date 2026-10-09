#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through submit_final_study.sh or use Python --local-smoke}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
# sbatch executes a spooled copy: BASH_SOURCE does not identify this checkout.
if [[ -n "${REPO:-}" ]]; then
  repo="$(cd -- "$REPO" && pwd -P)"
else
  repo="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Cannot locate repository; export REPO or submit with --chdir=<repository>." >&2
    exit 2
  }
fi
if [[ ! -f "$repo/CMakeLists.txt" || ! -f "$repo/scripts/edge_estimation/final_study/run.py" ]]; then
  echo "Invalid study repository: $repo" >&2
  exit 2
fi
export REPO="$repo"
cd "$repo"
# Lock the node for this user's study workers; Slurm --exclusive excludes other allocations.
exec 9>"/tmp/edge-pruning-qps-${USER}.lock"
flock -n 9 || { echo "Another measurement worker is active on this node" >&2; exit 1; }
exec srun --chdir="$repo" --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "${PYTHON:-python3}" -m scripts.edge_estimation.final_study.run "$@"
