#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through submit_final_study.sh or use Python --local-smoke}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo"
# Lock the node for this user's study workers; Slurm --exclusive excludes other allocations.
exec 9>"/tmp/edge-pruning-qps-${USER}.lock"
flock -n 9 || { echo "Another measurement worker is active on this node" >&2; exit 1; }
exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "${PYTHON:-python3}" -m scripts.edge_estimation.final_study.run "$@"
