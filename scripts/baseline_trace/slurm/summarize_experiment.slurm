#!/usr/bin/env bash
#SBATCH --job-name=hnsw_summary
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
: "${REPO_ROOT:?}" "${RUN_ROOT:?}" "${PYTHON_ENV:?}"
source "$PYTHON_ENV/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
python "$REPO_ROOT/scripts/baseline_trace/cross_ef_summary.py" \
  --run-root "$RUN_ROOT"
echo "summarize_experiment_job_ok"
