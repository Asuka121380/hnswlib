#!/usr/bin/env bash
#SBATCH --job-name=hnsw_cross_dataset
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
: "${REPO_ROOT:?}" "${SIFT_RUN_ROOT:?}" "${GIST_RUN_ROOT:?}" "${OUTPUT_DIR:?}" "${PYTHON_ENV:?}"
source "$PYTHON_ENV/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
python "$REPO_ROOT/scripts/baseline_trace/cross_dataset_summary.py" \
  --run-root "$SIFT_RUN_ROOT" \
  --run-root "$GIST_RUN_ROOT" \
  --output-dir "$OUTPUT_DIR"
echo "compare_datasets_job_ok output_dir=$OUTPUT_DIR"
