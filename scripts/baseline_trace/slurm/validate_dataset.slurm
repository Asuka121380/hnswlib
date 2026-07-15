#!/usr/bin/env bash
#SBATCH --job-name=hnsw_data_check
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
: "${REPO_ROOT:?}" "${DATASET_CONFIG:?}" "${PYTHON_ENV:?}"
source "$PYTHON_ENV/bin/activate"
python "$REPO_ROOT/scripts/baseline_trace/validate_dataset.py" \
  --dataset-config "$DATASET_CONFIG"
echo "validate_dataset_job_ok"
