#!/usr/bin/env bash
#SBATCH --job-name=sift1m_prepare
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
: "${REPO_ROOT:?REPO_ROOT is required}"
DATASET_DIR="${DATASET_DIR:-$HOME/IndividualProject/datasets/sift1m}"
bash "$REPO_ROOT/scripts/baseline_trace/prepare_sift1m.sh" "$DATASET_DIR"
echo "prepare_sift1m_job_ok dataset_config=$DATASET_DIR/dataset.json"
