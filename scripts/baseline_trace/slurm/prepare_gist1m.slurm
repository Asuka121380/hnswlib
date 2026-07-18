#!/usr/bin/env bash
#SBATCH --job-name=gist1m_prepare
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
: "${REPO_ROOT:?REPO_ROOT is required}"
DATASET_DIR="${DATASET_DIR:-$HOME/IndividualProject/datasets/gist1m}"
bash "$REPO_ROOT/scripts/baseline_trace/prepare_gist1m.sh" "$DATASET_DIR"
echo "prepare_gist1m_job_ok dataset_config=$DATASET_DIR/dataset.json"
