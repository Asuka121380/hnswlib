#!/usr/bin/env bash
#SBATCH --job-name=v0-full-beta-metrics
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

script_dir="$REPO_ROOT/scripts/v0/performance_ready"
source "$script_dir/full_beta_tradeoff_matrix.sh"

export RUN_ROOT="$RUN_ROOT/metrics"
export RUN_ROLE="full-tradeoff"
export QUERY_START="$FULL_BETA_QUERY_START"
export QUERY_COUNT="$FULL_BETA_QUERY_COUNT"
export K="$FULL_BETA_K"
export EF_SEARCH="$FULL_BETA_EF_SEARCH"
export BETAS="$FULL_BETA_VALUES"
export MODES="$FULL_BETA_MODES"

exec "$BASH" "$script_dir/run_calibration.slurm"
