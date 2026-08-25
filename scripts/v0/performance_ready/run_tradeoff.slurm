#!/usr/bin/env bash
#SBATCH --job-name=v0-fast-tradeoff
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

: "${TRADEOFF_SPLIT:?TRADEOFF_SPLIT is required (validation or heldout)}"
: "${REPO_ROOT:?REPO_ROOT is required}"

script_dir="$REPO_ROOT/scripts/v0/performance_ready"
source "$script_dir/tradeoff_matrix.sh"
configure_tradeoff_split "$TRADEOFF_SPLIT"

# This matrix is intentionally frozen. Change it only in a reviewed commit so
# every trade-off curve can be traced to an exact repository revision.
export RUN_ROLE="$TRADEOFF_SPLIT"
export QUERY_START="$TRADEOFF_QUERY_START"
export QUERY_COUNT="$TRADEOFF_QUERY_COUNT"
export K="$TRADEOFF_K"
export EF_SEARCH="$TRADEOFF_EF_SEARCH"
export BETAS="$TRADEOFF_BETAS"
export MODES="$TRADEOFF_MODES"

exec "$BASH" "$script_dir/run_calibration.slurm"
