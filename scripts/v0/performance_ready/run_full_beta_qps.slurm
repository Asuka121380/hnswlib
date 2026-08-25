#!/usr/bin/env bash
#SBATCH --job-name=v0-full-beta-qps
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclusive

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PERFORMANCE_BUILD:?PERFORMANCE_BUILD is required}"
: "${ANALYSIS_PYTHON:?ANALYSIS_PYTHON is required}"
: "${INDEX_PATH:?INDEX_PATH is required}"
: "${SIDECAR_PATH:?SIDECAR_PATH is required}"
: "${QUERY_PATH:?QUERY_PATH is required}"

script_dir="$REPO_ROOT/scripts/v0/performance_ready"
source "$script_dir/full_beta_tradeoff_matrix.sh"
cd "$REPO_ROOT"

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Commit mismatch: expected $EXPECTED_COMMIT, got $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Repository worktree is dirty" >&2
  git status --short >&2
  exit 2
fi

runner="$PERFORMANCE_BUILD/v0_performance_runner"
for required in \
    "$runner" "$PERFORMANCE_BUILD/CMakeCache.txt" \
    "$ANALYSIS_PYTHON" "$INDEX_PATH" "$SIDECAR_PATH" "$QUERY_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

"$ANALYSIS_PYTHON" "$script_dir/check_build_contract.py" \
  --build-dir "$PERFORMANCE_BUILD" --contract performance
"$ANALYSIS_PYTHON" "$script_dir/audit_fast_path_source.py" \
  --header "$REPO_ROOT/hnswlib/edge_quant_v0.h"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-unavailable}"
echo "start=$(date --iso-8601=seconds)"
echo "commit=$actual_commit"
echo "betas=$FULL_BETA_VALUES"
echo "modes=$FULL_BETA_MODES prefetches=$FULL_BETA_PREFETCHES"
echo "query_start=$FULL_BETA_QUERY_START query_count=$FULL_BETA_QUERY_COUNT"
echo "blocks=$FULL_BETA_BLOCKS repeats=$FULL_BETA_WITHIN_PROCESS_REPEATS"
taskset -pc $$ 2>&1 || true

"$ANALYSIS_PYTHON" "$script_dir/run_full_beta_qps.py" \
  --runner "$runner" \
  --index-path "$INDEX_PATH" \
  --sidecar-path "$SIDECAR_PATH" \
  --query-path "$QUERY_PATH" \
  --output-dir "$RUN_ROOT/qps" \
  --cmake-cache "$PERFORMANCE_BUILD/CMakeCache.txt" \
  --betas "$FULL_BETA_VALUES" \
  --modes "$FULL_BETA_MODES" \
  --prefetches "$FULL_BETA_PREFETCHES" \
  --dimension 960 \
  --query-start "$FULL_BETA_QUERY_START" \
  --query-count "$FULL_BETA_QUERY_COUNT" \
  --k "$FULL_BETA_K" \
  --ef-search "$FULL_BETA_EF_SEARCH" \
  --warmup-queries "$FULL_BETA_WARMUP_QUERIES" \
  --within-process-repeats "$FULL_BETA_WITHIN_PROCESS_REPEATS" \
  --blocks "$FULL_BETA_BLOCKS" \
  --seed "$FULL_BETA_SEED" \
  --experiment-commit "$actual_commit" \
  --numa-policy "${NUMA_POLICY:-slurm-exclusive-single-task}" \
  --power-policy "${POWER_POLICY:-cluster-default-unverified}" \
  --cpu-frequency-policy "${CPU_FREQUENCY_POLICY:-cluster-default-unverified}"

echo "end=$(date --iso-8601=seconds)"
echo "v0_full_beta_qps_ok"
