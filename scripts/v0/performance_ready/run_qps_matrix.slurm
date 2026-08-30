#!/usr/bin/env bash
#SBATCH --job-name=v0-qps-matrix
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${RESOURCE_PROFILE:?RESOURCE_PROFILE is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PERFORMANCE_BUILD:?PERFORMANCE_BUILD is required}"
: "${ANALYSIS_PYTHON:?ANALYSIS_PYTHON is required}"
: "${INDEX_PATH:?INDEX_PATH is required}"
: "${SIDECAR_PATH:?SIDECAR_PATH is required}"
: "${QUERY_PATH:?QUERY_PATH is required}"

script_dir="$REPO_ROOT/scripts/v0/performance_ready"
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
    "$CONFIG_PATH" "$runner" "$PERFORMANCE_BUILD/CMakeCache.txt" \
    "$ANALYSIS_PYTHON" "$INDEX_PATH" "$SIDECAR_PATH" "$QUERY_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

"$ANALYSIS_PYTHON" "$script_dir/qps_config.py" \
  --config "$CONFIG_PATH" \
  --resource-profile "$RESOURCE_PROFILE"
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
echo "config=$CONFIG_PATH"
echo "resource_profile=$RESOURCE_PROFILE"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unavailable}"
echo "slurm_cpus_on_node=${SLURM_CPUS_ON_NODE:-unavailable}"
echo "slurm_job_cpus_per_node=${SLURM_JOB_CPUS_PER_NODE:-unavailable}"
taskset -pc $$ 2>&1 || true
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -dd "$SLURM_JOB_ID" || true
fi

runner_args=(
  --config "$CONFIG_PATH"
  --resource-profile "$RESOURCE_PROFILE"
  --runner "$runner"
  --index-path "$INDEX_PATH"
  --sidecar-path "$SIDECAR_PATH"
  --query-path "$QUERY_PATH"
  --run-root "$RUN_ROOT"
  --cmake-cache "$PERFORMANCE_BUILD/CMakeCache.txt"
  --experiment-commit "$actual_commit"
  --numa-policy "${NUMA_POLICY:-slurm-single-task}"
  --power-policy "${POWER_POLICY:-cluster-default-unverified}"
  --cpu-frequency-policy "${CPU_FREQUENCY_POLICY:-cluster-default-unverified}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  runner_args+=(--resume)
fi

"$ANALYSIS_PYTHON" "$script_dir/run_qps_matrix.py" "${runner_args[@]}"
"$ANALYSIS_PYTHON" "$script_dir/summarize_qps_experiment.py" \
  --run-root "$RUN_ROOT"

echo "end=$(date --iso-8601=seconds)"
echo "v0_qps_matrix_ok"
