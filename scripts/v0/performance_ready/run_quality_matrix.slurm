#!/usr/bin/env bash
#SBATCH --job-name=v0-quality-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

# Site resource selection is supplied by submit_quality_experiment.sh.

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${RESOURCE_PROFILE:?RESOURCE_PROFILE is required}"
: "${REQUESTED_NODELIST:?REQUESTED_NODELIST is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${REFERENCE_BUILD:?REFERENCE_BUILD is required}"
: "${ANALYSIS_PYTHON:?ANALYSIS_PYTHON is required}"
: "${DATASET_CONFIG:?DATASET_CONFIG is required}"
: "${INDEX_PATH:?INDEX_PATH is required}"
: "${SIDECAR_PATH:?SIDECAR_PATH is required}"

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

runner="$REFERENCE_BUILD/v0_search_runner"
for required in "$CONFIG_PATH" "$runner" "$REFERENCE_BUILD/CMakeCache.txt" \
    "$ANALYSIS_PYTHON" "$DATASET_CONFIG" "$INDEX_PATH" "$SIDECAR_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

script_dir="$REPO_ROOT/scripts/v0/performance_ready"
"$ANALYSIS_PYTHON" "$script_dir/qps_config.py" \
  --config "$CONFIG_PATH" --resource-profile "$RESOURCE_PROFILE"
"$ANALYSIS_PYTHON" "$script_dir/check_build_contract.py" \
  --build-dir "$REFERENCE_BUILD" --contract reference

arguments=(
  --config "$CONFIG_PATH"
  --resource-profile "$RESOURCE_PROFILE"
  --runner "$runner"
  --dataset-config "$DATASET_CONFIG"
  --index-path "$INDEX_PATH"
  --sidecar-path "$SIDECAR_PATH"
  --run-root "$RUN_ROOT"
  --experiment-commit "$actual_commit"
  --git-branch "$(git branch --show-current)"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  arguments+=(--resume)
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-unavailable} commit=$actual_commit"
echo "resource_profile=$RESOURCE_PROFILE"
echo "requested_nodelist=$REQUESTED_NODELIST"
echo "slurm_job_partition=${SLURM_JOB_PARTITION:-unavailable}"
echo "slurm_job_qos=${SLURM_JOB_QOS:-unavailable}"
echo "slurm_timelimit=${SLURM_TIMELIMIT:-unavailable}"
taskset -pc $$ 2>&1 || true
if [[ "$(hostname -s)" != "$REQUESTED_NODELIST" ]]; then
  echo "node mismatch: requested $REQUESTED_NODELIST, got $(hostname -s)" >&2
  exit 2
fi
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -dd "$SLURM_JOB_ID" || true
fi

allowed_cpu_list="$(awk '$1 == "Cpus_allowed_list:" {print $2}' /proc/self/status)"
logical_cpu="${allowed_cpu_list%%,*}"
logical_cpu="${logical_cpu%%-*}"
if [[ ! "$logical_cpu" =~ ^[0-9]+$ ]]; then
  echo "unable to select one logical CPU from affinity list: $allowed_cpu_list" >&2
  exit 1
fi
echo "selected_logical_cpu=$logical_cpu"

srun --ntasks=1 --cpus-per-task=1 --cpu-bind=threads --mem-bind=local \
  taskset --cpu-list "$logical_cpu" \
  "$ANALYSIS_PYTHON" "$script_dir/run_quality_matrix.py" "${arguments[@]}"
