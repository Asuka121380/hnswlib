#!/usr/bin/env bash
#SBATCH --job-name=v0-quality-matrix
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
: "${CONFIG_PATH:?CONFIG_PATH is required}"
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
"$ANALYSIS_PYTHON" "$script_dir/qps_config.py" --config "$CONFIG_PATH"
"$ANALYSIS_PYTHON" "$script_dir/check_build_contract.py" \
  --build-dir "$REFERENCE_BUILD" --contract reference

arguments=(
  --config "$CONFIG_PATH"
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
"$ANALYSIS_PYTHON" "$script_dir/run_quality_matrix.py" "${arguments[@]}"
