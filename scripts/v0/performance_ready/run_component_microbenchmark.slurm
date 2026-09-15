#!/usr/bin/env bash
#SBATCH --job-name=v0-p0-component
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

# Site resource selection is supplied by submit_component_microbenchmark.sh.

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RESOURCE_PROFILE:?RESOURCE_PROFILE is required}"
: "${REQUESTED_NODELIST:?REQUESTED_NODELIST is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PERFORMANCE_BUILD:?PERFORMANCE_BUILD is required}"
: "${ANALYSIS_PYTHON:?ANALYSIS_PYTHON is required}"
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
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "LD_LIBRARY_PATH must be unset for the portable P0 environment" >&2
  exit 2
fi

runner="$PERFORMANCE_BUILD/v0_fast_kernel_microbenchmark"
for required in "$runner" "$PERFORMANCE_BUILD/CMakeCache.txt" \
    "$ANALYSIS_PYTHON" "$SIDECAR_PATH"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done
script_dir="$REPO_ROOT/scripts/v0/performance_ready"
"$ANALYSIS_PYTHON" "$script_dir/check_build_contract.py" \
  --build-dir "$PERFORMANCE_BUILD" --contract performance

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-unavailable} commit=$actual_commit"
echo "resource_profile=$RESOURCE_PROFILE"
echo "requested_nodelist=$REQUESTED_NODELIST"
echo "slurm_job_partition=${SLURM_JOB_PARTITION:-unavailable}"
echo "slurm_job_qos=${SLURM_JOB_QOS:-unavailable}"
echo "slurm_job_account=${SLURM_JOB_ACCOUNT:-unavailable}"
echo "slurm_timelimit=${SLURM_TIMELIMIT:-unavailable}"
echo "ld_library_path=unset"
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
  "$ANALYSIS_PYTHON" "$script_dir/run_component_microbenchmark.py" \
    --runner "$runner" \
    --cmake-cache "$PERFORMANCE_BUILD/CMakeCache.txt" \
    --sidecar-path "$SIDECAR_PATH" \
    --run-root "$RUN_ROOT" \
    --resource-profile "$RESOURCE_PROFILE" \
    --requested-node "$REQUESTED_NODELIST" \
    --experiment-commit "$actual_commit" \
    --git-branch "$(git branch --show-current)" \
    --blocks 7 --iterations 100000 --warmup-iterations 10000 \
    --working-set 64
