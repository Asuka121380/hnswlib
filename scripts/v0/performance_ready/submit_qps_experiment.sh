#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_qps_experiment.sh --config PATH --resource-profile PROFILE [options]

Required:
  --config PATH
  --resource-profile formal-exclusive|exploratory-shared

Options:
  --run-root PATH   Explicit output root.
  --resume          Resume an existing root with an identical contract.
  --dry-run         Validate and print the resolved contract without sbatch.
EOF
}

config_path=""
resource_profile=""
run_root=""
resume=0
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      config_path="$2"
      shift 2
      ;;
    --resource-profile)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      resource_profile="$2"
      shift 2
      ;;
    --run-root)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      run_root="$2"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$config_path" || -z "$resource_profile" ]]; then
  usage
  exit 2
fi
case "$resource_profile" in
  formal-exclusive|exploratory-shared) ;;
  *)
    echo "Unknown resource profile: $resource_profile" >&2
    exit 2
    ;;
esac

for command_name in git; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 2
  fi
done
if (( dry_run == 0 )) && ! command -v sbatch >/dev/null 2>&1; then
  echo "Missing required command: sbatch" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
if [[ "$config_path" != /* ]]; then
  config_path="$repo_root/$config_path"
fi
if [[ ! -f "$config_path" ]]; then
  echo "Missing configuration: $config_path" >&2
  exit 2
fi
config_path="$(cd -- "$(dirname -- "$config_path")" && pwd -P)/$(basename -- "$config_path")"
case "$config_path" in
  "$repo_root"/*) ;;
  *)
    echo "Configuration must be inside the repository: $config_path" >&2
    exit 2
    ;;
esac
config_relative="${config_path#"$repo_root"/}"
if ! git -C "$repo_root" ls-files --error-unmatch -- \
    "$config_relative" >/dev/null 2>&1; then
  echo "Configuration must be tracked by Git: $config_relative" >&2
  exit 2
fi

actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
branch="$(git -C "$repo_root" branch --show-current)"
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
if [[ ! -x "$analysis_python" ]]; then
  echo "Missing analysis Python: $analysis_python" >&2
  exit 2
fi

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Repository worktree is dirty; commit changes before submission" >&2
  git -C "$repo_root" status --short >&2
  exit 2
fi

echo "===== RESOLVED QPS CONTRACT ====="
"$analysis_python" "$script_dir/qps_config.py" \
  --config "$config_path" \
  --resource-profile "$resource_profile"

if [[ -z "$run_root" ]]; then
  timestamp="$(date +%Y%m%d-%H%M%S)"
  run_root="$HOME/IndividualProject/results/v0_performance_ready/qps-${short_commit}-${timestamp}"
fi
if [[ "$run_root" != /* ]]; then
  echo "Run root must be an absolute path: $run_root" >&2
  exit 2
fi

if (( resume == 1 )); then
  if [[ ! -f "$run_root/manifest.json" ]]; then
    echo "Resume requires an existing manifest: $run_root/manifest.json" >&2
    exit 2
  fi
elif [[ -e "$run_root" ]]; then
  echo "Run root already exists: $run_root" >&2
  exit 2
fi

if [[ -n "${PERFORMANCE_BUILD:-}" ]]; then
  performance_build="$PERFORMANCE_BUILD"
elif [[ -x "$repo_root/build-v0-performance-ready-15820d5/v0_performance_runner" ]]; then
  performance_build="$repo_root/build-v0-performance-ready-15820d5"
else
  performance_build="$repo_root/build-v0-performance-ready"
fi
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
query_path="${QUERY_PATH:-$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs}"

for required in \
    "$performance_build/v0_performance_runner" \
    "$performance_build/CMakeCache.txt" \
    "$index_path" "$sidecar_path" "$query_path"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

sbatch_args=(
  --parsable
  --job-name="v0-qps-${resource_profile}"
  --cpus-per-task=1
  --mem=32G
  --time=08:00:00
  --export=ALL
)
if [[ "$resource_profile" == "formal-exclusive" ]]; then
  sbatch_args+=(--exclusive)
else
  sbatch_args+=(--oversubscribe)
fi

echo "===== SUBMISSION PREVIEW ====="
echo "branch=$branch"
echo "commit=$actual_commit"
echo "config=$config_path"
echo "resource_profile=$resource_profile"
echo "run_root=$run_root"
echo "performance_build=$performance_build"
printf 'sbatch_arguments='
printf ' %q' "${sbatch_args[@]}"
printf '\n'
if (( dry_run == 1 )); then
  echo "dry_run_ok"
  exit 0
fi

mkdir -p "$run_root/logs"
export REPO_ROOT="$repo_root"
export RUN_ROOT="$run_root"
export CONFIG_PATH="$config_path"
export RESOURCE_PROFILE="$resource_profile"
export EXPECTED_COMMIT="$actual_commit"
export PERFORMANCE_BUILD="$performance_build"
export ANALYSIS_PYTHON="$analysis_python"
export INDEX_PATH="$index_path"
export SIDECAR_PATH="$sidecar_path"
export QUERY_PATH="$query_path"
export NUMA_POLICY="${NUMA_POLICY:-slurm-single-task}"
export POWER_POLICY="${POWER_POLICY:-cluster-default-unverified}"
export CPU_FREQUENCY_POLICY="${CPU_FREQUENCY_POLICY:-cluster-default-unverified}"
export RESUME="$resume"

job_id="$(
  sbatch "${sbatch_args[@]}" \
    --output="$run_root/logs/qps_%j.out" \
    --error="$run_root/logs/qps_%j.err" \
    "$script_dir/run_qps_matrix.slurm"
)"

manifest="$run_root/submission.env"
{
  printf 'schema_version=%q\n' 'v0-configurable-qps-v1'
  printf 'submitted_at=%q\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%q\n' "$actual_commit"
  printf 'git_branch=%q\n' "$branch"
  printf 'job_id=%q\n' "$job_id"
  printf 'run_root=%q\n' "$run_root"
  printf 'config_path=%q\n' "$config_path"
  printf 'resource_profile=%q\n' "$resource_profile"
  printf 'resume=%q\n' "$resume"
  printf 'performance_build=%q\n' "$performance_build"
  printf 'index_path=%q\n' "$index_path"
  printf 'sidecar_path=%q\n' "$sidecar_path"
  printf 'query_path=%q\n' "$query_path"
} > "$manifest"

echo "QPS_JOB=$job_id"
echo "RUN_ROOT=$run_root"
echo "MONITOR: squeue -j $job_id"
echo "LOG: tail -f $run_root/logs/qps_${job_id}.out"
