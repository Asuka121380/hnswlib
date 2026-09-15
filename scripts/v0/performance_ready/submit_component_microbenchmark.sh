#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: submit_component_microbenchmark.sh --resource-profile PROFILE
       --partition NAME --qos NAME --nodelist NAME
       --time-limit LIMIT --memory SIZE [--run-root PATH] [--dry-run]
EOF
}

resource_profile=""
partition=""
qos=""
nodelist=""
time_limit=""
memory=""
run_root=""
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --resource-profile) resource_profile="${2:-}"; shift 2 ;;
    --partition) partition="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --nodelist) nodelist="${2:-}"; shift 2 ;;
    --time-limit) time_limit="${2:-}"; shift 2 ;;
    --memory) memory="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$resource_profile" && -n "$partition" && -n "$qos" &&
   -n "$nodelist" && -n "$time_limit" && -n "$memory" ]] || {
  usage; exit 2; }
case "$resource_profile" in
  formal-exclusive|exploratory-shared) ;;
  *) echo "Unknown resource profile: $resource_profile" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
branch="$(git -C "$repo_root" branch --show-current)"
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Repository worktree is dirty; commit changes before submission" >&2
  exit 2
fi

analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
performance_build="${PERFORMANCE_BUILD:-$repo_root/build-v0-performance-portable-${short_commit}}"
runner="$performance_build/v0_fast_kernel_microbenchmark"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
for required in "$analysis_python" "$runner" \
    "$performance_build/CMakeCache.txt" "$sidecar_path"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done
"$analysis_python" "$script_dir/check_build_contract.py" \
  --build-dir "$performance_build" --contract performance

if [[ -z "$run_root" ]]; then
  run_root="$HOME/IndividualProject/results/v0_performance_ready/p0-component-microbenchmark-${short_commit}-$(date +%Y%m%d-%H%M%S)"
fi
[[ "$run_root" == /* ]] || { echo "Run root must be absolute" >&2; exit 2; }
[[ ! -e "$run_root" ]] || { echo "Run root already exists: $run_root" >&2; exit 2; }

echo "commit=$actual_commit branch=$branch"
echo "resource_profile=$resource_profile"
echo "partition=$partition qos=$qos nodelist=$nodelist time_limit=$time_limit memory=$memory"
echo "run_root=$run_root"
echo "performance_build=$performance_build"
echo "runner=$runner"
echo "sidecar=$sidecar_path"
echo "blocks=7 iterations=100000 warmup_iterations=10000 working_set=64"
echo "submission_environment=LD_LIBRARY_PATH removed before sbatch"
if (( dry_run == 1 )); then
  echo "dry_run_ok"
  exit 0
fi

command -v sbatch >/dev/null || { echo "Missing sbatch" >&2; exit 2; }
mkdir -p "$run_root/logs"
export REPO_ROOT="$repo_root" RUN_ROOT="$run_root"
export RESOURCE_PROFILE="$resource_profile" REQUESTED_NODELIST="$nodelist"
export EXPECTED_COMMIT="$actual_commit" PERFORMANCE_BUILD="$performance_build"
export ANALYSIS_PYTHON="$analysis_python" SIDECAR_PATH="$sidecar_path"
sbatch_args=(--parsable --partition="$partition" --qos="$qos" \
  --nodelist="$nodelist" --cpus-per-task=1 --mem="$memory" \
  --time="$time_limit" --export=ALL \
  --output="$run_root/logs/component_%j.out" \
  --error="$run_root/logs/component_%j.err")
if [[ "$resource_profile" == "formal-exclusive" ]]; then
  sbatch_args+=(--exclusive)
fi
job_id="$(env -u LD_LIBRARY_PATH sbatch "${sbatch_args[@]}" \
  "$script_dir/run_component_microbenchmark.slurm")"
printf 'git_commit=%q\njob_id=%q\nrun_root=%q\nresource_profile=%q\npartition=%q\nqos=%q\nnodelist=%q\ntime_limit=%q\nmemory=%q\n' \
  "$actual_commit" "$job_id" "$run_root" "$resource_profile" \
  "$partition" "$qos" "$nodelist" "$time_limit" "$memory" \
  > "$run_root/submission.env"
echo "COMPONENT_JOB=$job_id"
echo "RUN_ROOT=$run_root"
echo "MONITOR: squeue -j $job_id"
