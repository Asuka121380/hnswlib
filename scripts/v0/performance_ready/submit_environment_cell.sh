#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: submit_environment_cell.sh --platform LABEL --build-variant portable|native
       --resource-profile formal-exclusive|exploratory-shared
       --partition NAME --qos NAME --nodelist NAME
       --time-limit LIMIT --memory SIZE [--config PATH]
       [--run-root PATH] [--dry-run]

The job configures and builds both timing targets on the selected compute node,
then runs the frozen primary QPS pair and the P0.3 component microbenchmark.
EOF
}

platform_label=""
build_variant=""
resource_profile=""
partition=""
qos=""
nodelist=""
time_limit=""
memory=""
config_path="configs/v0/qps/p1_environment_primary.json"
run_root=""
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --platform) platform_label="${2:-}"; shift 2 ;;
    --build-variant) build_variant="${2:-}"; shift 2 ;;
    --resource-profile) resource_profile="${2:-}"; shift 2 ;;
    --partition) partition="${2:-}"; shift 2 ;;
    --qos) qos="${2:-}"; shift 2 ;;
    --nodelist) nodelist="${2:-}"; shift 2 ;;
    --time-limit) time_limit="${2:-}"; shift 2 ;;
    --memory) memory="${2:-}"; shift 2 ;;
    --config) config_path="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$platform_label" && -n "$build_variant" &&
   -n "$resource_profile" && -n "$partition" && -n "$qos" &&
   -n "$nodelist" && -n "$time_limit" && -n "$memory" ]] || {
  usage; exit 2; }
case "$build_variant" in portable|native) ;; *) echo "Invalid build variant" >&2; exit 2 ;; esac
case "$resource_profile" in
  formal-exclusive|exploratory-shared) ;;
  *) echo "Invalid resource profile" >&2; exit 2 ;;
esac
[[ "$platform_label" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Platform label must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
if [[ "$config_path" != /* ]]; then config_path="$repo_root/$config_path"; fi
[[ -f "$config_path" ]] || { echo "Missing config: $config_path" >&2; exit 2; }
config_path="$(cd -- "$(dirname -- "$config_path")" && pwd -P)/$(basename -- "$config_path")"
config_relative="${config_path#"$repo_root"/}"
git -C "$repo_root" ls-files --error-unmatch -- "$config_relative" >/dev/null 2>&1 || {
  echo "Configuration must be tracked by Git: $config_relative" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || {
  echo "Repository worktree is dirty; commit changes before submission" >&2; exit 2; }

actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
branch="$(git -C "$repo_root" branch --show-current)"
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
cmake_bin="${CMAKE_BIN:-/opt/apps/eb/software/CMake/3.20.1-GCCcore-10.3.0/bin/cmake}"
ninja_bin="${NINJA_BIN:-/opt/apps/eb/software/Ninja/1.10.2-GCCcore-10.3.0/bin/ninja}"
cxx_bin="${CXX_BIN:-/usr/bin/c++}"
openssl_lib="${OPENSSL_LIB:-/opt/apps/eb/software/CUDA/11.3.1/nsight-compute-2021.1.1/host/linux-desktop-glibc_2_11_3-x64}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
query_path="${QUERY_PATH:-$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs}"
for required in "$analysis_python" "$cmake_bin" "$ninja_bin" "$cxx_bin" \
    "$openssl_lib/libcrypto.so.1.1" "$index_path" "$sidecar_path" "$query_path"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done
"$analysis_python" "$script_dir/qps_config.py" \
  --config "$config_path" --resource-profile "$resource_profile"

if [[ -z "$run_root" ]]; then
  run_root="$HOME/IndividualProject/results/v0_performance_ready/p1-env-${platform_label}-${build_variant}-${short_commit}-$(date +%Y%m%d-%H%M%S)"
fi
[[ "$run_root" == /* ]] || { echo "Run root must be absolute" >&2; exit 2; }
[[ ! -e "$run_root" ]] || { echo "Run root already exists: $run_root" >&2; exit 2; }
build_dir="${PERFORMANCE_BUILD:-$repo_root/build-v0-performance-${build_variant}-${nodelist}-${short_commit}}"

sbatch_args=(--parsable --job-name="p1-${platform_label}-${build_variant}" \
  --partition="$partition" --qos="$qos" --nodelist="$nodelist" \
  --cpus-per-task=1 --mem="$memory" --time="$time_limit" --export=ALL)
if [[ "$resource_profile" == "formal-exclusive" ]]; then sbatch_args+=(--exclusive); fi

echo "===== P1 ENVIRONMENT CELL ====="
printf '%s=%s\n' platform "$platform_label" build_variant "$build_variant" \
  branch "$branch" commit "$actual_commit" resource_profile "$resource_profile" \
  partition "$partition" qos "$qos" nodelist "$nodelist" \
  run_root "$run_root" build_dir "$build_dir" config "$config_path"
printf 'sbatch_arguments='; printf ' %q' "${sbatch_args[@]}"; printf '\n'
if (( dry_run == 1 )); then echo "dry_run_ok"; exit 0; fi
command -v sbatch >/dev/null || { echo "Missing sbatch" >&2; exit 2; }

mkdir -p "$run_root/logs"
export REPO_ROOT="$repo_root" RUN_ROOT="$run_root" CONFIG_PATH="$config_path"
export PLATFORM_LABEL="$platform_label" BUILD_VARIANT="$build_variant"
export RESOURCE_PROFILE="$resource_profile" REQUESTED_NODELIST="$nodelist"
export EXPECTED_COMMIT="$actual_commit" PERFORMANCE_BUILD="$build_dir"
export ANALYSIS_PYTHON="$analysis_python" CMAKE_BIN="$cmake_bin"
export NINJA_BIN="$ninja_bin" CXX_BIN="$cxx_bin" OPENSSL_LIB="$openssl_lib"
export INDEX_PATH="$index_path" SIDECAR_PATH="$sidecar_path" QUERY_PATH="$query_path"
job_id="$(env -u LD_LIBRARY_PATH sbatch "${sbatch_args[@]}" \
  --output="$run_root/logs/cell_%j.out" --error="$run_root/logs/cell_%j.err" \
  "$script_dir/run_environment_cell.slurm")"
printf 'schema_version=%q\nplatform=%q\nbuild_variant=%q\ngit_commit=%q\ngit_branch=%q\njob_id=%q\nrun_root=%q\nbuild_dir=%q\nresource_profile=%q\npartition=%q\nqos=%q\nnodelist=%q\ntime_limit=%q\nmemory=%q\n' \
  'p1-environment-cell-v1' "$platform_label" "$build_variant" "$actual_commit" \
  "$branch" "$job_id" "$run_root" "$build_dir" "$resource_profile" \
  "$partition" "$qos" "$nodelist" "$time_limit" "$memory" > "$run_root/submission.env"
echo "ENVIRONMENT_CELL_JOB=$job_id"
echo "RUN_ROOT=$run_root"
echo "MONITOR: squeue -j $job_id"
