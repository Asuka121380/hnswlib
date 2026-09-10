#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: submit_quality_experiment.sh --config PATH [--run-root PATH] [--resume] [--dry-run]
EOF
}

config_path=""
run_root=""
resume=0
dry_run=0
while (( $# > 0 )); do
  case "$1" in
    --config) config_path="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --resume) resume=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$config_path" ]] || { usage; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
if [[ "$config_path" != /* ]]; then
  config_path="$repo_root/$config_path"
fi
[[ -f "$config_path" ]] || { echo "Missing configuration: $config_path" >&2; exit 2; }
config_path="$(cd -- "$(dirname -- "$config_path")" && pwd -P)/$(basename -- "$config_path")"
config_relative="${config_path#"$repo_root"/}"
git -C "$repo_root" ls-files --error-unmatch -- "$config_relative" >/dev/null

actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
branch="$(git -C "$repo_root" branch --show-current)"
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Repository worktree is dirty; commit changes before submission" >&2
  exit 2
fi
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
if [[ -n "${REFERENCE_BUILD:-}" ]]; then
  reference_build="$REFERENCE_BUILD"
elif [[ -x "$repo_root/build-v0-approx-reference-15820d5/v0_search_runner" ]]; then
  reference_build="$repo_root/build-v0-approx-reference-15820d5"
else
  reference_build="$repo_root/build-v0-approx-reference"
fi
dataset_config="${DATASET_CONFIG:-$HOME/IndividualProject/datasets/gist1m/dataset.json}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"

for required in "$analysis_python" "$reference_build/v0_search_runner" \
    "$reference_build/CMakeCache.txt" "$dataset_config" "$index_path" "$sidecar_path"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done
"$analysis_python" "$script_dir/qps_config.py" --config "$config_path"

if [[ -z "$run_root" ]]; then
  run_root="$HOME/IndividualProject/results/v0_performance_ready/quality-${short_commit}-$(date +%Y%m%d-%H%M%S)"
fi
[[ "$run_root" == /* ]] || { echo "Run root must be absolute" >&2; exit 2; }
if (( resume == 1 )); then
  [[ -f "$run_root/quality_manifest.json" ]] || {
    echo "Resume requires $run_root/quality_manifest.json" >&2; exit 2; }
elif [[ -e "$run_root" ]]; then
  echo "Run root already exists: $run_root" >&2
  exit 2
fi

echo "commit=$actual_commit branch=$branch"
echo "config=$config_path"
echo "run_root=$run_root"
echo "reference_build=$reference_build"
if (( dry_run == 1 )); then
  echo "dry_run_ok"
  exit 0
fi
command -v sbatch >/dev/null || { echo "Missing sbatch" >&2; exit 2; }
mkdir -p "$run_root/logs"
export REPO_ROOT="$repo_root" RUN_ROOT="$run_root" CONFIG_PATH="$config_path"
export EXPECTED_COMMIT="$actual_commit" REFERENCE_BUILD="$reference_build"
export ANALYSIS_PYTHON="$analysis_python" DATASET_CONFIG="$dataset_config"
export INDEX_PATH="$index_path" SIDECAR_PATH="$sidecar_path" RESUME="$resume"
job_id="$(sbatch --parsable --output="$run_root/logs/quality_%j.out" \
  --error="$run_root/logs/quality_%j.err" "$script_dir/run_quality_matrix.slurm")"
printf 'git_commit=%q\njob_id=%q\nrun_root=%q\nconfig_path=%q\n' \
  "$actual_commit" "$job_id" "$run_root" "$config_path" > "$run_root/submission.env"
echo "QUALITY_JOB=$job_id"
echo "RUN_ROOT=$run_root"
echo "MONITOR: squeue -j $job_id"
