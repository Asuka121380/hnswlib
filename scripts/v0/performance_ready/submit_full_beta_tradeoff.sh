#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [RUN_ROOT]" >&2
}

if (( $# > 1 )); then
  usage
  exit 2
fi

for command_name in git sbatch; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 2
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/full_beta_tradeoff_matrix.sh"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
branch="$(git -C "$repo_root" branch --show-current)"

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Repository worktree is dirty; commit or remove local changes first" >&2
  git -C "$repo_root" status --short >&2
  exit 2
fi

run_root="${1:-$HOME/IndividualProject/results/v0_performance_ready/full-beta-tradeoff-q0-999-${short_commit}-v1}"
reference_build="${REFERENCE_BUILD:-$repo_root/build-v0-approx-reference-15820d5}"
if [[ -n "${PERFORMANCE_BUILD:-}" ]]; then
  performance_build="$PERFORMANCE_BUILD"
elif [[ -x "$repo_root/build-v0-performance-ready-15820d5/v0_performance_runner" ]]; then
  performance_build="$repo_root/build-v0-performance-ready-15820d5"
else
  performance_build="$repo_root/build-v0-performance-ready"
fi
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
dataset_config="${DATASET_CONFIG:-$HOME/IndividualProject/datasets/gist1m/dataset.json}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
query_path="${QUERY_PATH:-$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs}"

if [[ -e "$run_root" ]]; then
  echo "Run root already exists: $run_root" >&2
  echo "Pass a new explicit RUN_ROOT for another attempt" >&2
  exit 2
fi
for required in \
    "$reference_build/v0_search_runner" \
    "$reference_build/CMakeCache.txt" \
    "$performance_build/v0_performance_runner" \
    "$performance_build/CMakeCache.txt" \
    "$analysis_python" "$dataset_config" "$index_path" \
    "$sidecar_path" "$query_path"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

mkdir -p "$run_root/logs"

export REPO_ROOT="$repo_root"
export RUN_ROOT="$run_root"
export EXPECTED_COMMIT="$actual_commit"
export REFERENCE_BUILD="$reference_build"
export PERFORMANCE_BUILD="$performance_build"
export ANALYSIS_PYTHON="$analysis_python"
export DATASET_CONFIG="$dataset_config"
export INDEX_PATH="$index_path"
export SIDECAR_PATH="$sidecar_path"
export QUERY_PATH="$query_path"
export NUMA_POLICY="${NUMA_POLICY:-slurm-exclusive-single-task}"
export POWER_POLICY="${POWER_POLICY:-cluster-default-unverified}"
export CPU_FREQUENCY_POLICY="${CPU_FREQUENCY_POLICY:-cluster-default-unverified}"

metrics_job="$(
  sbatch --parsable \
    --output="$run_root/logs/metrics_%j.out" \
    --error="$run_root/logs/metrics_%j.err" \
    --export=ALL \
    "$script_dir/run_full_beta_metrics.slurm"
)"
qps_job="$(
  sbatch --parsable \
    --output="$run_root/logs/qps_%j.out" \
    --error="$run_root/logs/qps_%j.err" \
    --export=ALL \
    "$script_dir/run_full_beta_qps.slurm"
)"
summary_job="$(
  sbatch --parsable \
    --dependency="afterok:${metrics_job}:${qps_job}" \
    --output="$run_root/logs/summary_%j.out" \
    --error="$run_root/logs/summary_%j.err" \
    --export=ALL \
    "$script_dir/run_full_beta_summary.slurm"
)"

manifest="$run_root/submission.env"
{
  printf 'schema_version=%q\n' 'v0-full-beta-tradeoff-v1'
  printf 'submitted_at=%q\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%q\n' "$actual_commit"
  printf 'git_branch=%q\n' "$branch"
  printf 'metrics_job_id=%q\n' "$metrics_job"
  printf 'qps_job_id=%q\n' "$qps_job"
  printf 'summary_job_id=%q\n' "$summary_job"
  printf 'run_root=%q\n' "$run_root"
  printf 'betas=%q\n' "$FULL_BETA_VALUES"
  printf 'modes=%q\n' "$FULL_BETA_MODES"
  printf 'prefetches=%q\n' "$FULL_BETA_PREFETCHES"
  printf 'query_start=%q\n' "$FULL_BETA_QUERY_START"
  printf 'query_count=%q\n' "$FULL_BETA_QUERY_COUNT"
  printf 'k=%q\n' "$FULL_BETA_K"
  printf 'ef_search=%q\n' "$FULL_BETA_EF_SEARCH"
  printf 'blocks=%q\n' "$FULL_BETA_BLOCKS"
  printf 'within_process_repeats=%q\n' "$FULL_BETA_WITHIN_PROCESS_REPEATS"
  printf 'seed=%q\n' "$FULL_BETA_SEED"
  printf 'reference_build=%q\n' "$reference_build"
  printf 'performance_build=%q\n' "$performance_build"
  printf 'index_path=%q\n' "$index_path"
  printf 'sidecar_path=%q\n' "$sidecar_path"
  printf 'query_path=%q\n' "$query_path"
  printf 'numa_policy=%q\n' "$NUMA_POLICY"
  printf 'power_policy=%q\n' "$POWER_POLICY"
  printf 'cpu_frequency_policy=%q\n' "$CPU_FREQUENCY_POLICY"
} > "$manifest"

echo "METRICS_JOB=$metrics_job"
echo "QPS_JOB=$qps_job"
echo "SUMMARY_JOB=$summary_job"
echo "RUN_ROOT=$run_root"
echo "MONITOR: squeue -j ${metrics_job},${qps_job},${summary_job}"
echo "METRICS_LOG: tail -f $run_root/logs/metrics_${metrics_job}.out"
echo "QPS_LOG: tail -f $run_root/logs/qps_${qps_job}.out"
