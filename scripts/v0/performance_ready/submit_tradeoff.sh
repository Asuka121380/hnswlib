#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 validation|heldout [RUN_ROOT]" >&2
}

if (( $# < 1 || $# > 2 )); then
  usage
  exit 2
fi

split="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/tradeoff_matrix.sh"
configure_tradeoff_split "$split" || {
  usage
  exit 2
}

for command_name in git sbatch; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 2
  fi
done

repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
branch="$(git -C "$repo_root" branch --show-current)"

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Repository worktree is dirty; commit or remove local changes first" >&2
  git -C "$repo_root" status --short >&2
  exit 2
fi

query_end=$((TRADEOFF_QUERY_START + TRADEOFF_QUERY_COUNT - 1))
default_root="$HOME/IndividualProject/results/v0_performance_ready/${split}-q${TRADEOFF_QUERY_START}-${query_end}-grid-${short_commit}-v1"
run_root="${2:-$default_root}"
reference_build="${REFERENCE_BUILD:-$repo_root/build-v0-approx-reference-15820d5}"
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
dataset_config="${DATASET_CONFIG:-$HOME/IndividualProject/datasets/gist1m/dataset.json}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"

if [[ -e "$run_root" ]]; then
  echo "Run root already exists: $run_root" >&2
  echo "Pass a new explicit RUN_ROOT for another attempt" >&2
  exit 2
fi

mkdir -p "$run_root/logs"

export REPO_ROOT="$repo_root"
export RUN_ROOT="$run_root"
export EXPECTED_COMMIT="$actual_commit"
export TRADEOFF_SPLIT="$split"
export REFERENCE_BUILD="$reference_build"
export ANALYSIS_PYTHON="$analysis_python"
export DATASET_CONFIG="$dataset_config"
export INDEX_PATH="$index_path"
export SIDECAR_PATH="$sidecar_path"

job_id="$(
  sbatch --parsable \
    --job-name="v0-fast-${split}-grid" \
    --output="$run_root/logs/${split}_%j.out" \
    --error="$run_root/logs/${split}_%j.err" \
    --export=ALL \
    "$script_dir/run_tradeoff.slurm"
)"

manifest="$run_root/submission.env"
{
  printf 'schema_version=%q\n' 'v0-tradeoff-v1'
  printf 'submitted_at=%q\n' "$(date --iso-8601=seconds)"
  printf 'slurm_job_id=%q\n' "$job_id"
  printf 'git_commit=%q\n' "$actual_commit"
  printf 'git_branch=%q\n' "$branch"
  printf 'tradeoff_split=%q\n' "$split"
  printf 'query_start=%q\n' "$TRADEOFF_QUERY_START"
  printf 'query_count=%q\n' "$TRADEOFF_QUERY_COUNT"
  printf 'query_end=%q\n' "$query_end"
  printf 'betas=%q\n' "$TRADEOFF_BETAS"
  printf 'modes=%q\n' "$TRADEOFF_MODES"
  printf 'k=%q\n' "$TRADEOFF_K"
  printf 'ef_search=%q\n' "$TRADEOFF_EF_SEARCH"
  printf 'reference_build=%q\n' "$reference_build"
  printf 'analysis_python=%q\n' "$analysis_python"
  printf 'dataset_config=%q\n' "$dataset_config"
  printf 'index_path=%q\n' "$index_path"
  printf 'sidecar_path=%q\n' "$sidecar_path"
  printf 'run_root=%q\n' "$run_root"
  printf 'slurm_script=%q\n' "$script_dir/run_tradeoff.slurm"
} > "$manifest"

echo "TRADEOFF_JOB=$job_id"
echo "RUN_ROOT=$run_root"
echo "MANIFEST=$manifest"
echo "MONITOR: squeue -j $job_id"
echo "LOG: tail -f $run_root/logs/${split}_${job_id}.out"
