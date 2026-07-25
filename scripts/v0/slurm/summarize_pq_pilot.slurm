#!/usr/bin/env bash
#SBATCH --job-name=v0-pq-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=v0-pq-summary-%j.out

set -euo pipefail

: "${V0_REPO_ROOT:?set V0_REPO_ROOT}"
: "${V0_PYTHON:?set V0_PYTHON}"
: "${V0_OUTPUT_ROOT:?set V0_OUTPUT_ROOT}"

command=(
  "${V0_PYTHON}" "${V0_REPO_ROOT}/scripts/v0/analyze_pq_pilot.py"
  --metrics "${V0_OUTPUT_ROOT}/pq_m8/training_metrics.json"
  --metrics "${V0_OUTPUT_ROOT}/pq_m16/training_metrics.json"
  --metrics "${V0_OUTPUT_ROOT}/pq_m32/training_metrics.json"
  --output-dir "${V0_OUTPUT_ROOT}/summary"
)
if [[ -n "${V0_MAX_P99_ERROR:-}" ]]; then
  command+=(--max-p99-error "${V0_MAX_P99_ERROR}")
fi
"${command[@]}"
