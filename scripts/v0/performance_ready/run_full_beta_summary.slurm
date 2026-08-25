#!/usr/bin/env bash
#SBATCH --job-name=v0-full-beta-summary
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${ANALYSIS_PYTHON:?ANALYSIS_PYTHON is required}"

cd "$REPO_ROOT"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Commit mismatch: expected $EXPECTED_COMMIT, got $actual_commit" >&2
  exit 2
fi

"$ANALYSIS_PYTHON" \
  scripts/v0/performance_ready/summarize_full_beta_tradeoff.py \
  --metrics-summary "$RUN_ROOT/metrics/full-tradeoff_summary.csv" \
  --qps-dir "$RUN_ROOT/qps" \
  --output "$RUN_ROOT/full_beta_tradeoff.csv" \
  --report "$RUN_ROOT/full_beta_tradeoff_report.json"

echo "v0_full_beta_summary_ok"
