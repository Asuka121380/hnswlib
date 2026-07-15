#!/usr/bin/env bash
#SBATCH --job-name=hnsw_analyze
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
: "${REPO_ROOT:?}" "${RUN_DIR:?}" "${PYTHON_ENV:?}"
source "$PYTHON_ENV/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$RUN_DIR/aggregated" "$RUN_DIR/analysis"
python "$REPO_ROOT/scripts/baseline_trace/aggregate_shards.py" \
  --run-dir "$RUN_DIR" --expected-queries "$QUERY_COUNT" \
  --expected-shards "$TRACE_SHARDS" --output-dir "$RUN_DIR/aggregated"
python "$REPO_ROOT/scripts/baseline_trace/analyze_real_trace.py" \
  --input-dir "$RUN_DIR/aggregated" --output-dir "$RUN_DIR/analysis"
echo "aggregate_analyze_ok"
