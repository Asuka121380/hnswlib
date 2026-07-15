#!/usr/bin/env bash
#SBATCH --job-name=hnsw_capacity
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
: "${REPO_ROOT:?}" "${RUN_DIR:?}" "${PYTHON_ENV:?}"
source "$PYTHON_ENV/bin/activate"
python "$REPO_ROOT/scripts/baseline_trace/estimate_trace_capacity.py" \
  --pilot-dir "$RUN_DIR/pilot" --target-queries "$QUERY_COUNT" \
  --budget-gb "$STORAGE_BUDGET_GB" --output "$RUN_DIR/capacity.json"
echo "capacity_gate_ok"
