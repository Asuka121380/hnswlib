#!/usr/bin/env bash
#SBATCH --job-name=hnsw_perf
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${BUILD_OFF:?}" "${DATASET_CONFIG:?}" "${INDEX_PATH:?}" "${RUN_DIR:?}"
OUT="$RUN_DIR/performance/ef${EF_SEARCH}"
mkdir -p "$OUT"
"$BUILD_OFF/real_data_trace_runner" \
  --dataset-config "$DATASET_CONFIG" --mode performance \
  --index-path "$INDEX_PATH" --output-dir "$OUT" --run-id "$RUN_ID" \
  --query-start "$QUERY_START" --query-count "$QUERY_COUNT" \
  --k "$K" --ef-search "$EF_SEARCH" --M "$M" \
  --ef-construction "$EF_CONSTRUCTION" --seed "$SEED" --threads 1 \
  --warmup-queries "$WARMUP_QUERIES" --performance-repeats "$PERFORMANCE_REPEATS"
echo "performance_baseline_ok"
