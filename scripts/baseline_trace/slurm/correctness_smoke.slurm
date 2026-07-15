#!/usr/bin/env bash
#SBATCH --job-name=hnsw_correct
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${BUILD_TRACE:?}" "${DATASET_CONFIG:?}" "${INDEX_PATH:?}" "${RUN_DIR:?}"
OUT="$RUN_DIR/correctness"
mkdir -p "$OUT"
"$BUILD_TRACE/real_data_trace_runner" \
  --dataset-config "$DATASET_CONFIG" --mode correctness \
  --index-path "$INDEX_PATH" --output-dir "$OUT" --run-id "$RUN_ID" \
  --query-start "$QUERY_START" --query-count "$CORRECTNESS_QUERY_COUNT" \
  --k "$K" --ef-search "$EF_SEARCH" --M "$M" \
  --ef-construction "$EF_CONSTRUCTION" --seed "$SEED" --threads 1 \
  --dco-sample-modulus 1 --dco-sample-remainder 0 \
  --edge-samples 128 --collect-geometry true --collect-distance-timing true
echo "correctness_smoke_ok"
