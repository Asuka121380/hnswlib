#!/usr/bin/env bash
#SBATCH --job-name=hnsw_pilot
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${BUILD_TRACE:?}" "${DATASET_CONFIG:?}" "${INDEX_PATH:?}" "${RUN_DIR:?}"
OUT="$RUN_DIR/pilot"
mkdir -p "$OUT"
"$BUILD_TRACE/real_data_trace_runner" \
  --dataset-config "$DATASET_CONFIG" --mode trace \
  --index-path "$INDEX_PATH" --output-dir "$OUT" --run-id "$RUN_ID" \
  --query-start "$QUERY_START" --query-count "$PILOT_QUERY_COUNT" \
  --k "$K" --ef-search "$EF_SEARCH" --M "$M" \
  --ef-construction "$EF_CONSTRUCTION" --seed "$SEED" --threads 1 \
  --dco-sample-modulus "$DCO_SAMPLE_MODULUS" --dco-sample-remainder 0 \
  --edge-samples 256 --collect-geometry true --collect-distance-timing false
echo "trace_pilot_ok"
