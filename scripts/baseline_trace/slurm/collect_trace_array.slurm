#!/usr/bin/env bash
#SBATCH --job-name=hnsw_trace
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
: "${BUILD_TRACE:?}" "${DATASET_CONFIG:?}" "${INDEX_PATH:?}" "${RUN_DIR:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

BASE=$((QUERY_COUNT / TRACE_SHARDS))
REMAINDER=$((QUERY_COUNT % TRACE_SHARDS))
TASK=$SLURM_ARRAY_TASK_ID
COUNT=$BASE
if (( TASK < REMAINDER )); then COUNT=$((COUNT + 1)); fi
START=$((QUERY_START + TASK * BASE + (TASK < REMAINDER ? TASK : REMAINDER)))
PART=$(printf 'part-%05d' "$TASK")
FINAL="$RUN_DIR/raw/shards/$PART"
TEMP="$RUN_DIR/raw/shards/.${PART}.tmp-${SLURM_JOB_ID}"
if [[ -f "$FINAL/complete.json" ]]; then
  echo "Shard already complete: $FINAL"
  exit 0
fi
mkdir -p "$TEMP"
"$BUILD_TRACE/real_data_trace_runner" \
  --dataset-config "$DATASET_CONFIG" --mode trace \
  --index-path "$INDEX_PATH" --output-dir "$TEMP" --run-id "$RUN_ID" \
  --query-start "$START" --query-count "$COUNT" \
  --k "$K" --ef-search "$EF_SEARCH" --M "$M" \
  --ef-construction "$EF_CONSTRUCTION" --seed "$SEED" --threads 1 \
  --dco-sample-modulus "$DCO_SAMPLE_MODULUS" --dco-sample-remainder 0 \
  --edge-samples "$EDGE_SAMPLES_PER_SHARD" \
  --collect-geometry true --collect-distance-timing false
mv "$TEMP" "$FINAL"
echo "collect_trace_shard_ok task=$TASK start=$START count=$COUNT"
