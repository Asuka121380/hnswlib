#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --dataset-config PATH [--experiment-config PATH] [--run-id ID]" >&2
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXPERIMENT_CONFIG="$REPO_ROOT/configs/baseline_trace/experiments/sift1m_first_full.json"
DATASET_CONFIG=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-config) DATASET_CONFIG=$2; shift 2 ;;
    --experiment-config) EXPERIMENT_CONFIG=$2; shift 2 ;;
    --run-id) RUN_ID=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -f "$DATASET_CONFIG" ]] || { echo "Dataset config not found: $DATASET_CONFIG" >&2; exit 1; }
[[ -f "$EXPERIMENT_CONFIG" ]] || { echo "Experiment config not found: $EXPERIMENT_CONFIG" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is not available" >&2; exit 1; }

DATASET_NAME=$(python3 - "$DATASET_CONFIG" "$EXPERIMENT_CONFIG" <<'PY'
import json
from pathlib import Path
import re
import sys

dataset_path = Path(sys.argv[1])
experiment_path = Path(sys.argv[2])
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
dataset_name = str(dataset.get("dataset", ""))
if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_name):
    raise SystemExit(f"Invalid dataset name: {dataset_name!r}")
if experiment.get("dataset") != dataset_name:
    raise SystemExit(
        f"Dataset mismatch: manifest={dataset_name!r}, experiment={experiment.get('dataset')!r}"
    )
required = [
    "k", "M", "ef_construction", "ef_search_values", "seed", "query_start", "query_count",
    "trace_shards", "dco_sample_modulus", "edge_samples_per_shard", "pilot_query_count",
    "correctness_query_count", "storage_budget_gb", "performance_repeats", "warmup_queries",
]
missing = [key for key in required if key not in experiment]
if missing:
    raise SystemExit(f"Experiment config is missing: {', '.join(missing)}")
query_start = int(experiment["query_start"])
query_count = int(experiment["query_count"])
n_query = int(dataset["n_query"])
k = int(experiment["k"])
ground_truth_k = int(dataset["ground_truth_k"])
shards = int(experiment["trace_shards"])
ef_values = [int(value) for value in experiment["ef_search_values"]]
if query_start < 0 or query_count <= 0 or query_start + query_count > n_query:
    raise SystemExit(
        f"Invalid query range [{query_start}, {query_start + query_count}) for n_query={n_query}"
    )
if k <= 0 or k > ground_truth_k:
    raise SystemExit(f"Invalid k={k} for ground_truth_k={ground_truth_k}")
if shards <= 0 or shards > query_count:
    raise SystemExit(f"Invalid trace_shards={shards} for query_count={query_count}")
if not ef_values or len(ef_values) != len(set(ef_values)) or any(value < k for value in ef_values):
    raise SystemExit(f"Invalid ef_search_values={ef_values} for k={k}")
for positive_key in (
    "M", "ef_construction", "dco_sample_modulus", "pilot_query_count",
    "correctness_query_count", "storage_budget_gb", "performance_repeats",
):
    if int(experiment[positive_key]) <= 0:
        raise SystemExit(f"{positive_key} must be positive")
if int(experiment["pilot_query_count"]) > query_count:
    raise SystemExit("pilot_query_count exceeds query_count")
if int(experiment["correctness_query_count"]) > query_count:
    raise SystemExit("correctness_query_count exceeds query_count")
print(dataset_name)
PY
)

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${DATASET_NAME}-first-full-$(date +%Y%m%dT%H%M%S)"
fi

json_value() {
  python3 - "$EXPERIMENT_CONFIG" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)[sys.argv[2]]
if isinstance(value, list):
    print(" ".join(map(str, value)))
else:
    print(value)
PY
}

K=$(json_value k)
M=$(json_value M)
EF_CONSTRUCTION=$(json_value ef_construction)
EF_SEARCH_VALUES=$(json_value ef_search_values)
SEED=$(json_value seed)
QUERY_START=$(json_value query_start)
QUERY_COUNT=$(json_value query_count)
TRACE_SHARDS=$(json_value trace_shards)
DCO_SAMPLE_MODULUS=$(json_value dco_sample_modulus)
EDGE_SAMPLES_PER_SHARD=$(json_value edge_samples_per_shard)
PILOT_QUERY_COUNT=$(json_value pilot_query_count)
CORRECTNESS_QUERY_COUNT=$(json_value correctness_query_count)
STORAGE_BUDGET_GB=$(json_value storage_budget_gb)
PERFORMANCE_REPEATS=$(json_value performance_repeats)
WARMUP_QUERIES=$(json_value warmup_queries)

MAX_EF=$(printf '%s\n' $EF_SEARCH_VALUES | sort -n | tail -n 1)
EF_SEARCH_COUNT=$(printf '%s\n' $EF_SEARCH_VALUES | wc -l | tr -d ' ')
GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
GIT_STATUS=$(git -C "$REPO_ROOT" status --short)
RUN_ROOT="$HOME/IndividualProject/results/baseline_trace/$RUN_ID"
BUILD_TRACE="$HOME/IndividualProject/build/hnsw-trace-on"
BUILD_OFF="$HOME/IndividualProject/build/hnsw-trace-off"
PYTHON_ENV="$HOME/IndividualProject/envs/baseline-trace-py312-v2"
DATASET_DIR=$(cd "$(dirname "$DATASET_CONFIG")" && pwd)
INDEX_PATH="$DATASET_DIR/indexes/${DATASET_NAME}_M${M}_efc${EF_CONSTRUCTION}_seed${SEED}_git${GIT_COMMIT:0:12}.bin"

[[ -x "$PYTHON_ENV/bin/python" ]] || { echo "Python environment missing: $PYTHON_ENV" >&2; exit 1; }
if [[ -n "$GIT_STATUS" ]]; then
  echo "Repository must be clean before a formal experiment:" >&2
  printf '%s\n' "$GIT_STATUS" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Run directory already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/config"
cp "$DATASET_CONFIG" "$RUN_ROOT/config/dataset.json"
cp "$EXPERIMENT_CONFIG" "$RUN_ROOT/config/experiment.json"
printf '%s\n' "$GIT_COMMIT" > "$RUN_ROOT/config/git_commit.txt"
printf '%s' "$GIT_STATUS" > "$RUN_ROOT/config/git_status.txt"

COMMON_EXPORT="ALL,REPO_ROOT=$REPO_ROOT,BUILD_TRACE=$BUILD_TRACE,BUILD_OFF=$BUILD_OFF,DATASET_CONFIG=$DATASET_CONFIG,INDEX_PATH=$INDEX_PATH,RUN_ID=$RUN_ID,K=$K,M=$M,EF_CONSTRUCTION=$EF_CONSTRUCTION,EF_SEARCH_COUNT=$EF_SEARCH_COUNT,SEED=$SEED,QUERY_START=$QUERY_START,QUERY_COUNT=$QUERY_COUNT,TRACE_SHARDS=$TRACE_SHARDS,DCO_SAMPLE_MODULUS=$DCO_SAMPLE_MODULUS,EDGE_SAMPLES_PER_SHARD=$EDGE_SAMPLES_PER_SHARD,PILOT_QUERY_COUNT=$PILOT_QUERY_COUNT,CORRECTNESS_QUERY_COUNT=$CORRECTNESS_QUERY_COUNT,STORAGE_BUDGET_GB=$STORAGE_BUDGET_GB,PERFORMANCE_REPEATS=$PERFORMANCE_REPEATS,WARMUP_QUERIES=$WARMUP_QUERIES,PYTHON_ENV=$PYTHON_ENV"

submit() {
  local name=$1; shift
  local job
  job=$(sbatch --parsable --output="$RUN_ROOT/logs/${name}_%A_%a.out" --error="$RUN_ROOT/logs/${name}_%A_%a.err" "$@")
  printf '%s=%s\n' "$name" "$job" | tee -a "$RUN_ROOT/submitted_jobs.txt" >&2
  printf '%s' "$job"
}

BUILD_JOB=$(submit build --export="$COMMON_EXPORT" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/build_binaries.slurm")

DATA_JOB=$(submit dataset_check --export="$COMMON_EXPORT" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/validate_dataset.slurm")

INDEX_JOB=$(submit index --dependency="afterok:$BUILD_JOB:$DATA_JOB" \
  --export="$COMMON_EXPORT,RUN_DIR=$RUN_ROOT" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/build_index.slurm")

CORRECT_JOB=$(submit correctness --dependency="afterok:$INDEX_JOB" \
  --export="$COMMON_EXPORT,RUN_DIR=$RUN_ROOT/correctness,EF_SEARCH=$MAX_EF" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/correctness_smoke.slurm")

PILOT_DIR="$RUN_ROOT/ef$MAX_EF"
PILOT_JOB=$(submit pilot_ef$MAX_EF --dependency="afterok:$CORRECT_JOB" \
  --export="$COMMON_EXPORT,RUN_DIR=$PILOT_DIR,EF_SEARCH=$MAX_EF" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/trace_pilot.slurm")

CAPACITY_JOB=$(submit capacity --dependency="afterok:$PILOT_JOB" \
  --export="$COMMON_EXPORT,RUN_DIR=$PILOT_DIR,EF_SEARCH=$MAX_EF" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/capacity_gate.slurm")

ARRAY_MAX=$((TRACE_SHARDS - 1))
SUMMARY_DEPENDENCIES=""
for EF_SEARCH in $EF_SEARCH_VALUES; do
  EF_DIR="$RUN_ROOT/ef$EF_SEARCH"
  mkdir -p "$EF_DIR/raw/shards"
  TRACE_JOB=$(submit trace_ef$EF_SEARCH --dependency="afterok:$CAPACITY_JOB" \
    --array="0-$ARRAY_MAX" \
    --export="$COMMON_EXPORT,RUN_DIR=$EF_DIR,EF_SEARCH=$EF_SEARCH" \
    "$REPO_ROOT/scripts/baseline_trace/slurm/collect_trace_array.slurm")
  ANALYZE_JOB=$(submit analyze_ef$EF_SEARCH --dependency="afterok:$TRACE_JOB" \
    --export="$COMMON_EXPORT,RUN_DIR=$EF_DIR,EF_SEARCH=$EF_SEARCH" \
    "$REPO_ROOT/scripts/baseline_trace/slurm/aggregate_analyze.slurm")
  PERFORMANCE_JOB=$(submit performance_ef$EF_SEARCH --dependency="afterok:$CORRECT_JOB" \
    --export="$COMMON_EXPORT,RUN_DIR=$RUN_ROOT,EF_SEARCH=$EF_SEARCH" \
    "$REPO_ROOT/scripts/baseline_trace/slurm/performance_baseline.slurm")
  SUMMARY_DEPENDENCIES="${SUMMARY_DEPENDENCIES}:$ANALYZE_JOB:$PERFORMANCE_JOB"
done

SUMMARY_JOB=$(submit summary --dependency="afterok:${SUMMARY_DEPENDENCIES#:}" \
  --export="$COMMON_EXPORT,RUN_ROOT=$RUN_ROOT" \
  "$REPO_ROOT/scripts/baseline_trace/slurm/summarize_experiment.slurm")

echo "first_full_experiment_submitted"
echo "run_id=$RUN_ID"
echo "run_root=$RUN_ROOT"
echo "summary_job=$SUMMARY_JOB"
echo "monitor: squeue -u $USER"
echo "jobs: $RUN_ROOT/submitted_jobs.txt"
