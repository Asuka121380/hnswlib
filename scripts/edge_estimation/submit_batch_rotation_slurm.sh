#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    repo="${UQ_REPO:?}"
    python="${UQ_PYTHON:?}"
    runner="${UQ_NATIVE_RUNNER:?}"
    run_root="${UQ_RUN_ROOT:?}"
    queries="${UQ_QUERIES:?}"
    matrix="${UQ_BATCH_MATRIX:?}"
    batch_sizes="${UQ_BATCH_SIZES:?}"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    cd "$repo"
    [[ -x "$runner" ]]
    [[ -f "$run_root/development/events.bin" ]]
    [[ -f "$queries" ]]
    [[ -f "$matrix" ]]
    "$runner" capabilities | "$python" -c \
        'import json,sys; value=json.load(sys.stdin); assert value.get("batch_rotation_engine") == "blas_sgemv_sgemm", value'
    echo "batch_rotation_start host=$(hostname) time=$(date --iso-8601=seconds)"
    IFS=',' read -r -a sizes <<< "$batch_sizes"
    for batch_size in "${sizes[@]}"; do
        output="$run_root/timing-batch-b${batch_size}"
        [[ ! -e "$output" ]]
        for method in opq jq; do
            parity_file="$run_root/batch-parity-b${batch_size}-${method}.json"
            [[ ! -e "$parity_file" ]]
            parity="$("$runner" validate-batch-artifact \
                "$run_root/artifact-$method" \
                "$run_root/development/events.bin" \
                "$queries" "$batch_size")"
            "$python" -c \
                'import json,sys; value=json.loads(sys.argv[1]); assert value.get("valid") is True, value; assert value.get("batch_rotation_engine") == "blas_sgemv_sgemm", value' \
                "$parity"
            printf '%s\n' "$parity" > "$parity_file"
        done
        "$python" scripts/edge_estimation/run.py bench \
            --matrix "$matrix" \
            --events "$run_root/development/events.bin" \
            --queries "$queries" \
            --runner "$runner" \
            --query-batch-size "$batch_size" \
            --out "$output"
        [[ -f "$output/result.json" ]]
        [[ -f "$output/complete.json" ]]
        "$python" - "$output/result.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
engines = {row.get("batch_rotation_engine") for row in value["raw_records"]}
if engines != {"blas_sgemv_sgemm"}:
    raise SystemExit(f"batch run did not use BLAS: {sorted(engines)}")
PY
    done
    echo "BATCH_ROTATION_COMPLETE time=$(date --iso-8601=seconds)"
    exit 0
fi

usage() {
    cat <<'EOF'
usage: submit_batch_rotation_slurm.sh --run-root DIR --account NAME \
       --partition NAME --qos NAME [options]

options:
  --repo DIR          repository checkout (default: script repository)
  --python PATH       Python 3 interpreter (default: python3 on PATH)
  --runner PATH       BLAS-enabled uq_native_runner executable
  --queries PATH      GIST query fvecs
  --batch-sizes CSV   query batch sizes (default: 1,8,32,128,600)
  --blocks N          randomized paired blocks per size (default: 3)
  --repeats N         repeats per method per block (default: 3)
  --seed N            schedule seed (default: 20260927)
  --time HH:MM:SS     job limit (default: 01:00:00)
  --mem SIZE          job memory (default: 8G)
EOF
}

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
default_repo="$(cd "$(dirname "$script_path")/../.." && pwd)"
run_root=""
account=""
partition=""
qos=""
repo="$default_repo"
python=""
runner="$HOME/IndividualProject/build/uq-batch-rotation/uq_native_runner"
queries="$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs"
batch_sizes="1,8,32,128,600"
blocks="3"
repeats="3"
seed="20260927"
time_limit="01:00:00"
memory="8G"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) run_root="$2"; shift 2 ;;
        --account) account="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --qos) qos="$2"; shift 2 ;;
        --repo) repo="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --runner) runner="$2"; shift 2 ;;
        --queries) queries="$2"; shift 2 ;;
        --batch-sizes) batch_sizes="$2"; shift 2 ;;
        --blocks) blocks="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --mem) memory="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in run_root account partition qos; do
    if [[ -z "${!value_name}" ]]; then
        echo "--${value_name//_/-} is required" >&2
        usage >&2
        exit 2
    fi
done
for value_name in blocks repeats; do
    if ! [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--${value_name//_/-} must be a positive integer" >&2
        exit 2
    fi
done
if ! [[ "$seed" =~ ^[0-9]+$ ]] ||
   ! [[ "$batch_sizes" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "invalid seed or batch-size list" >&2
    exit 2
fi
if [[ -z "$python" ]]; then
    python="$(command -v python3 || true)"
fi
[[ -n "$python" ]]
for path in "$repo" "$run_root"; do [[ -d "$path" ]]; done
for path in "$python" "$runner"; do [[ -x "$path" ]]; done
[[ -f "$queries" ]]
IFS=',' read -r -a sizes <<< "$batch_sizes"
for batch_size in "${sizes[@]}"; do
    [[ ! -e "$run_root/timing-batch-b${batch_size}" ]]
    for method in opq jq; do
        [[ ! -e "$run_root/batch-parity-b${batch_size}-${method}.json" ]]
    done
done
mkdir -p "$run_root/logs"
matrix="$run_root/batch-rotation-matrix.json"
"$python" "$repo/scripts/edge_estimation/prepare_timing_matrix.py" \
    --run-root "$run_root" \
    --out "$matrix" \
    --methods opq jq \
    --blocks "$blocks" \
    --repeats "$repeats" \
    --seed "$seed" \
    --reference opq

job_id="$(sbatch --parsable \
    --account="$account" \
    --partition="$partition" \
    --qos="$qos" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem="$memory" \
    --time="$time_limit" \
    --job-name=uq-batch-rotation \
    --output="$run_root/logs/%x-%j.out" \
    --error="$run_root/logs/%x-%j.err" \
    --export="ALL,UQ_RUN_ROOT=$run_root,UQ_REPO=$repo,UQ_PYTHON=$python,UQ_NATIVE_RUNNER=$runner,UQ_QUERIES=$queries,UQ_BATCH_MATRIX=$matrix,UQ_BATCH_SIZES=$batch_sizes" \
    "$script_path" --worker)"
echo "batch_rotation_job_id=$job_id"
echo "monitor: squeue -j $job_id"
