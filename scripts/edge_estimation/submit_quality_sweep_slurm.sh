#!/usr/bin/env bash
set -euo pipefail

METHODS=(pq8 pq4 opq prq jq rabitq)

if [[ "${1:-}" == "--worker" ]]; then
    method="${METHODS[${SLURM_ARRAY_TASK_ID:?}]}"
    repo="${UQ_REPO:?}"
    python="${UQ_PYTHON:?}"
    runner="${UQ_NATIVE_RUNNER:?}"
    run_root="${UQ_RUN_ROOT:?}"
    policy="${UQ_QUALITY_POLICY:?}"
    output="$run_root/quality-v2-$method"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    cd "$repo"
    [[ -x "$runner" ]]
    [[ -f "$run_root/development/events.bin" ]]
    [[ -f "$run_root/development/labels.bin" ]]
    [[ -f "$run_root/artifact-$method/complete.json" ]]
    [[ -f "$run_root/validation-$method/validation.json" ]]
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite existing output: $output" >&2
        exit 2
    fi
    "$python" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("valid") is not True:
    raise SystemExit("artifact validation evidence is not passing")
' "$run_root/validation-$method/validation.json"
    echo "quality_sweep_start method=$method host=$(hostname) time=$(date --iso-8601=seconds)"
    "$python" scripts/edge_estimation/run_quality_sweep.py \
        --runner "$runner" \
        --events "$run_root/development/events.bin" \
        --labels "$run_root/development/labels.bin" \
        --queries "$UQ_QUERIES" \
        --artifact "$run_root/artifact-$method" \
        --policy "$policy" \
        --out "$output"
    [[ -f "$output/complete.json" ]]
    [[ -f "$output/quality.json" ]]
    [[ -f "$output/selection.json" ]]
    echo "QUALITY_SWEEP_COMPLETE method=$method time=$(date --iso-8601=seconds)"
    exit 0
fi

usage() {
    cat <<'EOF'
usage: submit_quality_sweep_slurm.sh --run-root DIR --account NAME \
       --partition NAME --qos NAME [options]

options:
  --repo DIR          repository checkout (default: script repository)
  --python PATH       Faiss-capable Python interpreter
  --runner PATH       uq_native_runner executable
  --queries PATH      GIST query fvecs
  --policy PATH       quality policy JSON
  --time HH:MM:SS     per-array-task limit (default: 01:00:00)
  --mem SIZE          per-array-task memory (default: 8G)
  --concurrency N     maximum simultaneous tasks (default: 2)
EOF
}

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
default_repo="$(cd "$(dirname "$script_path")/../.." && pwd)"
run_root=""
account=""
partition=""
qos=""
repo="$default_repo"
python="$HOME/IndividualProject/envs/v0-pq/bin/python"
runner="$HOME/IndividualProject/build/uq-c0-6e46627/uq_native_runner"
queries="$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs"
policy="$default_repo/configs/edge_estimation/quality_policy_v2.json"
time_limit="01:00:00"
memory="8G"
concurrency="2"

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
        --policy) policy="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --mem) memory="$2"; shift 2 ;;
        --concurrency) concurrency="$2"; shift 2 ;;
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
if ! [[ "$concurrency" =~ ^[1-6]$ ]]; then
    echo "--concurrency must be an integer from 1 to 6" >&2
    exit 2
fi
for path in "$repo" "$run_root"; do [[ -d "$path" ]]; done
for path in "$python" "$runner"; do [[ -x "$path" ]]; done
for path in "$queries" "$policy"; do [[ -f "$path" ]]; done
mkdir -p "$run_root/logs"

job_id="$(sbatch --parsable \
    --account="$account" \
    --partition="$partition" \
    --qos="$qos" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem="$memory" \
    --time="$time_limit" \
    --array="0-5%$concurrency" \
    --job-name=uq-c1-quality-v2 \
    --output="$run_root/logs/%x-%A_%a.out" \
    --error="$run_root/logs/%x-%A_%a.err" \
    --export="ALL,UQ_RUN_ROOT=$run_root,UQ_REPO=$repo,UQ_PYTHON=$python,UQ_NATIVE_RUNNER=$runner,UQ_QUERIES=$queries,UQ_QUALITY_POLICY=$policy" \
    "$script_path" --worker)"
echo "quality_sweep_job_id=$job_id"
echo "monitor: squeue -j $job_id -r"
