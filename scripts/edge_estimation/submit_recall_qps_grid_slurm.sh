#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    cd "${UQ_REPO:?}"
    [[ -x "${UQ_V0_RUNNER:?}" ]]
    [[ -x "${UQ_OPQ_RUNNER:?}" ]]
    echo "recall_qps_grid_start host=$(hostname) time=$(date --iso-8601=seconds)"
    "${UQ_PYTHON:?}" scripts/edge_estimation/run_recall_qps_grid.py \
        --config "${UQ_CONFIG:?}" \
        --v0-runner "$UQ_V0_RUNNER" \
        --opq-runner "$UQ_OPQ_RUNNER" \
        --index "${UQ_INDEX:?}" \
        --queries "${UQ_QUERIES:?}" \
        --ground-truth "${UQ_GROUND_TRUTH:?}" \
        --pq-sidecar "${UQ_PQ_SIDECAR:?}" \
        --qjl-companion "${UQ_QJL_COMPANION:?}" \
        --opq-artifact "${UQ_OPQ_ARTIFACT:?}" \
        --output "${UQ_OUTPUT:?}"
    [[ -f "$UQ_OUTPUT/complete.json" ]]
    echo "RECALL_QPS_GRID_JOB_COMPLETE time=$(date --iso-8601=seconds)"
    exit 0
fi

usage() {
    cat <<'EOF'
usage: submit_recall_qps_grid_slurm.sh --account NAME --partition NAME --qos NAME \
       --v0-runner FILE --opq-runner FILE --index FILE --queries FILE \
       --ground-truth FILE --pq-sidecar FILE --qjl-companion FILE \
       --opq-artifact DIR --output DIR [options]

options:
  --repo DIR       repository checkout (default: script repository)
  --python FILE    Python 3 interpreter (default: python3)
  --config FILE    grid JSON (default: configs/edge_estimation/recall_qps_grid_v1.json)
  --time D-HH:MM   job limit (default: 0-12:00)
  --mem SIZE       job memory (default: 8G)
EOF
}

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
repo="$(cd "$(dirname "$script_path")/../.." && pwd)"
python="$(command -v python3 || true)"
config="$repo/configs/edge_estimation/recall_qps_grid_v1.json"
account=""
partition=""
qos=""
v0_runner=""
opq_runner=""
index=""
queries=""
ground_truth=""
pq_sidecar=""
qjl_companion=""
opq_artifact=""
output=""
time_limit="0-12:00"
memory="8G"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) repo="$2"; shift 2 ;;
        --python) python="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        --account) account="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --qos) qos="$2"; shift 2 ;;
        --v0-runner) v0_runner="$2"; shift 2 ;;
        --opq-runner) opq_runner="$2"; shift 2 ;;
        --index) index="$2"; shift 2 ;;
        --queries) queries="$2"; shift 2 ;;
        --ground-truth) ground_truth="$2"; shift 2 ;;
        --pq-sidecar) pq_sidecar="$2"; shift 2 ;;
        --qjl-companion) qjl_companion="$2"; shift 2 ;;
        --opq-artifact) opq_artifact="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --mem) memory="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for name in account partition qos v0_runner opq_runner index queries ground_truth pq_sidecar qjl_companion opq_artifact output; do
    if [[ -z "${!name}" ]]; then
        echo "--${name//_/-} is required" >&2
        exit 2
    fi
done
[[ -d "$repo" && -d "$opq_artifact" ]]
for path in "$python" "$v0_runner" "$opq_runner"; do [[ -x "$path" ]]; done
for path in "$config" "$index" "$queries" "$ground_truth" "$pq_sidecar" "$qjl_companion"; do [[ -f "$path" ]]; done
if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 2
fi
mkdir -p "$(dirname "$output")" "$output-logs"

job_id="$(sbatch --parsable \
    --account="$account" --partition="$partition" --qos="$qos" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$memory" --time="$time_limit" \
    --job-name=uq-recall-qps-grid \
    --output="$output-logs/%x-%j.out" --error="$output-logs/%x-%j.err" \
    --export="ALL,UQ_REPO=$repo,UQ_PYTHON=$python,UQ_CONFIG=$config,UQ_V0_RUNNER=$v0_runner,UQ_OPQ_RUNNER=$opq_runner,UQ_INDEX=$index,UQ_QUERIES=$queries,UQ_GROUND_TRUTH=$ground_truth,UQ_PQ_SIDECAR=$pq_sidecar,UQ_QJL_COMPANION=$qjl_companion,UQ_OPQ_ARTIFACT=$opq_artifact,UQ_OUTPUT=$output" \
    "$script_path" --worker)"
echo "recall_qps_grid_job_id=$job_id"
echo "monitor: squeue -j $job_id -r"
