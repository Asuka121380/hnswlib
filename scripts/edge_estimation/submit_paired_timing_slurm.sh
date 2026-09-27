#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    repo="${UQ_REPO:?}"
    python="${UQ_PYTHON:?}"
    runner="${UQ_NATIVE_RUNNER:?}"
    run_root="${UQ_RUN_ROOT:?}"
    queries="${UQ_QUERIES:?}"
    matrix="${UQ_TIMING_MATRIX:?}"
    output="$run_root/timing-v2"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    cd "$repo"
    [[ -x "$runner" ]]
    [[ -f "$run_root/development/events.bin" ]]
    [[ -f "$queries" ]]
    [[ -f "$matrix" ]]
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite existing output: $output" >&2
        exit 2
    fi
    echo "paired_timing_start host=$(hostname) time=$(date --iso-8601=seconds)"
    "$python" scripts/edge_estimation/run.py bench \
        --matrix "$matrix" \
        --events "$run_root/development/events.bin" \
        --queries "$queries" \
        --runner "$runner" \
        --formal \
        --out "$output"
    [[ -f "$output/result.json" ]]
    [[ -f "$output/complete.json" ]]
    echo "PAIRED_TIMING_COMPLETE time=$(date --iso-8601=seconds)"
    exit 0
fi

usage() {
    cat <<'EOF'
usage: submit_paired_timing_slurm.sh --run-root DIR --account NAME \
       --partition NAME --qos NAME [options]

options:
  --repo DIR          repository checkout (default: script repository)
  --python PATH       Python 3 interpreter (default: python3 on PATH)
  --runner PATH       uq_native_runner executable
  --queries PATH      GIST query fvecs
  --blocks N          randomized paired blocks (default: 5)
  --repeats N         repeats per method per block (default: 5)
  --seed N            schedule seed (default: 20260924)
  --reference NAME    ratio reference method (default: pq8)
  --time HH:MM:SS     job limit (default: 01:00:00)
  --mem SIZE          job memory (default: 4G)
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
runner="$HOME/IndividualProject/build/uq-c0-6e46627/uq_native_runner"
queries="$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs"
blocks="5"
repeats="5"
seed="20260924"
reference="pq8"
time_limit="01:00:00"
memory="4G"

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
        --blocks) blocks="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --reference) reference="$2"; shift 2 ;;
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
        echo "--${value_name} must be a positive integer" >&2
        exit 2
    fi
done
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer" >&2
    exit 2
fi
case "$reference" in pq8|pq4|opq|prq|jq|rabitq) ;; *)
    echo "invalid --reference: $reference" >&2; exit 2 ;;
esac
if [[ -z "$python" ]]; then
    python="$(command -v python3 || true)"
fi
if [[ -z "$python" ]]; then
    echo "python3 was not found; pass --python PATH" >&2
    exit 2
fi
for path in "$repo" "$run_root"; do [[ -d "$path" ]]; done
for path in "$python" "$runner"; do [[ -x "$path" ]]; done
[[ -f "$queries" ]]
[[ ! -e "$run_root/timing-v2" ]]
mkdir -p "$run_root/logs"
matrix="$run_root/timing-v2-matrix.json"
"$python" "$repo/scripts/edge_estimation/prepare_timing_matrix.py" \
    --run-root "$run_root" \
    --out "$matrix" \
    --blocks "$blocks" \
    --repeats "$repeats" \
    --seed "$seed" \
    --reference "$reference"

job_id="$(sbatch --parsable \
    --account="$account" \
    --partition="$partition" \
    --qos="$qos" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem="$memory" \
    --time="$time_limit" \
    --job-name=uq-c1-paired-timing-v2 \
    --output="$run_root/logs/%x-%j.out" \
    --error="$run_root/logs/%x-%j.err" \
    --export="ALL,UQ_RUN_ROOT=$run_root,UQ_REPO=$repo,UQ_PYTHON=$python,UQ_NATIVE_RUNNER=$runner,UQ_QUERIES=$queries,UQ_TIMING_MATRIX=$matrix" \
    "$script_path" --worker)"
echo "paired_timing_job_id=$job_id"
echo "monitor: squeue -j $job_id"
