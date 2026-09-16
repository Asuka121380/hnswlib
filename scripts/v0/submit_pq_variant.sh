#!/usr/bin/env bash
# Scheduler options go before --; run_pq_variant.py options go after --.
# Example: V0_PYTHON=/path/python bash submit_pq_variant.sh \
#   --partition testing --qos normal -- --pq-m 16 --nbits 6 ...
set -euo pipefail
: "${V0_PYTHON:?set V0_PYTHON to Python with numpy and faiss-cpu}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export V0_VARIANT_SCRIPT="$script_dir/run_pq_variant.py"
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$V0_PYTHON" "$script_dir/run_pq_variant.py" --dry-run "$@"
fi
scheduler=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  scheduler+=("$1")
  shift
done
[[ $# -gt 0 ]] || { echo 'Use: submit_pq_variant.sh [sbatch options] -- [variant options]' >&2; exit 2; }
shift
sbatch --parsable --job-name=pq-variant --cpus-per-task=8 --mem=32G --time=10:00:00 \
  --export=ALL "${scheduler[@]}" "$script_dir/slurm/run_pq_variant.slurm" "$@"
