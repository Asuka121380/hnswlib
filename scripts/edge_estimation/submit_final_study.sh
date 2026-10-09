#!/usr/bin/env bash
# Explicit submission only. One invocation creates three distinct exclusive allocations.
set -euo pipefail
if (( $# < 7 )); then
  echo "Usage: $0 PARTITION ACCOUNT TIME_LIMIT OUT_ROOT --protocol ... --dataset ... --assets ... --build ... --cases ..." >&2
  exit 2
fi
partition="$1"; account="$2"; time_limit="$3"; out_root="$4"; shift 4
export REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
worker="$REPO/scripts/edge_estimation/final_study/worker.sh"
mkdir -p "$out_root"
for allocation in 0 1 2; do
  sbatch --parsable --job-name="edge-final-${allocation}" --partition="$partition" --account="$account" \
    --time="$time_limit" --nodes=1 --ntasks=1 --cpus-per-task=1 --hint=nomultithread \
    --exclusive --mem=0 --output="$out_root/slurm-%j.log" --export=ALL --chdir="$REPO" \
    "$worker" "$@" --allocation "$allocation" --out "$out_root/allocation-$allocation"
done
