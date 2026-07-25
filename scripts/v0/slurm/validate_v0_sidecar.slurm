#!/usr/bin/env bash
#SBATCH --job-name=v0-sidecar-val
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=v0-sidecar-val-%j.out

set -euo pipefail

: "${V0_REPO_ROOT:?set V0_REPO_ROOT}"
: "${V0_PYTHON:?set V0_PYTHON}"
: "${V0_INSPECTOR:?set V0_INSPECTOR}"
: "${V0_INDEX_PATH:?set V0_INDEX_PATH}"
: "${V0_CODEBOOK_PATH:?set V0_CODEBOOK_PATH}"
: "${V0_SIDECAR_PATH:?set V0_SIDECAR_PATH}"

"${V0_INSPECTOR}" \
  --sidecar "${V0_SIDECAR_PATH}" \
  --index-path "${V0_INDEX_PATH}" \
  --sample-records "${V0_INSPECT_SAMPLE_RECORDS:-5}"

validator=(
  "${V0_PYTHON}"
  "${V0_REPO_ROOT}/scripts/v0/validate_v0_sidecar.py"
  --sidecar "${V0_SIDECAR_PATH}"
  --expected-v0pq "${V0_CODEBOOK_PATH}"
  --json
)
if [[ -n "${V0_EXPECTED_INDEX_SHA256:-}" ]]; then
  validator+=(--expected-index-sha256 "${V0_EXPECTED_INDEX_SHA256}")
fi
if [[ -n "${V0_EXPECTED_ADJACENCY_SHA256:-}" ]]; then
  validator+=(--expected-adjacency-sha256 "${V0_EXPECTED_ADJACENCY_SHA256}")
fi
if [[ -n "${V0_EXPECTED_CODEBOOK_SHA256:-}" ]]; then
  validator+=(--expected-codebook-sha256 "${V0_EXPECTED_CODEBOOK_SHA256}")
fi
if [[ -n "${V0_EXPECTED_NODE_COUNT:-}" ]]; then
  validator+=(--expected-node-count "${V0_EXPECTED_NODE_COUNT}")
fi
if [[ -n "${V0_EXPECTED_EDGE_COUNT:-}" ]]; then
  validator+=(--expected-edge-count "${V0_EXPECTED_EDGE_COUNT}")
fi
if [[ -n "${V0_EXPECTED_ZERO_LENGTH_COUNT:-}" ]]; then
  validator+=(
    --expected-zero-length-count "${V0_EXPECTED_ZERO_LENGTH_COUNT}"
  )
fi
"${validator[@]}"
