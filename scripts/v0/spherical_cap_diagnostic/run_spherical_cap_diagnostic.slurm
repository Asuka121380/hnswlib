#!/bin/bash
#SBATCH --job-name=v0-cap-phase1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=v0-cap-phase1-%j.out

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${DATASET_CONFIG:?set DATASET_CONFIG}"
: "${INDEX_PATH:?set INDEX_PATH}"
: "${SIDECAR_PATH:?set SIDECAR_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-v0-cap-diagnostic}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
QUERY_COUNT="${QUERY_COUNT:-100}"
QUERY_START="${QUERY_START:-0}"
K="${K:-10}"
EF_SEARCH="${EF_SEARCH:-200}"
SAMPLE_MODULUS="${SAMPLE_MODULUS:-16}"
SAMPLE_REMAINDER="${SAMPLE_REMAINDER:-0}"
RUN_ID="${RUN_ID:-gist1m-v0-cap-phase1-ef${EF_SEARCH}-q${QUERY_COUNT}-s${SAMPLE_MODULUS}}"

"${BUILD_DIR}/v0_search_runner" \
  --dataset-config "${DATASET_CONFIG}" \
  --index-path "${INDEX_PATH}" \
  --sidecar-path "${SIDECAR_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode shadow \
  --run-id "${RUN_ID}" \
  --query-start "${QUERY_START}" \
  --query-count "${QUERY_COUNT}" \
  --k "${K}" \
  --ef-search "${EF_SEARCH}" \
  --shadow-sample-modulus "${SAMPLE_MODULUS}" \
  --shadow-sample-remainder "${SAMPLE_REMAINDER}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/spherical_cap_diagnostic/analyze_spherical_cap.py" \
  --input "${OUTPUT_DIR}/cap_diagnostic_input.csv" \
  --output-dir "${OUTPUT_DIR}/cap_phase1_analysis"
