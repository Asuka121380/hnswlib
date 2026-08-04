#!/usr/bin/env bash
#SBATCH --job-name=v0-ratio-real
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the hnswlib checkout}"
: "${RUNNER_BIN:?set RUNNER_BIN to a ratio-real v0_search_runner}"
: "${DATASET_CONFIG:?set DATASET_CONFIG}"
: "${INDEX_PATH:?set INDEX_PATH}"
: "${SIDECAR_PATH:?set SIDECAR_PATH}"
: "${SELECTED_PATH:?set SELECTED_PATH}"
: "${CALIBRATOR_PATH:?set CALIBRATOR_PATH}"
: "${OPERATING_POINT_ID:?set OPERATING_POINT_ID}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

QUERY_START="${QUERY_START:-0}"
QUERY_COUNT="${QUERY_COUNT:-1}"
K="${K:-10}"
EF_SEARCH="${EF_SEARCH:-200}"
WARMUP_QUERY_COUNT="${WARMUP_QUERY_COUNT:-10}"
REPETITION="${REPETITION:-0}"
RUN_ID="${RUN_ID:-ratio-real-${OPERATING_POINT_ID}-${SLURM_JOB_ID}}"
ALLOW_DIAGNOSTIC="${ALLOW_DIAGNOSTIC:-false}"
PRODUCER_GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
WORKING_TREE_DIRTY="$([[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]] && echo true || echo false)"

mkdir -p "${OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}/complete.json"

"${RUNNER_BIN}" \
  --dataset-config "${DATASET_CONFIG}" \
  --index-path "${INDEX_PATH}" \
  --sidecar-path "${SIDECAR_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode ratio-prune \
  --ratio-selected-path "${SELECTED_PATH}" \
  --ratio-calibrator-path "${CALIBRATOR_PATH}" \
  --ratio-operating-point-id "${OPERATING_POINT_ID}" \
  --allow-diagnostic-operating-point "${ALLOW_DIAGNOSTIC}" \
  --run-id "${RUN_ID}" \
  --query-start "${QUERY_START}" \
  --query-count "${QUERY_COUNT}" \
  --warmup-query-count "${WARMUP_QUERY_COUNT}" \
  --repetition "${REPETITION}" \
  --k "${K}" \
  --ef-search "${EF_SEARCH}" \
  --producer-git-commit "${PRODUCER_GIT_COMMIT}" \
  --git-branch "${GIT_BRANCH}" \
  --working-tree-dirty "${WORKING_TREE_DIRTY}"

test ! -e "${OUTPUT_DIR}/ratio_shadow_records.csv"
find "${OUTPUT_DIR}" -type f ! -name phase4_run_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "${OUTPUT_DIR}/phase4_run_sha256.txt"
