#!/usr/bin/env bash
#SBATCH --job-name=v0-ratio-phase3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the hnswlib checkout}"
: "${INPUT_CSV:?set INPUT_CSV to the exact Phase-1 cap_diagnostic_input.csv}"
: "${PHASE1_DIR:?set PHASE1_DIR to the frozen PASS/GO_TO_PHASE2 artifacts}"
: "${PHASE2_DIR:?set PHASE2_DIR to the frozen PASS/GO_TO_PHASE3 artifacts}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for Phase-3 artifacts}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_METADATA="${RAW_METADATA:-$(dirname "${INPUT_CSV}")/metadata.json}"
QUERY_METRICS="${QUERY_METRICS:-$(dirname "${INPUT_CSV}")/query_metrics.csv}"
SENSITIVITY_KAPPA="${SENSITIVITY_KAPPA:-}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/test_analyze_ratio_estimator.py"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/test_calibrate_probabilistic_bound.py"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/test_simulate_probabilistic_pruning.py"

simulation_exit=0
"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/simulate_probabilistic_pruning.py" \
  --input "${INPUT_CSV}" \
  --phase1-dir "${PHASE1_DIR}" \
  --phase2-dir "${PHASE2_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --metadata "${RAW_METADATA}" \
  --query-metrics "${QUERY_METRICS}" \
  --sensitivity-kappa "${SENSITIVITY_KAPPA}" \
  || simulation_exit=$?

if [[ "${simulation_exit}" -ne 0 && "${simulation_exit}" -ne 3 ]]; then
  exit "${simulation_exit}"
fi

find "${OUTPUT_DIR}" \
  -type f \
  ! -name phase3_artifact_sha256.txt \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${OUTPUT_DIR}/phase3_artifact_sha256.txt"

exit "${simulation_exit}"
