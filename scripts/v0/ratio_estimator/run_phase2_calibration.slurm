#!/usr/bin/env bash
#SBATCH --job-name=v0-ratio-phase2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the hnswlib checkout}"
: "${INPUT_CSV:?set INPUT_CSV to the exact Phase-1 cap_diagnostic_input.csv}"
: "${PHASE1_DIR:?set PHASE1_DIR to the frozen PASS/GO_TO_PHASE2 artifacts}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for Phase-2 artifacts}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ALPHAS="${ALPHAS:-0.1,0.01,0.001,0.0001,0.00001}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/test_analyze_ratio_estimator.py"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/test_calibrate_probabilistic_bound.py"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/calibrate_probabilistic_bound.py" \
  --input "${INPUT_CSV}" \
  --phase1-dir "${PHASE1_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --alphas "${ALPHAS}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/apply_probabilistic_bound.py" \
  --input "${INPUT_CSV}" \
  --phase1-dir "${PHASE1_DIR}" \
  --calibration-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"

find "${OUTPUT_DIR}" \
  -type f \
  ! -name phase2_artifact_sha256.txt \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${OUTPUT_DIR}/phase2_artifact_sha256.txt"
