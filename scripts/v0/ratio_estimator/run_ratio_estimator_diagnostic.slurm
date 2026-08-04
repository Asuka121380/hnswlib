#!/usr/bin/env bash
#SBATCH --job-name=v0-ratio-phase1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT to the hnswlib checkout}"
: "${INPUT_CSV:?set INPUT_CSV to cap_diagnostic_input.csv}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for Phase-1 artifacts}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ANALYSIS_MODE="${ANALYSIS_MODE:-formal}"
KAPPA_MIN_CANDIDATES="${KAPPA_MIN_CANDIDATES:-0.1,0.2,0.3,0.4,0.5}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/test_analyze_ratio_estimator.py"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/analyze_ratio_estimator.py" \
  --input "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode "${ANALYSIS_MODE}" \
  --seed 42 \
  --kappa-min-candidates "${KAPPA_MIN_CANDIDATES}"

sha256sum \
  "${OUTPUT_DIR}/run_manifest.json" \
  "${OUTPUT_DIR}/query_split_manifest.json" \
  "${OUTPUT_DIR}/schema_audit.json" \
  "${OUTPUT_DIR}/summary.json" \
  > "${OUTPUT_DIR}/phase1_artifact_sha256.txt"
