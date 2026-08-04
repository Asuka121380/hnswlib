#!/usr/bin/env bash
#SBATCH --job-name=v0-ratio-phase4-report
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${RUNS_ROOT:?set RUNS_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${FRESH_QUERY_MANIFEST:?set FRESH_QUERY_MANIFEST}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/v0/ratio_estimator/test_check_ratio_cpp_python_parity.py"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/v0/ratio_estimator/test_compare_ratio_search_runs.py"

compare_exit=0
"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/v0/ratio_estimator/compare_ratio_search_runs.py" \
  --runs-root "${RUNS_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode "${ANALYSIS_MODE:-formal}" \
  --fresh-query-manifest "${FRESH_QUERY_MANIFEST}" \
  --expected-repetitions "${EXPECTED_REPETITIONS:-5}" \
  --expected-ef-search "${EXPECTED_EF_SEARCH:-100,200,400}" \
  --expected-operating-points "${EXPECTED_OPERATING_POINTS:-query_alpha_1e-2,query_alpha_1e-1,record_alpha_1e-1}" \
  || compare_exit=$?

if [[ "${compare_exit}" -ne 0 && "${compare_exit}" -ne 3 ]]; then
  exit "${compare_exit}"
fi
find "${OUTPUT_DIR}" -type f ! -name phase4_artifact_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "${OUTPUT_DIR}/phase4_artifact_sha256.txt"
exit "${compare_exit}"
