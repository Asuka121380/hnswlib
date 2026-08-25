#!/usr/bin/env bash
#SBATCH --job-name=v0-fast-calibration
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

repo_root="${REPO_ROOT:-$HOME/IndividualProject/src/hnswlib}"
reference_build="${REFERENCE_BUILD:-$repo_root/build-v0-approx-reference-15820d5}"
runner="$reference_build/v0_search_runner"
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
dataset_config="${DATASET_CONFIG:-$HOME/IndividualProject/datasets/gist1m/dataset.json}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
query_start="${QUERY_START:-0}"
query_count="${QUERY_COUNT:-300}"
k="${K:-10}"
ef_search="${EF_SEARCH:-200}"
run_role="${RUN_ROLE:-calibration}"
betas_csv="${BETAS:-1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.425,1.45,1.475,1.50,1.525,1.55,1.60,1.70,1.80,2.00}"
modes_csv="${MODES:-approx-no-retry,approx-retry}"

case "$run_role" in
  calibration|validation|heldout) ;;
  *)
    echo "RUN_ROLE must be calibration, validation, or heldout" >&2
    exit 2
    ;;
esac

cd "$repo_root"

for required in \
    "$repo_root/CMakeLists.txt" \
    "$runner" \
    "$analysis_python" \
    "$dataset_config" \
    "$index_path" \
    "$sidecar_path"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Commit mismatch: expected $EXPECTED_COMMIT, got $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Repository worktree is dirty" >&2
  git status --short >&2
  exit 2
fi

branch="$(git branch --show-current)"
IFS=',' read -r -a betas <<< "$betas_csv"
IFS=',' read -r -a modes <<< "$modes_csv"
if (( ${#betas[@]} == 0 )); then
  echo "BETAS must not be empty" >&2
  exit 2
fi
if (( ${#modes[@]} == 0 )); then
  echo "MODES must not be empty" >&2
  exit 2
fi
for mode in "${modes[@]}"; do
  case "$mode" in
    approx-no-retry|approx-retry) ;;
    *)
      echo "Unsupported calibration mode: $mode" >&2
      exit 2
      ;;
  esac
done

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-unavailable}"
echo "start=$(date --iso-8601=seconds)"
echo "commit=$actual_commit branch=$branch"
echo "query_start=$query_start query_count=$query_count k=$k ef_search=$ef_search"
echo "run_role=$run_role"
echo "betas=$betas_csv"
echo "modes=$modes_csv"
echo "reference_build=$reference_build"
echo "analysis_python=$analysis_python"
echo "dataset_config=$dataset_config"
echo "index_path=$index_path"
echo "sidecar_path=$sidecar_path"
cmake --version | head -n 1
g++ --version | head -n 1
"$analysis_python" --version

"$analysis_python" \
  scripts/v0/performance_ready/check_build_contract.py \
  --build-dir "$reference_build" \
  --contract reference

"$analysis_python" \
  scripts/v0/performance_ready/audit_fast_path_source.py \
  --header hnswlib/edge_quant_v0.h

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RUN_ROOT/runs"

for beta in "${betas[@]}"; do
  for mode in "${modes[@]}"; do
    label="${mode}-beta${beta}"
    output_dir="$RUN_ROOT/runs/$label"
    if [[ -f "$output_dir/complete.json" ]]; then
      echo "REUSE $label"
      continue
    fi

    echo "START mode=$mode beta=$beta time=$(date --iso-8601=seconds)"
    "$runner" \
      --dataset-config "$dataset_config" \
      --index-path "$index_path" \
      --sidecar-path "$sidecar_path" \
      --output-dir "$output_dir" \
      --mode "$mode" \
      --run-id "raw-fast-v1-${run_role}-q${query_start}-n${query_count}-${label}" \
      --query-start "$query_start" \
      --query-count "$query_count" \
      --k "$k" \
      --ef-search "$ef_search" \
      --approx-beta "$beta" \
      --producer-git-commit "$actual_commit" \
      --git-branch "$branch" \
      --working-tree-dirty false
    echo "DONE mode=$mode beta=$beta time=$(date --iso-8601=seconds)"
  done
done

expected_count=$(( ${#betas[@]} * ${#modes[@]} ))
"$analysis_python" \
  scripts/v0/performance_ready/summarize_calibration.py \
  --run-root "$RUN_ROOT" \
  --expected-count "$expected_count" \
  --role "$run_role" \
  --output "$RUN_ROOT/${run_role}_summary.csv"

echo "end=$(date --iso-8601=seconds)"
echo "v0_fast_${run_role}_ok"
