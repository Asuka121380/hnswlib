#!/usr/bin/env bash
#SBATCH --job-name=v0-active-recall
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

repo_root="${REPO_ROOT:-$HOME/IndividualProject/src/hnswlib}"
build_root="${BUILD_ROOT:-$HOME/IndividualProject/build/v0-margin-retry-active}"
run_root="${RUN_ROOT:-$HOME/IndividualProject/results/v0_margin_retry/stage3_5_active}"
dataset_config="${DATASET_CONFIG:-$HOME/IndividualProject/datasets/gist1m/dataset.json}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
analysis_python="${ANALYSIS_PYTHON:-$HOME/IndividualProject/envs/baseline-trace-py312-v2/bin/python}"
query_start="${QUERY_START:-0}"
query_count="${QUERY_COUNT:-100}"
betas_csv="${BETAS:-1.40,1.45,1.55}"
ef_values_csv="${EF_VALUES:-200}"
run_strict="${RUN_STRICT:-1}"

for required in "$repo_root/CMakeLists.txt" "$dataset_config" "$index_path" "$sidecar_path"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$analysis_python" ]]; then
  echo "Analysis Python is not executable: $analysis_python" >&2
  exit 2
fi

producer_commit="$(git -C "$repo_root" rev-parse HEAD)"
git_branch="$(git -C "$repo_root" branch --show-current)"
working_tree_dirty=false
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  working_tree_dirty=true
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cmake -S "$repo_root" -B "$build_root" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON \
  -DHNSWLIB_ENABLE_V0_APPROX_SHADOW=ON \
  -DHNSWLIB_ENABLE_V0_REAL_PRUNING=ON \
  -DHNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON
cmake --build "$build_root" --target \
  v0_search_runner v0_raw_estimator_test \
  v0_approx_real_pruning_test v0_feature_isolation_test \
  --parallel "${SLURM_CPUS_PER_TASK:-8}"
ctest --test-dir "$build_root" --output-on-failure \
  -R 'v0_(raw_estimator|approx_real_pruning|feature_isolation)_test'

runner="$build_root/v0_search_runner"
mkdir -p "$run_root/runs" "$run_root/analysis"
IFS=',' read -r -a betas <<< "$betas_csv"
IFS=',' read -r -a ef_values <<< "$ef_values_csv"
run_dirs=()

run_one() {
  local mode="$1"
  local ef_search="$2"
  local beta="$3"
  local label="$mode"
  local output_dir
  local args=()
  if [[ "$mode" == approx-* ]]; then
    label="${mode}-beta${beta}"
    args+=(--approx-beta "$beta")
  fi
  output_dir="$run_root/runs/ef${ef_search}/${label}"
  if [[ -f "$output_dir/complete.json" ]]; then
    echo "Reusing complete run: $output_dir"
    run_dirs+=("$output_dir")
    return
  fi
  mkdir -p "$output_dir"
  "$runner" \
    --dataset-config "$dataset_config" \
    --index-path "$index_path" \
    --sidecar-path "$sidecar_path" \
    --output-dir "$output_dir" \
    --mode "$mode" \
    --run-id "stage3_5-${mode}-ef${ef_search}-beta${beta}-q${query_count}" \
    --query-start "$query_start" \
    --query-count "$query_count" \
    --k 10 \
    --ef-search "$ef_search" \
    "${args[@]}" \
    --producer-git-commit "$producer_commit" \
    --git-branch "$git_branch" \
    --working-tree-dirty "$working_tree_dirty"
  run_dirs+=("$output_dir")
}

for ef_search in "${ef_values[@]}"; do
  if [[ "$run_strict" == 1 ]]; then
    run_one prune "$ef_search" 0
  fi
  for beta in "${betas[@]}"; do
    run_one approx-no-retry "$ef_search" "$beta"
    run_one approx-retry "$ef_search" "$beta"
  done
done

analysis_args=()
for run_dir in "${run_dirs[@]}"; do
  analysis_args+=(--run-dir "$run_dir")
done
if [[ "$run_strict" == 1 ]]; then
  analysis_args+=(--require-strict)
fi
minimum_query_count=1000
if (( query_count < minimum_query_count )); then
  minimum_query_count=1000
fi
"$analysis_python" \
  "$repo_root/scripts/v0/margin_retry/analyze_active_experiment.py" \
  "${analysis_args[@]}" \
  --output-dir "$run_root/analysis" \
  --minimum-query-count "$minimum_query_count" \
  --expected-betas "$betas_csv" \
  --expected-ef-search "$ef_values_csv" \
  --force

echo "stage3_5_active_experiment_ok run_root=$run_root"
