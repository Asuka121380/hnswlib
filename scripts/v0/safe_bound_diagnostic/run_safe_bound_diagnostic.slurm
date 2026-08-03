#!/usr/bin/env bash
#SBATCH --job-name=v0-safe-bound
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

repo_root="${REPO_ROOT:-$HOME/IndividualProject/src/hnswlib}"
build_root="${BUILD_ROOT:-$HOME/IndividualProject/build/hnsw-v0-safe-bound-diagnostic}"
run_root="${RUN_ROOT:-$HOME/IndividualProject/results/v0_safe_bound_diagnostic/gist1m/20260803-radius-v2}"
dataset_config="${DATASET_CONFIG:-$HOME/IndividualProject/datasets/gist1m/dataset.json}"
index_path="${INDEX_PATH:-$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin}"
sidecar_path="${SIDECAR_PATH:-$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta}"
analysis_script="${ANALYSIS_SCRIPT:-$repo_root/scripts/v0/safe_bound_diagnostic/analyze_radius_components.py}"
analysis_test="${ANALYSIS_TEST:-$repo_root/scripts/v0/safe_bound_diagnostic/test_analyze_radius_components.py}"
manifest_script="${MANIFEST_SCRIPT:-$repo_root/scripts/v0/safe_bound_diagnostic/write_run_manifest.py}"
run_stage_b="${RUN_STAGE_B:-0}"

if [[ -n "${ANALYSIS_PYTHON:-}" ]]; then
  analysis_python="$ANALYSIS_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  analysis_python="$repo_root/.venv/bin/python"
else
  analysis_python="$(command -v python3 || true)"
fi

for required in \
  "$repo_root/CMakeLists.txt" \
  "$dataset_config" \
  "$index_path" \
  "$sidecar_path" \
  "$analysis_script" \
  "$analysis_test" \
  "$manifest_script"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$analysis_python" ]]; then
  echo "Analysis Python is not executable: $analysis_python" >&2
  exit 2
fi
if [[ "$run_stage_b" != "0" && "$run_stage_b" != "1" ]]; then
  echo "RUN_STAGE_B must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$run_root"
producer_commit="$(git -C "$repo_root" rev-parse HEAD)"
git_branch="$(git -C "$repo_root" branch --show-current)"
working_tree_dirty=false
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  working_tree_dirty=true
fi

"$analysis_python" "$analysis_test"

cmake -S "$repo_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON \
  -DHNSWLIB_ENABLE_V0_REAL_PRUNING=OFF
cmake --build "$build_root" \
  --target v0_search_runner v0_shadow_validation_test \
  --parallel "${SLURM_CPUS_PER_TASK:-1}"
ctest --test-dir "$build_root" \
  -R v0_shadow_validation_test --output-on-failure

runner="$build_root/v0_search_runner"
if [[ ! -x "$runner" ]]; then
  echo "Runner is missing after build: $runner" >&2
  exit 2
fi

sha256sum "$dataset_config" "$index_path" "$sidecar_path" \
  > "$run_root/input_checksums.sha256"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "producer_commit=$producer_commit"
  echo "git_branch=$git_branch"
  echo "working_tree_dirty=$working_tree_dirty"
  echo "analysis_python=$analysis_python"
  "$analysis_python" --version
  cmake --version | head -n 1
} > "$run_root/run_environment.txt"

"$analysis_python" "$manifest_script" \
  --output "$run_root/run_manifest.json" \
  --repo-root "$repo_root" \
  --dataset-config "$dataset_config" \
  --index-path "$index_path" \
  --sidecar-path "$sidecar_path" \
  --build-root "$build_root" \
  --producer-commit "$producer_commit" \
  --git-branch "$git_branch" \
  --working-tree-dirty "$working_tree_dirty" \
  --run-stage-b "$run_stage_b"

run_one() {
  local label="$1"
  local query_count="$2"
  local output_dir="$run_root/$label"
  local analysis_dir="$run_root/analysis-$label"

  if [[ -e "$output_dir/complete.json" || -e "$analysis_dir/data_quality.json" ]]; then
    echo "Refusing to overwrite completed stage: $label" >&2
    return 2
  fi
  mkdir -p "$output_dir" "$analysis_dir"

  srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
    "$runner" \
      --dataset-config "$dataset_config" \
      --index-path "$index_path" \
      --sidecar-path "$sidecar_path" \
      --output-dir "$output_dir" \
      --mode shadow \
      --run-id "gist1m-v0-radius-v2-ef200-${label}-s16-${producer_commit:0:7}" \
      --query-start 0 \
      --query-count "$query_count" \
      --k 10 \
      --ef-search 200 \
      --shadow-sample-modulus 16 \
      --shadow-sample-remainder 0 \
      --producer-git-commit "$producer_commit" \
      --git-branch "$git_branch" \
      --working-tree-dirty "$working_tree_dirty"

  "$analysis_python" "$analysis_script" \
    --run-dir "$output_dir" \
    --output-dir "$analysis_dir" \
    --expected-query-count "$query_count"

  gzip -9 "$output_dir/shadow_records.csv"
  sha256sum \
    "$output_dir/metadata.json" \
    "$output_dir/summary.json" \
    "$output_dir/query_metrics.csv" \
    "$output_dir/shadow_records.csv.gz" \
    "$analysis_dir/data_quality.json" \
    "$analysis_dir/radius_component_summary.csv" \
    > "$output_dir/checksums.sha256"
}

run_one "stageA-radius-q100" 100

if [[ "$run_stage_b" == "1" ]]; then
  run_one "stageB-radius-q1000" 1000
else
  echo "Stage A passed; RUN_STAGE_B=0 so Stage B was not requested."
fi

printf '{"status":"complete","shadow_schema_version":2,"stage_b_run":%s}\n' \
  "$run_stage_b" > "$run_root/complete.json"
echo "finished_utc=$(date -u +%FT%TZ)" >> "$run_root/run_environment.txt"
