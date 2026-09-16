#!/usr/bin/env bash
#SBATCH --job-name=p1-environment-cell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

set -euo pipefail

for name in REPO_ROOT RUN_ROOT CONFIG_PATH PLATFORM_LABEL BUILD_VARIANT \
    RESOURCE_PROFILE REQUESTED_NODELIST EXPECTED_COMMIT PERFORMANCE_BUILD \
    ANALYSIS_PYTHON CMAKE_BIN NINJA_BIN CXX_BIN OPENSSL_LIB INDEX_PATH \
    SIDECAR_PATH QUERY_PATH; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
case "$BUILD_VARIANT" in portable) native_arch=OFF ;; native) native_arch=ON ;; *) exit 2 ;; esac

cd "$REPO_ROOT"
actual_commit="$(git rev-parse HEAD)"
[[ "$actual_commit" == "$EXPECTED_COMMIT" ]] || {
  echo "Commit mismatch: expected $EXPECTED_COMMIT, got $actual_commit" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "Repository worktree is dirty" >&2; git status --short >&2; exit 2; }
[[ "$(hostname -s)" == "$REQUESTED_NODELIST" ]] || {
  echo "Node mismatch: expected $REQUESTED_NODELIST, got $(hostname -s)" >&2; exit 2; }
[[ -z "${LD_LIBRARY_PATH:-}" ]] || {
  echo "LD_LIBRARY_PATH must be unset on entry" >&2; exit 2; }

echo "platform=$PLATFORM_LABEL build_variant=$BUILD_VARIANT host=$(hostname)"
echo "commit=$actual_commit start=$(date --iso-8601=seconds)"
echo "resource_profile=$RESOURCE_PROFILE job=${SLURM_JOB_ID:-unavailable}"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -dd "$SLURM_JOB_ID" || true
fi

LD_LIBRARY_PATH="$OPENSSL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$CMAKE_BIN" -S "$REPO_ROOT" -B "$PERFORMANCE_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_MAKE_PROGRAM="$NINJA_BIN" \
    -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
    -DHNSWLIB_ENABLE_BASELINE_TRACE=OFF \
    -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=OFF \
    -DHNSWLIB_ENABLE_V0_APPROX_SHADOW=OFF \
    -DHNSWLIB_ENABLE_V0_REAL_PRUNING=ON \
    -DHNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON \
    -DHNSWLIB_ENABLE_V0_STRICT_FP_CONTRACT=OFF \
    -DHNSWLIB_PERFORMANCE_COMPARABLE_FLAGS=ON \
    -DHNSWLIB_ENABLE_NATIVE_ARCH="$native_arch"
LD_LIBRARY_PATH="$OPENSSL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$CMAKE_BIN" --build "$PERFORMANCE_BUILD" \
    --target v0_performance_runner v0_fast_kernel_microbenchmark --parallel 1

native_arg="$(tr '[:upper:]' '[:lower:]' <<< "$native_arch")"
"$ANALYSIS_PYTHON" scripts/v0/performance_ready/check_build_contract.py \
  --build-dir "$PERFORMANCE_BUILD" --contract performance \
  --native-arch "$native_arg"
"$ANALYSIS_PYTHON" scripts/v0/performance_ready/audit_fast_path_source.py \
  --header "$REPO_ROOT/hnswlib/edge_quant_v0.h"

allowed_cpu_list="$(awk '$1 == "Cpus_allowed_list:" {print $2}' /proc/self/status)"
logical_cpu="${allowed_cpu_list%%,*}"
logical_cpu="${logical_cpu%%-*}"
[[ "$logical_cpu" =~ ^[0-9]+$ ]] || {
  echo "Cannot select one logical CPU from $allowed_cpu_list" >&2; exit 2; }
echo "selected_logical_cpu=$logical_cpu"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

qps_root="$RUN_ROOT/qps"
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=threads --mem-bind=local \
  taskset --cpu-list "$logical_cpu" \
  "$ANALYSIS_PYTHON" scripts/v0/performance_ready/run_qps_matrix.py \
    --config "$CONFIG_PATH" --resource-profile "$RESOURCE_PROFILE" \
    --runner "$PERFORMANCE_BUILD/v0_performance_runner" \
    --index-path "$INDEX_PATH" --sidecar-path "$SIDECAR_PATH" \
    --query-path "$QUERY_PATH" --run-root "$qps_root" \
    --cmake-cache "$PERFORMANCE_BUILD/CMakeCache.txt" \
    --experiment-commit "$actual_commit" --numa-policy slurm-mem-bind-local \
    --power-policy cluster-default-unverified \
    --cpu-frequency-policy cluster-default-unverified
"$ANALYSIS_PYTHON" scripts/v0/performance_ready/summarize_qps_experiment.py \
  --run-root "$qps_root"

component_root="$RUN_ROOT/component"
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=threads --mem-bind=local \
  taskset --cpu-list "$logical_cpu" \
  "$ANALYSIS_PYTHON" scripts/v0/performance_ready/run_component_microbenchmark.py \
    --runner "$PERFORMANCE_BUILD/v0_fast_kernel_microbenchmark" \
    --cmake-cache "$PERFORMANCE_BUILD/CMakeCache.txt" \
    --sidecar-path "$SIDECAR_PATH" --run-root "$component_root" \
    --resource-profile "$RESOURCE_PROFILE" --requested-node "$REQUESTED_NODELIST" \
    --experiment-commit "$actual_commit" --git-branch "$(git branch --show-current)" \
    --blocks 7 --iterations 100000 --warmup-iterations 10000 --working-set 64

printf 'schema_version=%s\nplatform=%s\nbuild_variant=%s\ngit_commit=%s\nhost=%s\nbuild_dir=%s\n' \
  'p1-environment-cell-v1' "$PLATFORM_LABEL" "$BUILD_VARIANT" \
  "$actual_commit" "$(hostname -s)" "$PERFORMANCE_BUILD" > "$RUN_ROOT/cell.env"
printf '%s\n' "$actual_commit" > "$RUN_ROOT/COMPLETE"
echo "end=$(date --iso-8601=seconds) p1_environment_cell_ok"
