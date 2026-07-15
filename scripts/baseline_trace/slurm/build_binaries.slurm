#!/usr/bin/env bash
#SBATCH --job-name=hnsw_build
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${BUILD_TRACE:?BUILD_TRACE is required}"
: "${BUILD_OFF:?BUILD_OFF is required}"

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
git -C "$REPO_ROOT" rev-parse HEAD
git -C "$REPO_ROOT" status --short
g++ --version | head -n 1
cmake --version | head -n 1

cmake -S "$REPO_ROOT" -B "$BUILD_TRACE" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_BASELINE_TRACE=ON
cmake --build "$BUILD_TRACE" --target \
  real_data_trace_runner baseline_trace_runner baseline_trace_test \
  --parallel "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$BUILD_TRACE" --output-on-failure -R baseline_trace_test

cmake -S "$REPO_ROOT" -B "$BUILD_OFF" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_BASELINE_TRACE=OFF
cmake --build "$BUILD_OFF" --target real_data_trace_runner \
  --parallel "$SLURM_CPUS_PER_TASK"

echo "build_binaries_ok end=$(date --iso-8601=seconds)"
