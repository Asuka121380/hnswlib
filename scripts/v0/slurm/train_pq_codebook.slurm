#!/usr/bin/env bash
#SBATCH --job-name=v0-pq-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=v0-pq-train-%A_%a.out

set -euo pipefail

: "${V0_REPO_ROOT:?set V0_REPO_ROOT}"
: "${V0_PYTHON:?set V0_PYTHON to the cluster Faiss Python}"
: "${V0_DIRECTIONS:?set V0_DIRECTIONS}"
: "${V0_MANIFEST:?set V0_MANIFEST}"
: "${V0_OUTPUT_ROOT:?set V0_OUTPUT_ROOT}"

widths=(8 16 32)
M_pq="${widths[${SLURM_ARRAY_TASK_ID}]}"
run_dir="${V0_OUTPUT_ROOT}/pq_m${M_pq}"
mkdir -p "${run_dir}"

"${V0_PYTHON}" "${V0_REPO_ROOT}/scripts/v0/train_pq_codebook.py" \
  --directions "${V0_DIRECTIONS}" \
  --manifest "${V0_MANIFEST}" \
  --M-pq "${M_pq}" \
  --nbits 8 \
  --seed "${V0_TRAINING_SEED:-17}" \
  --split-seed "${V0_SPLIT_SEED:-29}" \
  --iterations "${V0_ITERATIONS:-25}" \
  --nredo "${V0_NREDO:-1}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --producer-git-commit "${V0_PRODUCER_GIT_COMMIT:-unknown}" \
  --output-codebook "${run_dir}/pq_m${M_pq}_nbits8.v0pq" \
  --output-metrics "${run_dir}/training_metrics.json" \
  --output-errors "${run_dir}/validation_errors.f32"

"${V0_PYTHON}" "${V0_REPO_ROOT}/scripts/v0/validate_v0pq.py" \
  --codebook "${run_dir}/pq_m${M_pq}_nbits8.v0pq" \
  --expected-manifest "${V0_MANIFEST}" \
  --expected-directions "${V0_DIRECTIONS}"

"${V0_PYTHON}" "${V0_REPO_ROOT}/scripts/v0/compare_codebook_reconstruction.py" \
  --codebook "${run_dir}/pq_m${M_pq}_nbits8.v0pq" \
  --directions "${V0_DIRECTIONS}" \
  --manifest "${V0_MANIFEST}" \
  --sample-rows 1024 \
  --faiss-check \
  --output-metrics "${run_dir}/reconstruction_check.json"
