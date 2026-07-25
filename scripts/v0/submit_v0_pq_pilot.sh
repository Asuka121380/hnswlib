#!/usr/bin/env bash
set -euo pipefail

: "${V0_REPO_ROOT:?set V0_REPO_ROOT}"
: "${V0_PYTHON:?set V0_PYTHON}"
: "${V0_DIRECTIONS:?set V0_DIRECTIONS}"
: "${V0_MANIFEST:?set V0_MANIFEST}"
: "${V0_OUTPUT_ROOT:?set V0_OUTPUT_ROOT}"

train_job="$(
  sbatch --parsable \
    --export=ALL,V0_REPO_ROOT,V0_PYTHON,V0_DIRECTIONS,V0_MANIFEST,V0_OUTPUT_ROOT \
    "${V0_REPO_ROOT}/scripts/v0/slurm/train_pq_codebook.slurm"
)"
summary_job="$(
  sbatch --parsable \
    --dependency="afterok:${train_job}" \
    --export=ALL,V0_REPO_ROOT,V0_PYTHON,V0_OUTPUT_ROOT \
    "${V0_REPO_ROOT}/scripts/v0/slurm/summarize_pq_pilot.slurm"
)"
printf 'train_job=%s\nsummary_job=%s\n' "${train_job}" "${summary_job}"
