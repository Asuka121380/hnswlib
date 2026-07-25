#!/usr/bin/env bash
set -euo pipefail

: "${V0_REPO_ROOT:?set V0_REPO_ROOT}"
: "${V0_ENCODER:?set V0_ENCODER}"
: "${V0_INSPECTOR:?set V0_INSPECTOR}"
: "${V0_PYTHON:?set V0_PYTHON}"
: "${V0_INDEX_PATH:?set V0_INDEX_PATH}"
: "${V0_CODEBOOK_PATH:?set V0_CODEBOOK_PATH}"
: "${V0_SIDECAR_PATH:?set V0_SIDECAR_PATH}"
: "${V0_ENCODING_METRICS:?set V0_ENCODING_METRICS}"

producer_commit="${V0_PRODUCER_GIT_COMMIT:-$(
  git -C "${V0_REPO_ROOT}" rev-parse HEAD
)}"
export V0_PRODUCER_GIT_COMMIT="${producer_commit}"

encode_job="$(
  sbatch --parsable \
    --export=ALL \
    "${V0_REPO_ROOT}/scripts/v0/slurm/encode_v0_sidecar.slurm"
)"
validate_job="$(
  sbatch --parsable \
    --dependency="afterok:${encode_job}" \
    --export=ALL \
    "${V0_REPO_ROOT}/scripts/v0/slurm/validate_v0_sidecar.slurm"
)"
printf 'encode_job=%s\nvalidate_job=%s\n' \
  "${encode_job}" "${validate_job}"
