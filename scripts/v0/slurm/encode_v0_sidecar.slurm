#!/usr/bin/env bash
#SBATCH --job-name=v0-edge-encode
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=v0-edge-encode-%j.out

set -euo pipefail

: "${V0_REPO_ROOT:?set V0_REPO_ROOT}"
: "${V0_ENCODER:?set V0_ENCODER to v0_faiss_edge_encoder}"
: "${V0_INDEX_PATH:?set V0_INDEX_PATH}"
: "${V0_CODEBOOK_PATH:?set V0_CODEBOOK_PATH to the selected M32 codebook}"
: "${V0_SIDECAR_PATH:?set V0_SIDECAR_PATH}"
: "${V0_ENCODING_METRICS:?set V0_ENCODING_METRICS}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
block_size="${V0_BLOCK_SIZE:-4096}"

command=(
  "${V0_ENCODER}"
  --index-path "${V0_INDEX_PATH}"
  --codebook-path "${V0_CODEBOOK_PATH}"
  --output-sidecar "${V0_SIDECAR_PATH}"
  --output-metrics "${V0_ENCODING_METRICS}"
  --block-size "${block_size}"
  --faiss-version "${V0_FAISS_VERSION:-1.14.3}"
  --faiss-source-commit "${V0_FAISS_SOURCE_COMMIT:-0ca9df4792b173d573044ee14ca0704780176e82}"
  --producer-git-commit "${V0_PRODUCER_GIT_COMMIT:-unknown}"
)
if [[ -n "${V0_BASE_PATH:-}" ]]; then
  command+=(--base-path "${V0_BASE_PATH}")
fi

printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
printf 'host=%s\n' "$(hostname)"
printf 'omp_threads=%s\n' "${OMP_NUM_THREADS}"
"${command[@]}"
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
