#!/usr/bin/env bash
# Submit explicitly: sbatch --cpus-per-task=1 --mem=32G --time=12:00:00 \
# scripts/edge_estimation/submit_ivf_slurm.sh train --config ... --assets ... --catalog ... --out ...
# Resource limits are starting estimates, not measured GIST peak memory/time.
set -euo pipefail
cd "${UQ_REPO:?set UQ_REPO to the checkout containing this script}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${UQ_PYTHON:?set UQ_PYTHON to a Faiss-enabled Python}" scripts/edge_estimation/ivf_experiments.py "$@"
