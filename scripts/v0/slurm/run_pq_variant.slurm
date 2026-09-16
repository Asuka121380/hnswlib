#!/usr/bin/env bash
set -euo pipefail
: "${V0_PYTHON:?set V0_PYTHON}"
# Slurm copies this file to its spool; the submitter exports the original path.
: "${V0_VARIANT_SCRIPT:?use submit_pq_variant.sh}"
exec "$V0_PYTHON" "$V0_VARIANT_SCRIPT" "$@" \
  --threads "${SLURM_CPUS_PER_TASK:-8}"
