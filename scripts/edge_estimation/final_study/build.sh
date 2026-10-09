#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo"
exec "${PYTHON:-python3}" -m scripts.edge_estimation.final_study.build "$@"
