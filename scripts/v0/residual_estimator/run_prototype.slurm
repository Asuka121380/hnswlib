#!/usr/bin/env bash
#SBATCH --job-name=v0-residual
#SBATCH --output=logs/v0-residual-%j.out
#SBATCH --error=logs/v0-residual-%j.err
set -euo pipefail

: "${RESIDUAL_STAGE:?set RESIDUAL_STAGE}"
: "${RESIDUAL_RUN_ROOT:?set RESIDUAL_RUN_ROOT}"
: "${RESIDUAL_PYTHON:?set RESIDUAL_PYTHON}"
: "${RESIDUAL_REFERENCE_BUILD:?set RESIDUAL_REFERENCE_BUILD}"
: "${RESIDUAL_PERFORMANCE_BUILD:?set RESIDUAL_PERFORMANCE_BUILD}"

cd "${SLURM_SUBMIT_DIR:?run from the repository root}"
root=$(pwd -P)
run_root=$(realpath -m "$RESIDUAL_RUN_ROOT")
mkdir -p "$run_root"

"$RESIDUAL_PYTHON" -c 'import numpy, pandas, pyarrow; print("Python dependencies OK")'
"$RESIDUAL_PYTHON" -c 'import os; print("BLAS/OpenMP:", {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})'
git status --short --branch

script=scripts/v0/residual_estimator
contract="$run_root/contract/manifest.json"
dataset="$run_root/dataset/manifest.json"
sketch="$run_root/sketch/manifest.json"
selection="$run_root/selection.json"

case "$RESIDUAL_STAGE" in
  A0)
    : "${RESIDUAL_ASSETS:?set RESIDUAL_ASSETS}"
    "$RESIDUAL_PYTHON" "$script/freeze_contract.py" \
      --config configs/v0/residual_estimator/prototype_v1.json \
      --assets "$RESIDUAL_ASSETS" --output "$run_root/contract"
    "$RESIDUAL_PYTHON" "$script/build_dataset.py" \
      --contract "$contract" \
      --exporter "$RESIDUAL_REFERENCE_BUILD/v0_residual_exporter" \
      --output "$run_root/dataset"
    ;;
  A1)
    "$RESIDUAL_PYTHON" "$script/encode_sketch.py" \
      --contract "$contract" --dataset "$dataset" --output "$run_root/sketch"
    "$RESIDUAL_PYTHON" "$script/replay_frontier.py" \
      --dataset "$dataset" --sketch "$sketch" \
      --split development,selection --output "$run_root/replay-selection"
    "$RESIDUAL_PYTHON" "$script/select_candidates.py" \
      --input "$run_root/replay-selection/manifest.json" --output "$selection"
    ;;
  A2)
    "$RESIDUAL_PYTHON" "$script/capture_current_trace.py" \
      --config configs/v0/residual_estimator/current_shadow_v1.json \
      --contract "$contract" --runner "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" \
      --output "$run_root/current-shadow"
    ;;
  B)
    "$RESIDUAL_PYTHON" "$script/run_kernel_benchmark.py" \
      --config configs/v0/residual_estimator/kernel_v1.json \
      --selection "$selection" \
      --runner "$RESIDUAL_PERFORMANCE_BUILD/v0_residual_kernel_microbenchmark" \
      --output "$run_root/kernel"
    ;;
  C)
    : "${RESIDUAL_MATRIX:?set RESIDUAL_MATRIX to the selected matrix.f32}"
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION output path}"
    : "${RESIDUAL_BITS:?set RESIDUAL_BITS}"
    : "${RESIDUAL_SEED:?set RESIDUAL_SEED}"
    mapfile -t asset_paths < <("$RESIDUAL_PYTHON" - "$contract" <<'PY'
import json, sys
contract = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("index", "sidecar"):
    print(contract["assets"][key]["path"])
print(contract["config"]["dimension"])
PY
)
    mkdir -p "$(dirname "$RESIDUAL_COMPANION")"
    resume_args=()
    [[ "${RESIDUAL_RESUME:-0}" != 1 ]] || resume_args+=(--resume)
    "$RESIDUAL_REFERENCE_BUILD/v0_residual_encoder" \
      --index "${asset_paths[0]}" --sidecar "${asset_paths[1]}" \
      --dimension "${asset_paths[2]}" --matrix "$RESIDUAL_MATRIX" \
      --bits "$RESIDUAL_BITS" --seed "$RESIDUAL_SEED" \
      --output "$RESIDUAL_COMPANION" "${resume_args[@]}"
    ;;
  D-quality)
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION}"
    "$RESIDUAL_PYTHON" "$script/run_active_matrix.py" --stage quality \
      --config configs/v0/residual_estimator/active_quality_v1.json \
      --contract "$contract" --selection "$selection" \
      --runner "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" \
      --companion "$RESIDUAL_COMPANION" --output "$run_root/active-quality"
    ;;
  D-match)
    "$RESIDUAL_PYTHON" "$script/select_matched_recall.py" \
      --quality "$run_root/active-quality/manifest.json" \
      --config configs/v0/residual_estimator/active_qps_v1.json \
      --output "$run_root/matched-cases.json"
    ;;
  D-qps)
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION}"
    "$RESIDUAL_PYTHON" "$script/run_active_matrix.py" --stage qps \
      --config "$run_root/matched-cases.json" \
      --contract "$contract" --selection "$selection" \
      --runner "$RESIDUAL_PERFORMANCE_BUILD/v0_performance_runner" \
      --companion "$RESIDUAL_COMPANION" --output "$run_root/active-qps"
    "$RESIDUAL_PYTHON" "$script/summarize_active.py" \
      --quality "$run_root/active-quality/manifest.json" \
      --qps "$run_root/active-qps/manifest.json" \
      --output "$run_root/analysis/active-summary.json"
    ;;
  D-tune-quality)
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION}"
    "$RESIDUAL_PYTHON" scripts/v0/performance_ready/check_build_contract.py \
      --build-dir "$RESIDUAL_REFERENCE_BUILD" --contract residual-quality \
      --native-arch off
    if ! "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" --help 2>&1 | \
        grep -Fq -- '--query-id-file'; then
      echo "quality runner lacks --query-id-file: $RESIDUAL_REFERENCE_BUILD/v0_search_runner" >&2
      exit 2
    fi
    "$RESIDUAL_PYTHON" "$script/run_active_matrix.py" --stage quality \
      --config configs/v0/residual_estimator/residual_tuning_quality_v1.json \
      --contract "$contract" --selection "$selection" \
      --runner "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" \
      --companion "$RESIDUAL_COMPANION" \
      --output "$run_root/residual-tuning-quality"
    ;;
  D-tune-supplement-quality)
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION}"
    "$RESIDUAL_PYTHON" scripts/v0/performance_ready/check_build_contract.py \
      --build-dir "$RESIDUAL_REFERENCE_BUILD" --contract residual-quality \
      --native-arch off
    if ! "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" --help 2>&1 | \
        grep -Fq -- '--query-id-file'; then
      echo "quality runner lacks --query-id-file: $RESIDUAL_REFERENCE_BUILD/v0_search_runner" >&2
      exit 2
    fi
    "$RESIDUAL_PYTHON" "$script/run_active_matrix.py" --stage quality \
      --config configs/v0/residual_estimator/residual_tuning_supplement_quality_v1.json \
      --contract "$contract" --selection "$selection" \
      --runner "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" \
      --companion "$RESIDUAL_COMPANION" \
      --output "$run_root/residual-tuning-supplement-quality"
    ;;
  D-tune-fine-quality)
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION}"
    "$RESIDUAL_PYTHON" scripts/v0/performance_ready/check_build_contract.py \
      --build-dir "$RESIDUAL_REFERENCE_BUILD" --contract residual-quality \
      --native-arch off
    if ! "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" --help 2>&1 | \
        grep -Fq -- '--query-id-file'; then
      echo "quality runner lacks --query-id-file: $RESIDUAL_REFERENCE_BUILD/v0_search_runner" >&2
      exit 2
    fi
    "$RESIDUAL_PYTHON" "$script/run_active_matrix.py" --stage quality \
      --config configs/v0/residual_estimator/residual_tuning_fine_quality_v1.json \
      --contract "$contract" --selection "$selection" \
      --runner "$RESIDUAL_REFERENCE_BUILD/v0_search_runner" \
      --companion "$RESIDUAL_COMPANION" \
      --output "$run_root/residual-tuning-fine-quality"
    ;;
  D-tune-select)
    "$RESIDUAL_PYTHON" "$script/select_residual_tuning.py" \
      --quality "$run_root/residual-tuning-quality/manifest.json" \
      --config configs/v0/residual_estimator/residual_tuning_qps_v1.json \
      --output "$run_root/residual-tuning-cases.json"
    ;;
  D-tune-qps)
    : "${RESIDUAL_COMPANION:?set RESIDUAL_COMPANION}"
    "$RESIDUAL_PYTHON" scripts/v0/performance_ready/check_build_contract.py \
      --build-dir "$RESIDUAL_PERFORMANCE_BUILD" \
      --contract residual-performance --native-arch off
    "$RESIDUAL_PYTHON" "$script/run_active_matrix.py" --stage qps \
      --config "$run_root/residual-tuning-cases.json" \
      --contract "$contract" --selection "$selection" \
      --runner "$RESIDUAL_PERFORMANCE_BUILD/v0_performance_runner" \
      --companion "$RESIDUAL_COMPANION" \
      --output "$run_root/residual-tuning-qps"
    "$RESIDUAL_PYTHON" "$script/summarize_residual_tuning.py" \
      --quality "$run_root/residual-tuning-quality/manifest.json" \
      --qps "$run_root/residual-tuning-qps/manifest.json" \
      --output "$run_root/analysis/residual-tuning-summary.json"
    ;;
  *) echo "unknown stage: $RESIDUAL_STAGE" >&2; exit 2 ;;
esac
printf 'completed stage=%s repo=%s run=%s\n' "$RESIDUAL_STAGE" "$root" "$run_root"
