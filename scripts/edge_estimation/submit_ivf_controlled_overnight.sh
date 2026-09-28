#!/usr/bin/env bash
# Cluster-specific one-job bundle: frozen source, build, smoke, 161-point grid.
set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run="$1"
    stage=environment
    trap 'rc=$?; printf "FAILED stage=%s exit=%s time=%s\n" "$stage" "$rc" "$(date -Is)" > "$run/FAILED.txt"; exit "$rc"' ERR
    type module >/dev/null 2>&1 && module purge
    export PATH="/usr/bin:/bin:$PATH"
    unset LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH GCC_EXEC_PREFIX COMPILER_PATH
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export MPLCONFIGDIR="$run/matplotlib-cache"
    mkdir -p "$MPLCONFIGDIR"
    source_repo="$run/source"
    cd "$source_repo"
    py="$HOME/IndividualProject/envs/v0-pq/bin/python"
    cmake="$HOME/IndividualProject/envs/v0-pq/bin/cmake"
    ninja="/opt/apps/eb/software/Ninja/1.10.2-GCCcore-10.3.0/bin/ninja"
    blas="$HOME/IndividualProject/envs/v0-pq/lib/python3.12/site-packages/faiss_cpu.libs/libopenblaso-r0-d77a1985.3.15.so"
    # Wheel libraries have hashed dependency names. Keep only their directory,
    # not inherited compiler/module directories that can select an old linker.
    export LD_LIBRARY_PATH="$(dirname "$blas")"
    build="$run/build"
    artifacts_root="$(cat "$run/artifacts-root.txt")"
    index="$HOME/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin"
    queries="$HOME/IndividualProject/datasets/gist1m/gist/gist_query.fvecs"
    truth="$HOME/IndividualProject/datasets/gist1m/gist/gist_groundtruth.ivecs"
    pq="$HOME/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta"
    qjl="$HOME/IndividualProject/results/v0_residual_estimator/p3-prototype-v1-092a1ef/companion/seed-42-k128.v0res"
    opq="$HOME/IndividualProject/results/edge_estimation/c1-six-method-6e46627-20260926-165959/artifact-opq"

    {
        hostname
        date -Is
        cat "$run/source-commit.txt"
        lscpu
        "$cmake" --version
        /usr/bin/c++ --version
        /usr/bin/ld --version
        "$py" -c 'import sys,numpy,faiss,matplotlib; print(sys.executable); print(numpy.__version__,faiss.__version__,matplotlib.__version__)'
        ldd "$blas"
    } > "$run/environment.txt" 2>&1

    stage=preflight
    ldd "$blas" > "$run/blas-dependencies.txt"
    if grep -q 'not found' "$run/blas-dependencies.txt"; then
        cat "$run/blas-dependencies.txt" >&2
        false
    fi
    for path in "$index" "$queries" "$truth" "$pq" "$qjl" "$opq/manifest.json" "$opq/native.cfg" "$blas"; do
        test -f "$path"
    done
    for method in ivf_pq ivf_opq ivf_pq_qjl; do
        test -f "$artifacts_root/$method/complete.json"
    done
    sha256sum "$index" "$queries" "$truth" "$pq" "$qjl" "$opq/manifest.json" "$opq/native.cfg" "$blas" > "$run/input-sha256.txt"
    "$py" - "$index" "$queries" "$truth" <<'PY'
import sys
from scripts.edge_estimation.contracts import sha256_file
expected = [
    "35be7096a3e482b94f429ccf17dbeaaf70732f0e5b7bed690d9a27ecbbdcb1a6",
    "0d1d620049de12da455ed7201e97cbab372c4d54d0e6dedbc8c503f62c911299",
    "01f7eda9dc98600c6288f606e55f94f13ae95f8b07d818572260440062d609f6",
]
for path, digest in zip(sys.argv[1:], expected):
    assert sha256_file(path) == digest, f"Frozen input mismatch: {path}"
PY
    stage=build
    "$cmake" -S . -B "$build" -G Ninja \
      -DCMAKE_MAKE_PROGRAM="$ninja" -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DHNSWLIB_ENABLE_NATIVE_ARCH=OFF \
      -DHNSWLIB_PERFORMANCE_COMPARABLE_FLAGS=ON \
      -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
      -DHNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON \
      -DHNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR=ON \
      -DHNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING=ON \
      -DHNSWLIB_BUILD_EDGE_ESTIMATION_TOOLS=ON \
      -DHNSWLIB_ENABLE_EDGE_ESTIMATION_ACTIVE=ON \
      -DHNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE=ON \
      -DUQ_WITH_BLAS=ON -DUQ_BLAS_LIBRARY="$blas" \
      > "$run/configure.log" 2>&1
    "$cmake" --build "$build" --target v0_performance_runner uq_opq_performance_runner \
      uq_ivf_performance_runner uq_native_runner uq_capture uq_capture_fixture \
      --parallel "${SLURM_CPUS_PER_TASK:-1}" > "$run/build.log" 2>&1

    stage=runtime_libraries
    for runner in uq_native_runner uq_ivf_performance_runner uq_opq_performance_runner v0_performance_runner; do
        ldd "$build/$runner" > "$run/$runner-dependencies.txt"
        if grep -q 'not found' "$run/$runner-dependencies.txt"; then
            cat "$run/$runner-dependencies.txt" >&2
            false
        fi
    done

    stage=integration
    "$py" tests/python/uq_ivf_integration_test.py --build "$build" --out "$run/integration" \
      > "$run/integration.log" 2>&1

    # One core for performance processes, even though the allocation permits parallel compilation.
    cpu="$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    cpu="${cpu%%,*}"
    cpu="${cpu%%-*}"
    printf '%s\n' "$cpu" > "$run/performance-cpu.txt"
    common=(--artifacts "$artifacts_root" --index "$index" --queries "$queries"
      --ground-truth "$truth" --runner "$build/uq_ivf_performance_runner"
      --v0-runner "$build/v0_performance_runner" --opq-runner "$build/uq_opq_performance_runner"
      --pq-sidecar "$pq" --qjl-companion "$qjl" --opq-artifact "$opq")
    stage=smoke
    taskset -c "$cpu" "$py" scripts/edge_estimation/ivf_experiments.py grid \
      --config configs/edge_estimation/ivf_k256_controlled_smoke.json \
      "${common[@]}" --out "$run/smoke" > "$run/smoke.log" 2>&1

    stage=baseline_bridge
    taskset -c "$cpu" "$build/v0_performance_runner" \
      --index-path "$index" --query-path "$queries" --dimension 960 \
      --query-start 0 --query-count 100 --k 10 --ef-search 400 \
      --warmup-queries 20 --repeats 1 --method baseline \
      --output "$run/legacy-baseline.json" --result-records-output "$run/legacy-baseline-results.csv" \
      > "$run/baseline-bridge.log" 2>&1
    "$py" - "$run" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = list(csv.DictReader((root/"smoke/points.csv").open()))
assert len(rows) == 7
assert len({r["method"] for r in rows}) == 7
for row in rows:
    method = row["method"]
    perf = json.loads((root/"smoke"/row["case_id"]/"performance.json").read_text())
    if method in ("pq8", "pq_qjl"):
        # Legacy performance output does not expose pruning counters. Validate
        # the selected search path without inventing a zero or missing count.
        expected = "approx-no-retry" if method == "pq8" else "residual-threshold"
        assert perf["method"] == expected, (method, perf.get("method"))
        parameter = "beta" if method == "pq8" else "theta"
        assert abs(float(perf[parameter])-float(row["beta"])) < 1e-10
        assert perf["result_records_written"] and perf["measured_queries"] == 100
    elif method != "hnsw":
        assert int(row["pruned_estimates"]) > 0, row
baseline = next(r for r in rows if r["method"] == "hnsw")
def labels(path):
    with path.open() as f:
        return {(int(r["query_id"]), int(r["rank"])): int(r["label"]) for r in csv.DictReader(f)}
a = labels(root/"legacy-baseline-results.csv")
b = labels(root/"smoke"/baseline["case_id"]/"results.csv")
assert len(a) == 1000 and a == b, "Baseline runner top-k mismatch"
legacy = json.loads((root/"legacy-baseline.json").read_text())
report = {"topk_equal": True, "legacy_qps": legacy["qps"],
          "ivf_runner_no_prune_qps": float(baseline["qps"]),
          "legacy_pruning_counts": "not reported by legacy runner",
          "note": "100-query smoke timings are diagnostic only; not a performance equivalence test."}
(root/"baseline-bridge.json").write_text(json.dumps(report, indent=2)+"\n")
PY
    if [[ -f "$run/opq-confirm-study.txt" ]]; then
        stage=opq_confirmation
        taskset -c "$cpu" /usr/bin/time -v "$py" scripts/edge_estimation/ivf_opq_confirmation.py \
          "$run" "${common[@]}" 2> "$run/opq-resource.log"
        printf 'COMPLETE OPQ study run=%s time=%s\n' "$run" "$(date -Is)"
        exit 0
    fi
    stage=grid
    taskset -c "$cpu" /usr/bin/time -v "$py" scripts/edge_estimation/ivf_experiments.py grid \
      --config configs/edge_estimation/ivf_k256_controlled_full.json \
      "${common[@]}" --out "$run/grid" 2> "$run/grid-resource.log"
    stage=plots
    "$py" scripts/edge_estimation/ivf_experiments.py summarize --out "$run/grid" --plot \
      > "$run/plots.log" 2>&1
    stage=report
    "$py" - "$run" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
complete=json.loads((root/"grid/complete.json").read_text())
assert complete["points"] == 161
rows=list(csv.DictReader((root/"grid/thresholds.csv").open()))
lines=["# K256 seven-method exploratory comparison", "",
       "161 points, one repeat, OPQ query batch 128, native OFF.",
       "Threshold-qualified points are not exact matched-recall comparisons.", "",
       "| Recall threshold | Method | Actual recall | QPS | ef | beta/theta |",
       "|---|---|---|---|---|---|"]
for r in rows:
    if float(r["threshold"]) not in (.95,.97,.98): continue
    if r["status"] != "OK":
        lines.append(f'| {r["threshold"]} | {r["method"]} | NOT_REACHED | | | |')
    else:
        lines.append(f'| {r["threshold"]} | {r["method"]} | {float(r["recall_at_k"]):.4f} | {float(r["qps"]):.2f} | {r["ef_search"]} | {r["beta"]} |')
(root/"SUMMARY.md").write_text("\n".join(lines)+"\n")
(root/"COMPLETE.json").write_text(json.dumps({"valid":True,"points":161,"smoke_points":7,
    "source_commit":(root/"source-commit.txt").read_text().strip()},indent=2)+"\n")
PY
    printf 'COMPLETE run=%s time=%s\n' "$run" "$(date -Is)"
    exit 0
fi

account="" partition="" qos="" dry_run=0 opq_confirm=0
memory=32G time_limit=12:00:00 cpus=4
artifacts_root="$HOME/IndividualProject/results/ivf/gist-k256-GnqHj6/artifacts/k256"
while (( $# )); do
    case "$1" in
        --account) account="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --qos) qos="$2"; shift 2 ;;
        --mem) memory="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        --artifacts) artifacts_root="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --opq-confirm) opq_confirm=1; shift ;;
        --help)
            echo "bash scripts/edge_estimation/submit_ivf_controlled_overnight.sh --account NAME --partition NAME --qos NAME [--opq-confirm] [--mem 32G] [--time 12:00:00] [--cpus 4] [--artifacts DIR] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$account" && -n "$partition" && -n "$qos" ]] || { echo "Explicit account, partition and qos required" >&2; exit 2; }
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
for method in ivf_pq ivf_opq ivf_pq_qjl; do test -f "$artifacts_root/$method/complete.json"; done
# Refuse to snapshot an older commit that lacks this bundle.
git -C "$repo" cat-file -e HEAD:scripts/edge_estimation/submit_ivf_controlled_overnight.sh
git -C "$repo" cat-file -e HEAD:configs/edge_estimation/ivf_k256_controlled_full.json
mkdir -p "$HOME/IndividualProject/results/ivf"
run="$(mktemp -d "$HOME/IndividualProject/results/ivf/controlled-overnight-XXXXXX")"
mkdir "$run/source"
git -C "$repo" rev-parse HEAD > "$run/source-commit.txt"
git -C "$repo" archive HEAD | tar -x -C "$run/source"
if (( opq_confirm )); then
    test -f "$run/source/scripts/edge_estimation/ivf_opq_confirmation.py"
    printf 'opq-confirm\n' > "$run/opq-confirm-study.txt"
fi
printf '%s\n' "$artifacts_root" > "$run/artifacts-root.txt"
printf 'RUN=%s\n' "$run"
if (( dry_run )); then
    echo "Snapshot prepared; no job submitted."
    exit 0
fi
job=$(sbatch --parsable --account="$account" --partition="$partition" --qos="$qos" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$memory" --time="$time_limit" \
    --job-name=ivf-k256-controlled --export=ALL \
    --output="$run/slurm-%j.out" --error="$run/slurm-%j.err" \
    "$run/source/scripts/edge_estimation/submit_ivf_controlled_overnight.sh" --worker "$run")
printf '%s\n' "$job" > "$run/job-id.txt"
printf 'JOB_ID=%s\nRUN=%s\n' "$job" "$run"
