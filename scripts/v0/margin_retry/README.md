# V0 Margin + Retry: Stage 0–1

This directory contains only the Stage 0 baseline/input contract and the Stage 1 offline margin-frontier replay. It does not implement retry or active pruning.

## Stage 0

```powershell
python scripts/v0/margin_retry/freeze_stage0.py `
  --repo . `
  --shadow-run-dir C:\ANU\IndividualProject\8.3\20260803-minimal-v3\stageB-q1000 `
  --output-dir results/v0_margin_retry/<run_id>/stage0 `
  --initial-clean-observed
```

Stage 0 records the exact input hashes, branch/commit contract, environment, baseline Recall/DCO metadata, and provenance limitations of the frozen shadow run.

## Stage 1

```powershell
python scripts/v0/margin_retry/replay_margin_frontier.py `
  --input C:\ANU\IndividualProject\8.3\20260803-minimal-v3\stageB-q1000\shadow_records.csv.gz `
  --output-dir results/v0_margin_retry/<run_id>/stage1
```

The script freezes a deterministic query-level 60/20/20 split before reading distance values. All beta values are evaluated on train/validation; at most three beta values are selected from validation and evaluated once on test.

The Stage 1 process exits with code 0 only when both the frozen-input correctness checks and the held-out scientific viability gate pass.

## Stage 2

Build the V0 runner with both shadow flags:

```powershell
cmake -S . -B build-v0-margin-retry-stage2 -G Ninja `
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON `
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON `
  -DHNSWLIB_ENABLE_V0_APPROX_SHADOW=ON
```

Run the baseline-path hypothetical retry observer with the beta values frozen by Stage 1:

```powershell
v0_search_runner --mode retry-shadow `
  --retry-betas 1.4,1.45,1.55 `
  --shadow-sample-modulus 16 `
  <dataset/index/sidecar/output arguments>
```

The runner keeps full retry denominators in `retry_shadow_per_query_raw.csv`; `retry_shadow_records.csv` is a deterministic structural sample only. It does not skip exact distances or alter visited semantics.

Aggregate and classify Gate 2:

```powershell
python scripts/v0/margin_retry/analyze_retry_shadow.py `
  --runner-output-dir <retry-shadow-run> `
  --stage1-candidates results/v0_margin_retry/<run_id>/stage1/candidate_operating_points.json `
  --output-dir results/v0_margin_retry/<run_id>/stage2
```

## Unified Stage 3–5 active Recall/DCO experiment

Stage 3–5 is implemented as one correctness/metrics experiment with three internal checkpoints: raw-estimator parity, active retry state-machine correctness, and fixed-`efSearch` Recall/DCO validation. The initial `evaluateRaw()` intentionally reuses the strict evaluator (`raw_fast_path=false`); performance optimization remains deferred.

The frozen raw formula and fail-closed contract are documented in `RAW_ESTIMATOR_FORMULA.md`.

Build all active and regression modes:

```powershell
cmake -S . -B build-v0-margin-retry-active -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON `
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON `
  -DHNSWLIB_ENABLE_V0_APPROX_SHADOW=ON `
  -DHNSWLIB_ENABLE_V0_REAL_PRUNING=ON `
  -DHNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON
```

The active runner accepts one frozen beta per invocation:

```powershell
v0_search_runner --mode approx-no-retry --approx-beta 1.40 <common arguments>
v0_search_runner --mode approx-retry    --approx-beta 1.40 <common arguments>
```

Repeat both modes for `beta={1.40,1.45,1.55}`. Beta 1.40 retains its Stage 1 fail provenance but is deliberately included because offline/shadow local false-prune labels cannot determine final ground-truth Recall.

`query_metrics.csv` records per-query ground-truth hits, Recall loss, result overlap, baseline/active exact DCOs, first prunes, retry repayment, retry insertion, and state memory. `summary.json` reports fixed-budget pooled Recall and baseline-relative DCO reduction. These builds are not valid latency/QPS benchmarks.

Aggregate paired runs:

```powershell
python scripts/v0/margin_retry/analyze_active_experiment.py `
  --run-dir <ef200/approx-no-retry-beta1.40> `
  --run-dir <ef200/approx-retry-beta1.40> `
  --run-dir <additional paired runs> `
  --output-dir <stage3_5/analysis>
```

On the cluster, `run_stage3_5_active_experiment.slurm` builds, checks Phase A/B, runs strict V0 plus paired retry-off/on active modes, and aggregates Phase C. Defaults are q100, `efSearch=200`, and all three frozen betas. Override `QUERY_COUNT=1000` and `EF_VALUES=50,100,200,400` for the formal matrix.
