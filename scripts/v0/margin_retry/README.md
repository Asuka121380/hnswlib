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
