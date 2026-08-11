# EQ-RCP Stage 3 Input Freeze and Runbook

Stage 3 is offline progressive-code development for the frozen
`gain_shape_opq` representation. It does not authorize sidecar construction,
shadow integration, or real pruning from a development-only Stage 2.1 Gate.

Freeze the source artifacts and experiment contract first:

```powershell
python scripts/eq_rcp/freeze_stage3_inputs.py `
  --stage2-1-result-dir results/eq_rcp/stage2_1/development_full/20260810-200725-326 `
  --reassessment-dir results/eq_rcp/stage2_1/reassessments/20260810-200725-326 `
  --contract configs/eq_rcp/stage3_experiment_contract_v1.json `
  --output results/eq_rcp/stage3/input_freeze/stage3_input_manifest.json
```

Until a separately materialized, fresh evaluation-query manifest is supplied,
the output must remain `PENDING_FRESH_STAGE3_EVALUATION_QUERIES`. This pending
manifest is intentional: all five query roles in the source Stage 2.1 run have
already been consumed and cannot be reused to tune Stage 3 allocation or
stopping decisions.

The fresh evaluation manifest must use format
`eq_rcp_stage3_evaluation_queries`, version 1, certify
`fresh_against_stage2_1=true`, and bind three non-empty disjoint roles:
`bit_allocation_selection`, `stopping_calibration`, and `final_test`. Re-run the
freezer with `--evaluation-manifest`; only then may it emit
`READY_FOR_OFFLINE_STAGE3`.

The frozen comparison is flat 32B versus 28B+4B, 24B+8B, and 16B+16B. The
residual is tangent/direction residual under exact gain. Flat representation
families may not be reopened, the flat 32B baseline may not be retrained, and
randomized residual remains a Stage 11 challenger.

All Stage 3 risk/coverage values are diagnostics while risk status is
`PENDING_RISK_ESTIMATOR`; no output from this stage authorizes real pruning or
constitutes a full-search probability guarantee.
