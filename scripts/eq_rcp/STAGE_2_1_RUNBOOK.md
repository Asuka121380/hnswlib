# EQ-RCP Stage 2.1 Runbook

Stage 2.1 is an offline representation-selection gate. It does not modify HNSW search.

1. Copy `configs/eq_rcp/stage2_1_development_template.json` and bind it to an immutable Stage 1 dataset/hash.
2. Every config must bind five disjoint roles: `quantizer_train`, `decoder_train`, `model_selection`, `risk_calibration`, and `final_test`.
3. Run the pilot with bounded `max_queries`; its only valid decision is `PENDING_FORMAL_STAGE2_1`.
4. Use `configs/eq_rcp/stage2_1_development_full_gist1m.json`, where all five `max_queries` are `null`, for `GO_STAGE3_DEV` eligibility. A legacy three-way manifest is rejected.
5. Use a fresh formal Stage 1 dataset, formal adjacency validation, at least three seeds, `run_role=formal_representation_selection`, and `allow_formal_gate_decision=true` for `GO_CPP` eligibility.

```powershell
scripts/eq_rcp/run_stage2_1_development.ps1 `
  -Config configs/eq_rcp/my_stage2_1.json `
  -OutputDir results/eq_rcp/stage2_1/run-name
```

The primary artifacts include `selection_report.json`, `selection_comparison.parquet`,
`calibration_manifest.json`, final-only `comparison.parquet`,
`radial_tangential_strata.parquet`, `gate_report.json`, and
`stage2_1_manifest.json`. Selection is frozen before risk calibration; event-level
risk UCB and cutoffs are frozen before final test. OAE is not privileged: the selected result may be
norm-corrected OPQ, gain-shape, grouped OAE, refined OPQ, or raw OPQ fallback.

## Split-gate closeout

The historical run directory is immutable. Generate the representation and
estimator-risk reports in a separate empty directory, then validate both the
source run and the derived closeout:

```powershell
python scripts/eq_rcp/reassess_stage2_1_closeout.py `
  --result-dir results/eq_rcp/stage2_1/development_full/run-name `
  --output-dir results/eq_rcp/stage2_1/reassessments/run-name

python scripts/eq_rcp/validate_stage2_1_selection.py `
  --config configs/eq_rcp/my_stage2_1.json `
  --result-dir results/eq_rcp/stage2_1/development_full/run-name `
  --reassessment-dir results/eq_rcp/stage2_1/reassessments/run-name `
  --report-output results/eq_rcp/stage2_1/reassessments/run-name/validation_report.json
```

`representation_gate_report.json` may authorize offline representation work.
`estimator_risk_diagnostic_report.json` remains `PENDING_RISK_ESTIMATOR`; its
event-level diagnostics do not authorize real HNSW pruning. The derived
`stage2_1_read_only_reassessment.json` records both orthogonal states and hashes
the original manifest and mixed Gate without rewriting either file.
