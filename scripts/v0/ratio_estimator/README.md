# V0 ratio-corrected PQ estimator — Phase 1

This directory implements the offline estimator diagnostic defined in
`01_PHASE1_OFFLINE_ESTIMATOR_DIAGNOSTIC_IMPLEMENTATION_PLAN.md`.

It compares the existing normalized PQ projection with the ratio-corrected
projection, verifies that the online metadata can reproduce the denominator,
freezes a deterministic query-level 60/20/20 split, and reports overall and
conditional projection/distance error.  It does not construct a probabilistic
bound, alter the sidecar, or change HNSW search control flow.

## Input contract

The required fields are:

```text
query_id,current_node_id,candidate_id,threshold,edge_length,
reconstruction_norm,x_norm,x_dot_r,x_dot_true_direction,
actual_direction_error,direction_error,current_lb,diagnostic_valid,
shadow_exact_squared_distance
```

Schema-v2 cap exports provide `geometric_squared_distance`, accumulated from
query/candidate coordinates in long double.  The analyzer prefers that value
for the strict geometry closure.  Older cap exports used the lower-precision
operational `exact_squared_distance`; it remains an explicit fallback and the
selected source plus any difference between coexisting sources is recorded in
`schema_audit.json`.  Missing columns and invalid booleans fail closed.

## Local test

Use a Python environment containing NumPy, pandas, and Matplotlib:

```bash
python scripts/v0/ratio_estimator/test_analyze_ratio_estimator.py
```

The tests cover hand-computable geometry, exact reconstruction, direction
attenuation correction, small-denominator fallback, invalid values, clipping,
query split isolation and order invariance, schema rejection, and formal
exact-distance closure failure.

## q100 smoke diagnostic

The existing 100-query, 1/16-sampled cap input is suitable only for smoke
diagnosis:

```bash
python scripts/v0/ratio_estimator/analyze_ratio_estimator.py \
  --input /path/to/cap_diagnostic_input.csv \
  --output-dir /path/to/results/v0_ratio_estimator/gist1m/RUN_ID/phase1 \
  --mode smoke
```

`SMOKE_ONLY` never authorizes Phase 2.  Smoke mode preserves and reports
closure failures but does not turn the result into a formal Phase-1 decision.

## Formal q1000 diagnostic

Export at least 1,000 queries with deterministic `sample_modulus=16`, then run:

```bash
python scripts/v0/ratio_estimator/analyze_ratio_estimator.py \
  --input /path/to/q1000/cap_diagnostic_input.csv \
  --output-dir /path/to/results/v0_ratio_estimator/gist1m/RUN_ID/phase1 \
  --mode formal
```

Formal mode uses the plan's mixed closure tolerance:

```text
abs(a-b) <= 1e-10 + 1e-9 * max(abs(a), abs(b))
```

Any closure failure makes the process exit non-zero with status
`FAIL_CLOSED`.  Do not relax this tolerance to obtain a Go result; fix the
export precision or the underlying geometry instead.  A selected `kappa_min`
may be frozen explicitly with `--selected-kappa-min` after train-only review.

## Outputs

The analyzer writes:

```text
run_manifest.json
query_split_manifest.json
schema_audit.json
estimator_record_summary.csv
estimator_query_summary.csv
conditional_bias_by_kappa.csv
conditional_bias_by_rho.csv
conditional_bias_by_edge_length.csv
conditional_bias_by_x_norm.csv
conditional_bias_by_margin.csv
invalid_records.csv
kappa_min_scan.csv
summary.json
figures/
  rho_error_histogram.png
  distance_error_histogram.png
  raw_vs_ratio_mae.png
  bias_vs_kappa.png
  bias_vs_margin.png
  predicted_vs_true_distance.png
PHASE1_ESTIMATOR_DIAGNOSTIC_REPORT.md
```

The split is created from sorted query IDs followed by a seed-42 shuffle.
Kappa selection and gate metrics use train records only.  Calibration and test
IDs remain assigned in the frozen manifest for later phases.

## SLURM

Set `REPO_ROOT`, `INPUT_CSV`, and `OUTPUT_DIR`, then submit:

```bash
sbatch scripts/v0/ratio_estimator/run_ratio_estimator_diagnostic.slurm
```

Optional variables are `PYTHON_BIN`, `ANALYSIS_MODE`, and
`KAPPA_MIN_CANDIDATES`.  The job runs the synthetic tests before analysis and
writes SHA-256 values for the principal artifacts.

## Phase 2 empirical bound calibration

Phase 2 consumes the exact input CSV and the frozen Phase-1 directory. It
builds independent record-level and query-level split-conformal calibrators
from calibration queries, freezes their hashes, and only then applies them to
the held-out test queries:

```bash
python scripts/v0/ratio_estimator/calibrate_probabilistic_bound.py \
  --input /path/to/cap_diagnostic_input.csv \
  --phase1-dir /path/to/RUN_ID/phase1 \
  --output-dir /path/to/RUN_ID/phase2

python scripts/v0/ratio_estimator/apply_probabilistic_bound.py \
  --input /path/to/cap_diagnostic_input.csv \
  --phase1-dir /path/to/RUN_ID/phase1 \
  --calibration-dir /path/to/RUN_ID/phase2 \
  --output-dir /path/to/RUN_ID/phase2
```

The default risks are `1e-1,1e-2,1e-3,1e-4,1e-5`. A work point whose
finite-sample conformal index exceeds its calibration sample count is emitted
as `unsupported_sample_size`; it is never silently replaced by the sample
maximum. Records that fail the frozen `kappa_min` or validity gate use the
existing lower bound and are counted explicitly.

The application reports record/query violation rates with exact 95%
Clopper-Pearson intervals, violation magnitudes, fallback usage, and a frozen
`GO_TO_PHASE3` or `NO_GO_EMPIRICAL_BOUND` decision. These are held-out
empirical guarantees for the sampled search distribution, not deterministic
lower bounds under a changed HNSW control flow.

For SLURM, set `REPO_ROOT`, `INPUT_CSV`, `PHASE1_DIR`, and `OUTPUT_DIR`, then:

```bash
sbatch scripts/v0/ratio_estimator/run_phase2_calibration.slurm
```

The job runs both synthetic test suites before calibration and writes a
SHA-256 inventory for all Phase-2 artifacts.

## Phase 3 offline pruning simulation

Phase 3 reads the exact frozen Phase-2 calibrator files and applies them to
the frozen test split. It never accepts a quantile on the command line and
never changes HNSW traversal. A simulated prune uses the strict runtime
comparison `LB > threshold`; invalid or low-kappa records use the existing
current lower bound and remain in every denominator.

```bash
python scripts/v0/ratio_estimator/simulate_probabilistic_pruning.py \
  --input /path/to/RUN_ID/raw/cap_diagnostic_input.csv \
  --phase1-dir /path/to/RUN_ID/phase1 \
  --phase2-dir /path/to/RUN_ID/phase2 \
  --output-dir /path/to/RUN_ID/phase3 \
  --metadata /path/to/RUN_ID/raw/metadata.json \
  --query-metrics /path/to/RUN_ID/raw/query_metrics.csv
```

The report compares current LB, optional cap LB, the raw point estimate,
record/query calibrated bounds, and the oracle on identical sampled records.
It reports potential (not realized) exact-distance savings, extra coverage,
oracle recovery, false-prune event/decision/query exposure, margin behavior,
query concentration, and kappa-conditioned precision. Adjacent kappa values
are sensitivity-only and cannot affect the formal decision or selected points.

At most three Phase-4 candidates are frozen as conservative, balanced, and
aggressive. Since this selection compares held-out Phase-3 outcomes, Phase 4
must use fresh final-evaluation queries. A Phase-3 Go means only that potential
record-level benefit exists on the unchanged observe-only trajectory; it is
not a recall or speedup result.

For SLURM, set `REPO_ROOT`, `INPUT_CSV`, `PHASE1_DIR`, `PHASE2_DIR`, and
`OUTPUT_DIR`, then submit:

```bash
sbatch scripts/v0/ratio_estimator/run_phase3_simulation.slurm
```

Optional variables are `PYTHON_BIN`, `RAW_METADATA`, `QUERY_METRICS`, and
`SENSITIVITY_KAPPA`. The job runs all three synthetic suites and hashes every
Phase-3 artifact even when the valid scientific decision is No-Go (exit 3).
