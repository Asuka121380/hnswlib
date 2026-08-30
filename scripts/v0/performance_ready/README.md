# V0 performance-ready runner

The performance build deliberately keeps the V0 feature compiled in for every
method. `baseline`, `approx-no-retry`, and `approx-retry` are selected at run
time in the same executable, so compiler flags, index loading, query loading,
timing, and result serialization are identical. The baseline method does not
load or access a sidecar.

Before collecting timing data, validate the fresh build directory:

```text
python scripts/v0/performance_ready/check_build_contract.py \
  --build-dir build-v0-performance-ready --contract performance
```

Also run `audit_fast_path_source.py --header hnswlib/edge_quant_v0.h`.
It fails if strict-only rounding, padding, checked codebook decoding, or the
strict bound result type leaks back into the `raw_fast_v1` LUT/evaluator.

Run one method per process. Use the same query slice, `k`, `ef-search`, warmup,
repeat count, and optional CPU pin for every method. Approximate methods require
an explicitly calibrated `--beta`; the runner rejects beta values below one.
The JSON is written only after all timed repetitions finish.

```text
v0_performance_runner --method baseline --index-path INDEX \
  --query-path QUERY.fvecs --dimension 128 --query-count 10000 \
  --output baseline.json --repeats 5

v0_performance_runner --method approx-no-retry --index-path INDEX \
  --sidecar-path INDEX.v0eq --query-path QUERY.fvecs --dimension 128 \
  --query-count 10000 --beta CALIBRATED_BETA --output approx.json \
  --repeats 5 --prefetch gate
```

`raw_fast_v1` changes the numerical kernel, so beta values calibrated for the
strict/reference estimator must not be reused for formal performance claims.

For the formal warm-cache block experiment, use `run_latin_square.py`. It
freezes a seeded balanced six-order schedule before the first process starts,
runs each method in an isolated process, hashes the runner/CMake cache/results,
and updates `manifest.json` after every completed position. Use
`--aa-baseline` first to detect a stable order effect without changing the
schedule machinery.

`v0_fast_kernel_microbenchmark` produces the component costs required by the
plan. Combine its JSON with an active metrics `summary.json` using
`evaluate_break_even.py`; the script applies the frozen 25% safety-headroom
gate and exits nonzero when the estimated overhead exceeds the allowance.

Submit dense-beta calibration through `run_calibration.slurm`. It requires a
new `RUN_ROOT` and the exact `EXPECTED_COMMIT`, supports restart by reusing only
completed run directories, and calls `summarize_calibration.py` after all
no-retry/retry configurations finish. Set `RUN_ROLE=validation` or
`RUN_ROLE=heldout` for later frozen splits so run IDs, summaries, and completion
markers preserve their scientific role.

Use `submit_tradeoff.sh` for the frozen trade-off experiment instead of
manually exporting query ranges or beta lists. The wrapper fixes the validation
split to queries 300--599, the held-out split to queries 600--999, and both to
the reviewed beta grid from 1.30 through 1.70. It rejects dirty worktrees and
existing result roots, derives `EXPECTED_COMMIT` from HEAD, and writes the
submission parameters and Slurm job ID to `submission.env`.

```text
bash scripts/v0/performance_ready/submit_tradeoff.sh validation
bash scripts/v0/performance_ready/submit_tradeoff.sh heldout
```

Pass an explicit second argument when a new result-root version is required.
The submitted `run_tradeoff.slurm` delegates execution and summarization to
`run_calibration.slurm`. Both submission and execution source the frozen
`tradeoff_matrix.sh`, so the experiment matrix is not accepted from the shell
environment and can change only in a reviewed repository commit.

For the complete exploratory Recall/DCO/QPS curve, use
`submit_full_beta_tradeoff.sh`. It submits three dependency-linked jobs: a
reference metrics sweep, an instrumentation-free QPS sweep, and a final join.
Both sweeps use all 1,000 frozen GIST queries and one shared 25-point beta grid:
0.05 steps from 1.00 through 2.00 plus 1.525, 1.575, 1.625, and 1.675.

The metrics job executes 50 active configurations (25 beta values by retry
mode). The QPS job runs on an exclusive node and uses five seeded randomized
complete blocks. Every block contains one baseline plus all 100 combinations
of beta, retry mode, and legacy/gate-aware prefetch; every process performs
five timed repetitions after warmup. Metrics and timing remain in separate
binaries so Recall collection cannot contaminate QPS.

```text
bash scripts/v0/performance_ready/submit_full_beta_tradeoff.sh
```

The final dependency job writes `full_beta_tradeoff.csv` and
`full_beta_tradeoff_report.json`. The CSV reports paired block-level QPS
speedup confidence intervals alongside Recall, exact-DCO reduction, retry,
and latency percentiles. `qps/manifest.json` is updated atomically after each
process and supports reuse of checksum-verified completed results if the QPS
job must be resumed under the identical contract.

## Configurable QPS experiments

The frozen `tradeoff_matrix.sh` and `full_beta_tradeoff_matrix.sh` workflows
remain the reproduction entry points for historical experiments. Do not pass
environment overrides to those workflows: their repository-reviewed values
are intentionally authoritative.

Use `submit_qps_experiment.sh` for new targeted or exploratory QPS matrices.
Its only algorithm-parameter source is a version-controlled JSON contract.
The contract can define either a Cartesian product of `betas`, `modes`, and
`prefetches`, or an explicit `cases` array; mixing the two forms is rejected.

Validate the formal ef500 targeted contract without submitting a job:

```text
python scripts/v0/performance_ready/qps_config.py \
  --config configs/v0/qps/ef500_targeted.json \
  --resource-profile formal-exclusive

bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/ef500_targeted.json \
  --resource-profile formal-exclusive \
  --dry-run
```

The checked-in ef500 contract resolves to 12 active configurations plus one
baseline in each of five blocks, for exactly 65 process-result JSON files.
The submit preview prints `ef_search`, the active configuration count, blocks,
the expected result count, resource profile, commit, build, and output root.
Review that preview before removing `--dry-run`.

Use the shared profile only for exploratory screening:

```text
bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/beta_130_140_scan.json \
  --resource-profile exploratory-shared \
  --dry-run
```

`formal-exclusive` passes `--exclusive` to `sbatch`. The runner remains
single-threaded, but the exclusive allocation prevents another job from
contending for the node's caches and memory bandwidth. `exploratory-shared`
passes `--oversubscribe`, requests one CPU, and must not be used for formal
speedup claims or mixed with exclusive results. A configuration whose role is
`formal` is rejected under the shared profile.

For an actual submission, pass an explicit new root when a stable experiment
name is useful:

```text
bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/ef500_targeted.json \
  --resource-profile formal-exclusive \
  --run-root "$HOME/IndividualProject/results/v0_performance_ready/ef500-targeted-v1"
```

An interrupted experiment can be resumed only with the same configuration,
resource profile, commit, runner, CMake cache, and input-file identities:

```text
bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/ef500_targeted.json \
  --resource-profile formal-exclusive \
  --run-root "$HOME/IndividualProject/results/v0_performance_ready/ef500-targeted-v1" \
  --resume
```

Every generic run root contains `requested_config.json`,
`resolved_config.json`, `build_contract.json`, `manifest.json`, per-process
JSON under `qps/`, and the generated `qps_summary.csv` and
`qps_summary.json`. The manifest records the resolved contract hash, binary and
CMake-cache checksums, input identities, Slurm allocation observations, the
exact command for every process, and every result checksum. `COMPLETE` is
written only after the expected result count has been verified.
