# P0 performance attribution experiments

P0 freezes the primary operating point at M32,b8, beta 1.45,
no-retry, legacy prefetch, ef=500 and compares it with baseline ef=435.
Baseline ef=500 separates same-search-budget substitution cost from the
matched-recall endpoint. Beta 1.35 retry/ef=500 with baseline ef=360 is the
retry control; beta 1.30 no-retry/ef=500 is an aggressive-pruning diagnostic.

## Experiment order

1. Submit `p0_attribution_aa.json`. Reject the environment if the two
   identical baseline cases show a stable order effect or material drift.
2. Submit `p0_attribution_formal.json`. This writes ordered per-query latency
   CSV files and, after timing has stopped, one top-k result CSV per process.
3. Run the existing quality matrix with the updated reference build to obtain
   per-query active and baseline work ledgers. The ledger now includes graph,
   exact, duplicate, candidate/result queue, and threshold-update counts.
   The P0 quality configuration includes carrier cases for matched baselines
   ef=360 and ef=435; baseline ef=500 is emitted by every ef=500 active case.
   Submit this matrix on the same named node and resource profile as the P0
   timing run. The reference build must disable native architecture flags so
   it is executable on that node.
4. Submit `p0_attribution_pmu.json` as a separate diagnostic pass. PMU results
   are not formal QPS: enabling/disabling counters makes syscalls at the timed
   region boundaries. Each event reports `available`, `error_number`, and a
   multiplex `running_ratio`; unsupported events do not abort the run.
5. Run `analyze_p0_attribution.py` to join ordered latency, result parity, and
   query metrics. Do not join old sorted latency arrays to query metrics.
6. Run P0.3 through `submit_component_microbenchmark.sh` on the same named
   node and resource profile. Use the portable performance build, not the
   reference/metrics build. Seven isolated blocks report medians and block
   confidence intervals. Evaluate primary and retry matched pairs separately;
   pass each matched baseline summary to `evaluate_break_even.py`.
7. Run P0.4 through `p0_prefetch_ab.json`. It holds beta, retry mode, ef,
   queries, build, and sidecar fixed while comparing legacy and gate-aware
   prefetch against the same baseline ef435 in five randomized blocks.

All P0 configs require the Python orchestrator to observe exactly one allowed
CPU. `run_qps_matrix.slurm` launches it through `srun --cpu-bind=threads
--mem-bind=local` and narrows the resulting allocation to its first allowed
logical CPU with `taskset`, so the runner inherits a single logical-CPU binding
before loading the index. The explicit narrowing handles clusters that expose
both SMT siblings despite a one-CPU Slurm request.
For iterative bottleneck attribution, run the same frozen designs with
`exploratory-shared`. This allocates and pins one CPU but permits unrelated jobs
on the node. The resolved contract records `claim_scope=exploratory`; reserve
`formal-exclusive` for the final short confirmation of selected candidates.

Example dry runs:

```text
python scripts/v0/performance_ready/qps_config.py \
  --config configs/v0/qps/p0_attribution_aa.json \
  --resource-profile exploratory-shared --json

bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/p0_attribution_formal.json \
  --resource-profile exploratory-shared \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --time-limit 01:00:00 \
  --memory 24G --dry-run
```

Example analysis:

```text
python scripts/v0/performance_ready/analyze_p0_attribution.py \
  --run-root RESULTS/p0-performance-attribution-formal-v1 \
  --metrics approx-no-retry-beta1p45-legacy-ef500=QUALITY/quality/approx-no-retry-beta1p45-ef500/query_metrics.csv \
  --baseline-metrics baseline-ef500=QUALITY/quality/approx-no-retry-beta1p45-ef500/query_metrics.csv \
  --baseline-metrics baseline-ef435=QUALITY/quality/approx-no-retry-beta1p55-ef435/query_metrics.csv \
  --pair approx-no-retry-beta1p45-legacy-ef500=baseline-ef435 \
  --ground-truth-ivecs DATA/gist_groundtruth.ivecs \
  --quality-summary approx-no-retry-beta1p45-legacy-ef500=QUALITY/quality/approx-no-retry-beta1p45-ef500/summary.json \
  --output-dir RESULTS/p0-analysis
```

Example P0 quality submission matching the exploratory testing environment:

```text
REFERENCE_BUILD="$PWD/build-v0-approx-reference-portable-$(git rev-parse --short=7 HEAD)" \
bash scripts/v0/performance_ready/submit_quality_experiment.sh \
  --config configs/v0/qps/p0_attribution_quality.json \
  --resource-profile exploratory-shared \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --time-limit 01:00:00 --memory 24G \
  --run-root "$HOME/IndividualProject/results/v0_performance_ready/p0-attribution-quality-testing-portable-$(git rev-parse --short=7 HEAD)"
```

Example P0.3 submission matching that environment:

```text
PERFORMANCE_BUILD="$PWD/build-v0-performance-portable-$(git rev-parse --short=7 HEAD)" \
bash scripts/v0/performance_ready/submit_component_microbenchmark.sh \
  --resource-profile exploratory-shared \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --time-limit 00:10:00 --memory 24G \
  --run-root "$HOME/IndividualProject/results/v0_performance_ready/p0-component-microbenchmark-testing-portable-$(git rev-parse --short=7 HEAD)"
```

Example P0.4 dry run matching the P0 testing environment:

```text
PERFORMANCE_BUILD="$PWD/build-v0-performance-portable-$(git rev-parse --short=7 HEAD)" \
bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/p0_prefetch_ab.json \
  --resource-profile exploratory-shared \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --time-limit 01:00:00 --memory 24G \
  --run-root "$HOME/IndividualProject/results/v0_performance_ready/p0-prefetch-ab-testing-portable-$(git rev-parse --short=7 HEAD)" \
  --dry-run
```

Example P0.5 PMU dry run matching the same testing node, portable build,
single logical CPU, and shared exploratory resource profile used by P0.1-P0.4:

```text
PERFORMANCE_BUILD="$PWD/build-v0-performance-portable-$(git rev-parse --short=7 HEAD)" \
bash scripts/v0/performance_ready/submit_qps_experiment.sh \
  --config configs/v0/qps/p0_attribution_pmu.json \
  --resource-profile exploratory-shared \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --time-limit 01:00:00 --memory 24G \
  --run-root "$HOME/IndividualProject/results/v0_performance_ready/p0-attribution-pmu-testing-portable-$(git rev-parse --short=7 HEAD)" \
  --dry-run
```

Remove `--dry-run` only after the preview reports the intended commit, build,
node, and `performance build contract OK`. The completed `qps_summary.csv`
contains per-query medians for all seven requested events, IPC, branch/cache
miss rates, and PMU diagnostics. `pmu_status=partial` means one or more events
were unavailable in at least one block; `pmu_status=multiplexed` means every
event was available but at least one `running_ratio` was below 0.90. In either
case, inspect `pmu_unavailable_events` and `pmu_min_running_ratio` before making
a claim; `pmu_error_numbers` preserves the kernel errno values needed to
distinguish permission failures from unsupported events. A PMU result remains
exploratory even if all events are available.

After the run completes, compare gate directly with legacy as well as both
active cases with the matched baseline. Supplying the ground truth also checks
that the performance-run results preserve Recall@10:

```text
python scripts/v0/performance_ready/analyze_p0_attribution.py \
  --run-root "$RUN_ROOT" \
  --pair approx-no-retry-beta1p45-gate-ef500=approx-no-retry-beta1p45-legacy-ef500 \
  --pair approx-no-retry-beta1p45-legacy-ef500=baseline-ef435 \
  --pair approx-no-retry-beta1p45-gate-ef500=baseline-ef435 \
  --ground-truth-ivecs "$HOME/IndividualProject/datasets/gist1m/gist/gist_groundtruth.ivecs" \
  --output-dir "$RUN_ROOT/analysis"
```

The submission wrapper removes inherited `LD_LIBRARY_PATH` before `sbatch`.
This is part of the P0 environment contract: it prevents a login-shell CMake
module from injecting an older `libstdc++` into a runner built by the system
compiler.

The formal and PMU runs must use different run roots. A failed PMU capability
probe is recorded as an unavailable diagnostic; it must not be converted into
an inferred cache-miss claim. Generic `cache_misses` is not an L2-miss event;
the separate `l1d_read_misses` event is used only when the target PMU supports
it. Treat heavily multiplexed values as diagnostic scale estimates.
