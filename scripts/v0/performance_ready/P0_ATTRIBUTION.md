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
4. Submit `p0_attribution_pmu.json` as a separate diagnostic pass. PMU results
   are not formal QPS: enabling/disabling counters makes syscalls at the timed
   region boundaries. Each event reports `available`, `error_number`, and a
   multiplex `running_ratio`; unsupported events do not abort the run.
5. Run `analyze_p0_attribution.py` to join ordered latency, result parity, and
   query metrics. Do not join old sorted latency arrays to query metrics.

All P0 configs require the Python orchestrator to observe exactly one allowed
CPU. `run_qps_matrix.slurm` launches it through `srun --cpu-bind=cores
--mem-bind=local`, so the runner inherits the binding before loading the index.
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
  --partition normal --qos normal --time-limit 00:15:00 --dry-run
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

The formal and PMU runs must use different run roots. A failed PMU capability
probe is recorded as an unavailable diagnostic; it must not be converted into
an inferred cache-miss claim. Generic `cache_misses` is not an L2-miss event;
the separate `l1d_read_misses` event is used only when the target PMU supports
it. Treat heavily multiplexed values as diagnostic scale estimates.
