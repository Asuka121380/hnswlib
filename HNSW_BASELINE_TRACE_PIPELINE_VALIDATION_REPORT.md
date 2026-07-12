# HNSW Baseline Trace Pipeline Validation Report

## 1. Purpose and conclusion

This report maps the local high-dimensional, small-data validation run to HNSW_Baseline_Trace_Experiment_Brief.md.

Pipeline readiness: **PASS**.

The run proves that the local pipeline builds, preserves baseline HNSW results, records internally consistent level-0 query/DCO data, computes exact ground truth, and produces CSV, Parquet, statistics, PCA, reports, and plots end to end.

Method-feasibility readiness: **NOT YET DECIDED**.

The synthetic validation data cannot establish that edge-quantized routing will accelerate real HNSW workloads. That requires real datasets, multiple experimental axes, isolated cost benchmarks, and a final Conditions A-E analysis.

## 2. Source and execution

| Item | Value |
| --- | --- |
| Branch | baseline-trace |
| Baseline commit | d9b3608 |
| Trace schema | version 1 |
| Distance | float32 squared L2 |
| Trace flag | HNSWLIB_ENABLE_BASELINE_TRACE=ON |
| Validation date | 2026-07-12 |

Run command:

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\baseline_trace\run_pipeline_validation.ps1
~~~

The script configures and builds the trace targets, runs the C++ test, generates high-dimensional data, builds HNSW, compares ordinary and traced searches, computes ground truth, writes trace files, validates identities, converts data to Parquet, and generates statistics and plots.

A second run with DcoSampleModulus=5 also passed, validating deterministic sampled-DCO mode.

## 3. Dataset and HNSW parameters

| Parameter | Value |
| --- | ---: |
| Dataset | synthetic_high_dim_validation |
| Base vectors | 2,000 |
| Queries | 32 |
| Dimension | 1,024 |
| k | 10 |
| efSearch | 100 |
| M | 16 |
| efConstruction | 100 |
| Seed | 42 |
| DCO sample modulus | 1, full trace |
| Edge-direction samples | 256 |
| Index build time | 506.881 ms |

Queries are deterministic perturbations of selected base vectors. This fixture validates pipeline behavior, not a real workload distribution.

## 4. Evidence that the pipeline completed

### 4.1 Build

- All original CMake targets built with tracing enabled.
- All original CMake targets built with tracing disabled.
- The trace-disabled build preserves the original search API.
- The D=1024 baseline trace C++ test passed.

### 4.2 Baseline behavior

For all 32 queries, ordinary searchKnn and traced searchKnnWithTrace results were compared by queue size, distance, label, and order.

No mismatch occurred. Instrumentation did not change the returned search result.

### 4.3 Accounting identities

~~~text
N_edge_scan = N_duplicate + N_unique_neighbor
82,754      = 43,661     + 39,093

N_unique_neighbor = N_dist
39,093           = 39,093

full DCO rows = N_dist
39,093         = 39,093
~~~

Validation flags:

~~~text
edge_accounting_ok     = true
distance_accounting_ok = true
~~~

### 4.4 Numerical identities

| Check | Maximum error | Result |
| --- | ---: | --- |
| Margin identity | 4.55e-13 | pass |
| Edge distance reconstruction | 3.28e-4 relative | pass |

The independent D=1024 C++ test observed maximum relative geometry error 1.98e-4.

### 4.5 Artifact chain

All expected raw files, Parquet files, reports, statistics, PCA output, and plots were generated. The raw DCO CSV is approximately 10.6 MB and its Parquet conversion approximately 3.0 MB.

## 5. Correspondence to the experiment brief

### 5.1 Per-query fields

The query file records:

~~~text
query_id, requested_k, ef_search, ef_effective, dimension, seed
n_entry_distance, n_upper_edge_scan, n_upper_dist, n_base_entry_distance
n_edge_scan, n_duplicate, n_unique_neighbor, n_dist, n_expanded
n_inserted_candidate, n_inserted_result, n_threshold_changed
n_expanded_later, n_final_topk, n_strict_state_neutral
trace_records_written, n_exact_calls_total
baseline_query_latency_ns, trace_query_latency_ns
exact_distance_time_ns, recall_at_k
~~~

All required level-0 counts, queue-event counts, latency, exact-distance time, and recall are present. Upper-layer work is aggregate-only.

### 5.2 Per-DCO fields

The level-0 DCO file contains:

~~~text
query_id, graph_layer, expansion_index, dco_index
current_node_id, neighbor_id
dist_qc, dist_qd, threshold_before, threshold_after
candidate_queue_size_before/after
result_queue_size_before/after
inserted_candidate_queue, inserted_result_queue
threshold_changed, expanded_later, in_final_topk
edge_length_cd, edge_dot_qcd, edge_cosine_qcd
triangle_lower_bound
absolute_margin, relative_margin
is_negative_at_evaluation, is_state_neutral
search_progress_fraction, threshold_stability_indicator
~~~

Additional fields include external labels, threshold-valid flags, and geometry-valid flags. Detailed rows currently have graph_layer=0.

## 6. Workload results

| Metric | Total | Mean/query | Median | p90 | p95 | p99 | Min-max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N_edge_scan | 82,754 | 2,586.06 | 2,591 | 2,662.2 | 2,668.8 | 2,681.35 | 2,457-2,686 |
| N_duplicate | 43,661 | 1,364.41 | 1,363.5 | 1,421.7 | 1,440.35 | 1,442 | 1,281-1,442 |
| N_dist | 39,093 | 1,221.66 | 1,224.5 | 1,248.9 | 1,254.6 | 1,267.28 | 1,173-1,271 |
| N_expanded | 3,209 | 100.28 | 100 | 101 | 101 | 103.07 | 100-104 |
| All exact calls | 40,562 | 1,267.56 | 1,270 | 1,294 | 1,295 | 1,308.11 | 1,226-1,314 |

Exact-call sources:

| Source | Count |
| --- | ---: |
| Initial query entry | 32 |
| Upper-layer neighbors | 1,405 |
| Level-0 entry | 32 |
| Level-0 edge DCO | 39,093 |

The central edge-method amortization count is level-0 N_dist: 1,221.66 per query in this fixture.

## 7. State-neutral and candidate importance

| Class | Count | Fraction |
| --- | ---: | ---: |
| Immediate reject / strict neutral | 30,294 | 77.49% |
| Inserted but never expanded | 5,622 | 14.38% |
| Expanded but not final | 2,873 | 7.35% |
| Final top-k DCO | 304 | 0.78% |

| Event | Count | Fraction |
| --- | ---: | ---: |
| Inserted candidate | 8,799 | 22.51% |
| Inserted result | 8,799 | 22.51% |
| Threshold changed | 5,662 | 14.48% |
| Later expanded | 3,177 | 8.13% |

The calls returned 320 entries. The n_final_topk value of 304 counts final entries originating from edge DCOs; 16 came from initialization paths.

## 8. Margin results

Threshold was valid before 35,925 of 39,093 DCOs, or 91.90%.

### All threshold-valid DCOs

| Metric | Value |
| --- | ---: |
| Mean margin | 4.18% |
| Median | 4.32% |
| p90 | 9.65% |
| p95 | 11.17% |
| p99 | 14.03% |
| Greater than 1% | 78.53% |
| Greater than 5% | 43.55% |
| Greater than 10% | 8.55% |
| Absolute margin below 1% | 10.20% |

### Strict state-neutral DCOs

| Metric | Value |
| --- | ---: |
| Samples | 30,294 |
| Mean margin | 5.52% |
| Median | 5.15% |
| p90 | 10.04% |
| p95 | 11.51% |
| p99 | 14.37% |
| Greater than 1% | 93.12% |
| Greater than 5% | 51.64% |
| Greater than 10% | 10.14% |
| Absolute margin below 1% | 6.88% |

These results prove the margin analysis works. They are not real-dataset evidence.

## 9. Threshold evolution

Stability is the earliest suffix whose thresholds remain within 1% of the final threshold.

| Stage | DCO count | Neutral ratio | Median margin | Margin above 5% |
| --- | ---: | ---: | ---: | ---: |
| Before stability | 16,147 | 53.58% | 1.83% | 22.05% |
| Stable suffix | 22,946 | 94.32% | 5.55% | 55.70% |

The stable suffix contains 58.70% of all DCOs.

## 10. Triangle bound and edge angle

| Measurement | Result |
| --- | ---: |
| Triangle opportunities | 0 / 35,925 |
| Triangle rate | 0% |
| Cosine-margin Pearson | -0.589 |
| Cosine-margin Spearman | -0.645 |
| Edge length mean | 42.77 |
| Edge length median | 42.74 |
| Edge cosine mean | 0.447 |
| Edge cosine median | 0.454 |

Because thresholds are squared distances, the implemented triangle test is triangle_lower_bound squared greater than the squared threshold.

A zero triangle rate is a fixture result, not a pipeline failure.

## 11. Edge-direction PCA

The runner wrote 256 normalized edge directions with 1,024 components.

~~~text
First 32 components cumulative explained variance = 58.73%
~~~

The sampler is deterministic but not yet stratified by graph layer, degree, or access frequency.

## 12. Recall and timing

### Recall

| Metric | Value |
| --- | ---: |
| Mean recall at 10 | 0.98125 |
| Median | 1.0 |
| Minimum | 0.9 |
| Maximum | 1.0 |

314 of 320 requested neighbor slots matched brute-force ground truth.

### Correctness-run timing

| Metric | Mean/query | Median | p95 | Min-max |
| --- | ---: | ---: | ---: | ---: |
| Baseline call | 409.6 us | 388.7 us | 584.2 us | 307.7-614.1 us |
| Full trace call | 2,293.1 us | 2,189.8 us | 2,720.0 us | 1,929.6-3,651.5 us |
| Timed distance bodies | 372.0 us | 368.0 us | 446.0 us | 300.3-487.4 us |

Trace is approximately 5.79 times baseline in this correctness binary. This is instrumentation overhead, not baseline ANN performance.

## 13. Break-even pipeline

The current plot uses provisional values:

~~~text
t_exact  = 420 ns
t_lookup = 80 ns
t_LUT    = 10, 50, and 100 us
~~~

This validates the formula and plotting pipeline. It does not establish Condition E because LUT and lookup costs are assumptions.

## 14. Artifact manifest

The following artifacts are generated locally by the validation script and are intentionally excluded from Git. Paths are listed for reproducibility rather than as repository links.

### Raw data

- results/pipeline_validation/raw/metadata.json
- results/pipeline_validation/raw/query_stats.csv
- results/pipeline_validation/raw/dco_trace.csv
- results/pipeline_validation/raw/edge_directions.csv

### Tables and reports

- results/pipeline_validation/analysis/summary.csv
- results/pipeline_validation/analysis/query_stats.parquet
- results/pipeline_validation/analysis/dco_trace.parquet
- results/pipeline_validation/analysis/validation_report.json
- results/pipeline_validation/analysis/report.md

### Brief-required plots

- results/pipeline_validation/analysis/plots/n_dist_cdf.png
- results/pipeline_validation/analysis/plots/relative_margin_cdf.png
- results/pipeline_validation/analysis/plots/neutral_over_progress.png
- results/pipeline_validation/analysis/plots/threshold_over_progress.png
- results/pipeline_validation/analysis/plots/triangle_bound_rate.png
- results/pipeline_validation/analysis/plots/edge_cosine_vs_margin.png
- results/pipeline_validation/analysis/plots/break_even_vs_n_dist.png
- results/pipeline_validation/analysis/plots/edge_direction_pca.png

## 15. What this proves

1. The environment supports the validation pipeline.
2. Traced and ordinary searches return identical results.
3. Required level-0 counters and DCO scalars are recorded.
4. Deferred expanded and final-top-k labels work.
5. Margin and edge geometry are numerically consistent.
6. Full and sampled DCO modes work.
7. Ground-truth recall is computed.
8. CSV-to-Parquet conversion works.
9. Statistics, stage analysis, break-even curves, correlation, PCA, and plots are generated.
10. Expected artifacts are checked automatically.

## 16. What this does not prove

- The data is synthetic and small.
- Only one default parameter combination is reported.
- Detailed DCO trace covers level 0 only.
- Candidate-class margin and geometry comparisons are incomplete.
- Edge samples are not stratified.
- The requested margin histogram is not yet generated.
- A separate trace-compiled-out performance run is not recorded.
- LUT and lookup costs are not measured.
- Server-scale logging needs sharding, columnar output, or sampling.
- Conditions A-E have not been evaluated on real datasets.

## 17. Readiness decision

### Pipeline

**PASS.** The pipeline is ready as the correctness gate for server experiments.

### Proposed method

**NOT YET DECIDED.** Current smoke signals are 1,221.66 level-0 DCOs/query, 77.49% strict neutral DCOs, and 94.32% late-stage neutral ratio. These are synthetic-fixture observations only.

Before implementing a full edge codebook, server experiments must add real datasets, multiple axes, isolated microbenchmarks, scalable logging, and a final Conditions A-E report.
