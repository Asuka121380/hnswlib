# HNSW GIST1M Trace Experiment Implementation Plan

## 1. Purpose

This document defines the implementation and execution plan for the second complete real-dataset baseline-trace experiment, using GIST1M.

The experiment extends the completed SIFT1M study from 128-dimensional vectors to 960-dimensional vectors while keeping the database scale and distance family comparable. Its purpose is to determine whether the trace observations that motivate edge approximation are stable on a substantially higher-dimensional real dataset.

This remains an observational experiment. It must use the original HNSW search algorithm and exact squared-L2 distance calculations. It does not implement edge quantization, an approximate distance function, a codebook, a lookup table, or a modified routing decision.

The main research question is:

> Does a high-dimensional, million-scale, real L2 dataset exhibit the same large population of non-influential distance-comparison operations and useful decision margins observed on SIFT1M, together with edge-direction structure that could support a later compact approximation?

## 2. Motivation and relationship to FINGER

FINGER evaluates graph-search acceleration on six ANN benchmark datasets. Its L2 group consists of:

| Dataset | Base vectors | Dimension | Distance |
|---|---:|---:|---|
| FashionMNIST | 60,000 | 784 | L2 |
| SIFT1M | 1,000,000 | 128 | L2 |
| GIST1M | 1,000,000 | 960 | L2 |

GIST1M is the best immediate follow-up to SIFT1M because it preserves the million-vector scale and L2 distance while increasing dimensionality from 128 to 960. This provides a cleaner comparison than switching simultaneously to cosine distance, a different file representation, and a different dataset scale.

FINGER reports that many graph-search distance calculations are larger than the current upper bound and therefore do not update the search. The existing baseline-trace instrumentation measures a closely related phenomenon directly in hnswlib, with additional fields for candidate insertion, threshold changes, later expansion, final-top-k membership, decision margin, and edge geometry.

The GIST1M experiment is not intended to reproduce FINGER's approximation method or speedup. It tests whether the underlying opportunity observed by FINGER and by the SIFT1M trace remains present in a high-dimensional setting relevant to the proposed edge-based method.

## 3. Current baseline

The `baseline-trace` branch already contains:

- optional level-0 trace instrumentation guarded by `HNSWLIB_ENABLE_BASELINE_TRACE`;
- exact per-query DCO accounting;
- deterministic per-DCO sampling;
- threshold, absolute-margin, relative-margin, queue-update, later-expansion, final-top-k, and edge-geometry fields;
- strict `fvecs` and `ivecs` readers;
- supplied-ground-truth Recall@K evaluation;
- persistent HNSW index build/load and manifest validation;
- separate correctness, trace, and trace-disabled performance modes;
- restartable Slurm query shards;
- capacity projection and storage gating;
- bounded-memory aggregation to Parquet;
- per-configuration analysis and cross-`efSearch` summaries;
- a complete SIFT1M run, `sift1m-first-full-20260716`, collected from clean commit `a02f0fd545b8721e01164f532919cc9f53b903dd`;
- the later cross-`efSearch` summary implementation at commit `a666dc3ee745425770a5c2409b80a30aed7bad9c`.

No new C++ vector format or distance implementation is required for native GIST1M files.

## 4. Scope

### 4.1 In scope

1. Download and structurally validate the native GIST1M benchmark files.
2. Create a checksum-pinned GIST1M dataset manifest.
3. Generalize or add dataset-specific preparation and submission entry points.
4. Build or reuse one GIST1M HNSW index for the complete search matrix.
5. Prove trace-on versus ordinary-search equivalence on GIST1M.
6. Run a maximum-`efSearch` pilot and enforce an explicit storage gate.
7. Collect the complete exact-search trace over all 1,000 standard queries.
8. Measure a trace-disabled HNSW performance baseline.
9. Aggregate and analyze every `efSearch` configuration.
10. Produce a direct SIFT1M-versus-GIST1M comparison with explicit counts and denominators.
11. Correct ambiguous margin and neutral-ratio reporting before using the metrics for cross-dataset claims.

### 4.2 Out of scope

- implementing FINGER;
- implementing edge quantization or a quantized HNSW search path;
- choosing a bit width or codebook;
- claiming Recall preservation under approximation;
- claiming an end-to-end speedup;
- GPU execution;
- changing HNSW candidate ordering, queue semantics, visited-set behavior, termination, or returned neighbors;
- adding cosine/angular datasets in the same experimental step.

## 5. Dataset specification

### 5.1 Standard GIST1M split

Use the standard GIST1M/TEXMEX split:

```text
base vectors:        1,000,000
query vectors:       1,000
dimension:           960
ground-truth K:      100
base format:         fvecs
query format:        fvecs
ground-truth format: ivecs
distance:            squared L2 over float32 vectors
```

Expected native files are:

```text
gist_base.fvecs
gist_query.fvecs
gist_groundtruth.ivecs
```

The optional learning set is not required by this baseline-trace experiment and must not be copied into each run directory.

### 5.2 Source

Prefer the original TEXMEX/INRIA native-format distribution because the current runner already consumes its `fvecs` and `ivecs` files. ANN-Benchmarks may be used as an independent reference for standard counts, dimension, metric, and ground truth, but its HDF5 representation would require conversion.

Record the exact source URL, retrieval timestamp, archive size, extracted file sizes, and SHA-256 checksums in the dataset preparation output. A mirror may be used if the original host is unavailable, but the resolved source and checksums must be preserved.

### 5.3 Dataset manifest

Generate, rather than hand-edit, a manifest with this schema:

```json
{
  "dataset": "gist1m",
  "distance_kind": "squared_l2_float32",
  "base_path": "/home/remote/<user>/IndividualProject/datasets/gist1m/gist/gist_base.fvecs",
  "query_path": "/home/remote/<user>/IndividualProject/datasets/gist1m/gist/gist_query.fvecs",
  "ground_truth_path": "/home/remote/<user>/IndividualProject/datasets/gist1m/gist/gist_groundtruth.ivecs",
  "base_format": "fvecs",
  "query_format": "fvecs",
  "ground_truth_format": "ivecs",
  "dimension": 960,
  "n_base": 1000000,
  "n_query": 1000,
  "ground_truth_k": 100,
  "checksums": {
    "base_sha256": "<generated>",
    "query_sha256": "<generated>",
    "ground_truth_sha256": "<generated>"
  }
}
```

The preparation job must fail if record counts, dimensions, prefixes, file lengths, or ground-truth labels are inconsistent.

## 6. Experimental design

### 6.1 Fixed parameters

Keep the first GIST1M matrix aligned with SIFT1M:

```text
k                  = 10
M                  = 16
efConstruction     = 200
efSearch           = 50, 100, 200, 400
seed               = 42
query_start        = 0
query_count        = 1,000
dco_sample_modulus = 10
performance_repeats= 5 initially
warmup_queries     = 100
```

The full standard GIST1M query set contains only 1,000 independent queries. Do not repeat queries to manufacture a 10,000-query run. Query-level uncertainty must instead be reported with confidence intervals or bootstrap intervals.

### 6.2 Initial sharding

Use 20 trace shards unless the pilot shows excessive per-task startup overhead:

```text
20 shards x 50 queries = 1,000 queries
```

This retains the SIFT1M shard count and produces simple, non-overlapping global query ranges. If index loading dominates the 50-query tasks, reduce to 10 shards before the full run and record the resolved value. Do not change the shard count after full trace collection begins.

### 6.3 Edge-direction samples

The existing value of 256 edge samples per shard produces at most 5,120 sampled directions per `efSearch`. At 960 dimensions this is substantially larger than the SIFT1M geometry output but remains bounded.

Retain 256 samples per shard for the first pilot so that the number of sampled directions remains comparable across datasets. The capacity gate may reduce this number only if geometry output or PCA memory is unsafe. Any change must be recorded and accounted for in cross-dataset comparisons.

### 6.4 Performance interpretation

Trace-enabled timings measure instrumentation cost and are not performance evidence. QPS and latency must come from the trace-disabled Release executable.

GIST1M has one tenth as many queries as SIFT1M. Five repeats may therefore be insufficient for a stable performance estimate. Run five repeats initially, calculate the coefficient of variation, and increase to at least ten repeats if it exceeds 5%. Use fixed physical-core binding and record node load where possible.

## 7. Required implementation changes

### 7.1 GIST1M preparation script

Add either:

```text
scripts/baseline_trace/prepare_gist1m.sh
scripts/baseline_trace/slurm/prepare_gist1m.slurm
```

or generalize the existing SIFT-specific preparation code into a dataset-parameterized script while keeping the SIFT1M workflow working.

The preparation code must:

- download into a temporary path;
- refuse to overwrite a completed validated dataset;
- extract on a compute node, not the login node;
- locate the three required GIST files explicitly;
- validate every vector dimension prefix;
- validate exact record counts and file lengths;
- validate ground-truth K and label range;
- calculate SHA-256 incrementally;
- write `dataset.json` atomically only after all checks pass;
- write a completion marker;
- avoid retaining duplicate archives when storage is constrained.

### 7.2 Dataset validation

Reuse `scripts/baseline_trace/validate_dataset.py`. Add a GIST1M-specific test case that asserts:

- `dimension == 960`;
- `n_base == 1,000,000`;
- `n_query == 1,000`;
- `ground_truth_k == 100`;
- `distance_kind == squared_l2_float32`;
- all three SHA-256 values match.

The generic validator must remain authoritative; dataset-name checks supplement rather than replace structural validation.

### 7.3 Experiment configuration

Add:

```text
configs/baseline_trace/experiments/gist1m_first_full.json
```

with the resolved initial matrix. Suggested contents:

```json
{
  "dataset": "gist1m",
  "k": 10,
  "M": 16,
  "ef_construction": 200,
  "ef_search_values": [50, 100, 200, 400],
  "seed": 42,
  "query_start": 0,
  "query_count": 1000,
  "trace_shards": 20,
  "dco_sample_modulus": 10,
  "edge_samples_per_shard": 256,
  "pilot_query_count": 50,
  "correctness_query_count": 10,
  "storage_budget_gb": 60,
  "performance_repeats": 5,
  "warmup_queries": 100
}
```

The storage budget is a hard upper bound, not an expected output size. Reassess it against current quota usage before submission.

### 7.4 Submission entry point

The existing `submit_first_full_experiment.sh` has SIFT-specific defaults and run naming. Prefer generalizing it to accept:

```text
--dataset-config <path>
--experiment-config <path>
--run-id <id>
```

The run ID default should be derived from the experiment configuration or dataset name rather than hard-coded to SIFT1M. Preserve backward compatibility with the completed SIFT1M workflow.

If generalization would add risk immediately before the experiment, add a small `submit_gist1m_full_experiment.sh` wrapper that calls shared submission logic. Do not maintain two independent copies of the Slurm dependency chain.

Before submitting any job, the entry point must verify:

- the experiment dataset name matches the dataset manifest;
- `query_start + query_count <= n_query`;
- `k <= ground_truth_k`;
- shard count is positive and does not exceed query count;
- `efSearch >= k` for every configured value;
- the run directory does not exist;
- the Git commit and dirty state can be recorded.

### 7.5 Margin and neutral-ratio metric corrections

Correct the analysis definitions before generating the GIST1M report so the cross-dataset comparison has unambiguous denominators.

The current trace defines:

```text
absolute_margin = dist_qd - threshold_before
relative_margin = absolute_margin / max(abs(threshold_before), epsilon)
```

Margin is valid only when `threshold_valid_before` is true.

The current `neutral_margin_gt_*` numerator counts valid positive margins, while its denominator is all sampled state-neutral DCOs, including any rows for which the threshold is not yet valid. Replace or supplement it with:

```text
valid_neutral_count
valid_neutral_margin_gt_1pct
valid_neutral_margin_gt_5pct
valid_neutral_margin_gt_10pct
```

where every percentage uses `valid_neutral_count` as its denominator.

Retain the old metrics only if needed for backward compatibility, label them as legacy, and calculate the corrected metrics for both SIFT1M and GIST1M from the existing Parquet files.

Also distinguish:

- `rejected_at_evaluation_ratio`: exact all-DCO ratio for records not inserted into the candidate queue, currently stored as `n_strict_state_neutral / n_dist`;
- `fully_state_neutral_ratio`: sampled ratio satisfying no candidate insertion, no result insertion, no threshold change, no later expansion, and no final-top-k membership.

Do not present these as identical definitions even when their values are numerically close.

### 7.6 Cross-dataset summary

Add a script such as:

```text
scripts/baseline_trace/cross_dataset_summary.py
```

It must read only compact per-configuration metrics and performance files, plus immutable run provenance. It must not rescan raw CSV shards.

At minimum, emit:

```text
summary.csv
summary.json
report.md
plots/
```

Each row must include dataset, dimension, base count, query count, `efSearch`, metric definitions/schema version, trace commit, summary commit, sampling rules, and explicit counts used as denominators.

### 7.7 Provenance enhancement

The completed SIFT1M experiment separately records the experiment commit and later summary code. Preserve this distinction for GIST1M:

```text
experiment_commit
experiment_git_status
analysis_commit
dataset checksums
index checksum
experiment config
dataset manifest
Slurm job IDs
```

Copy commit and checksum values into the final machine-readable summary rather than referring only to neighboring files.

## 8. Validation and execution gates

### Gate A: repository freeze

- work from `baseline-trace` or a dedicated GIST experiment branch based on it;
- commit all experiment-affecting changes;
- ensure `git status --short` is empty before submission;
- push the exact commit used on the cluster;
- record both the commit hash and dirty status.

Failure blocks all formal jobs.

### Gate B: dataset integrity

- all three files exist;
- dimension prefixes are consistently 960;
- base count is 1,000,000;
- query count is 1,000;
- ground-truth K is 100;
- every ground-truth label is in `[0, 1,000,000)`;
- file sizes agree with record layouts;
- SHA-256 checksums are recorded and revalidated.

Failure blocks index construction.

### Gate C: build and index integrity

- clean checkout produces trace-on and trace-off Release binaries;
- the index is built once with `M=16`, `efConstruction=200`, seed 42;
- index manifest matches the GIST1M dataset manifest;
- index byte size and SHA-256 are recorded;
- loading the saved index succeeds;
- no SIFT1M index is accidentally reused.

Failure blocks search jobs.

### Gate D: search neutrality

Run ten deterministic queries with full DCO sampling (`modulus=1`) and geometry enabled. Require:

- ordinary and traced priority queues match exactly;
- Recall@10 is computed from supplied ground truth;
- `n_edge_scan == n_duplicate + n_unique_neighbor`;
- `n_unique_neighbor == n_dist`;
- every sampled valid margin satisfies the margin identity;
- every valid geometry row satisfies the squared-L2 reconstruction identity within tolerance;
- output completion markers appear only after successful file closure.

Failure blocks pilot and full trace jobs.

### Gate E: capacity pilot

Run 50 queries at `efSearch=400`, using the proposed 1/10 DCO sampling and 256 edge samples per shard. Measure:

- mean and distribution of DCOs per query;
- sampled DCO bytes per query;
- edge-direction bytes per query and per shard;
- wall time per query;
- index-load overhead;
- peak RSS;
- projected total raw and aggregated size for all four `efSearch` values;
- projected total Slurm CPU time.

The capacity estimator must use actual on-disk byte counts. Accept the full run only when projected output fits both the configured 60 GB gate and the actual remaining quota with a safety margin.

Failure requires changing query/shard/geometry sampling configuration and rerunning the pilot. Do not silently change trace semantics.

### Gate F: shard integrity

For every `efSearch`:

- exactly the configured number of completion markers exists;
- global query ranges cover `[0, 1000)` exactly once;
- no duplicate query IDs exist;
- DCO row counts reconcile with deterministic sampling;
- metadata agrees on dataset, parameters, run ID, commit, and schema;
- aggregation is deterministic and bounded-memory.

Failure blocks analysis and summary claims.

### Gate G: analysis integrity

- corrected margin denominators are present;
- rejected-at-evaluation and fully-state-neutral ratios are separately named;
- all percentages report their counts and denominators;
- PCA reports sample count, dimension, number of components, and cumulative explained variance;
- trace and trace-disabled Recall agree at every `efSearch`;
- performance variability is reported;
- the report preserves the observational-only interpretation boundary.

## 9. Slurm workflow

Use the existing WEIRDO resource rules and dependency handling:

```text
GIST1M prepare/validate --------+
                                |
trace-on/off Release build -----+
                                |
                                v
                      index build or validated reuse
                                |
                                v
                       correctness smoke (10 queries)
                                |
                                v
                       efSearch=400 pilot (50 queries)
                                |
                                v
                         capacity budget gate
                                |
                 +--------------+--------------+
                 |              |              |
                 v              v              v
          trace arrays      performance     remaining ef values
                 |              |              |
                 +--------------+--------------+
                                v
                    aggregate and analyze by ef
                                |
                                v
                    cross-efSearch GIST summary
                                |
                                v
                    SIFT1M/GIST1M comparison
```

Heavy download, extraction, checksum, build, index, trace, conversion, analysis, and performance work must run through Slurm. The login node is limited to repository updates, lightweight inspection, submission, and monitoring.

## 10. Expected run layout

```text
results/baseline_trace/<gist-run-id>/
  config/
    dataset.json
    experiment.json
    git_commit.txt
    git_status.txt
  index/
  logs/
  submitted_jobs.txt
  correctness/
  pilot/
  ef50/
    raw/shards/part-*/
    aggregated/
    analysis/
  ef100/
  ef200/
  ef400/
  performance/
    ef50/performance.csv
    ef100/performance.csv
    ef200/performance.csv
    ef400/performance.csv
  summary/
    summary.csv
    summary.json
    report.md
    plots/
```

The dataset and reusable index remain outside the run directory. The run config contains immutable manifests and checksums rather than duplicate data.

## 11. Required analyses

### 11.1 Per-configuration results

For each `efSearch`, report:

- Recall@10;
- mean, median, P95, and P99 `N_dist` per query;
- exact rejected-at-evaluation count and ratio;
- sampled fully-state-neutral count and ratio;
- valid-neutral margin count;
- positive relative-margin CDF for valid neutral DCOs;
- valid-neutral margin fractions above 1%, 5%, and 10%;
- threshold-valid fraction and threshold stabilization behavior;
- final-top-k, later-expanded, queue-inserted, and threshold-changing counts;
- edge-direction sample count;
- PCA explained variance by component and cumulative variance for the fixed component budget;
- QPS, total latency, per-query latency, standard deviation, and coefficient of variation from trace-disabled runs.

### 11.2 Search-progress analysis

FINGER's motivating observation is expressed over greedy-search progress. Add or retain decile-based plots for:

- rejected-at-evaluation ratio;
- fully-state-neutral ratio;
- threshold-valid ratio;
- median relative margin among valid rejected DCOs;
- candidate insertion rate;
- threshold change rate.

Use DCO-index progress within each query and report the number of contributing DCOs in every bin. Do not infer causality from a plot based only on pooled rows.

### 11.3 SIFT1M versus GIST1M comparison

Compare like-for-like configurations at `efSearch=50,100,200,400`:

| Category | Required comparison |
|---|---|
| Search quality | Recall@10 |
| Search work | `N_dist` distribution |
| Opportunity volume | rejected and fully-neutral DCO ratios |
| Decision robustness signal | valid-neutral relative-margin CDF and thresholds |
| Search dynamics | progress-binned rejection and threshold behavior |
| Edge structure | PCA curve with equal sample budget and component count |
| Cost | trace-disabled QPS and latency |
| Provenance | dataset, dimension, counts, commits, checksums, sampling rules |

Do not interpret a larger relative margin as a directly allowable vector-quantization error. Margin is a local distance-decision allowance relative to the current HNSW threshold, not a measured representation error.

### 11.4 Statistical reporting

Because GIST1M has 1,000 queries:

- use query-level bootstrap confidence intervals for Recall and `N_dist` summaries;
- report raw DCO counts as well as ratios;
- avoid treating millions of DCOs from the same queries as independent samples;
- where possible, calculate per-query neutral ratios before reporting their distribution;
- preserve pooled ratios for workload-volume estimates, clearly labeled as pooled.

## 12. Interpretation boundary

The experiment may establish that:

- high-dimensional GIST1M also contains a large population of exact distance calculations that do not update HNSW search state;
- many valid rejected comparisons are far from the current threshold;
- these patterns are stable or unstable across `efSearch` and across SIFT1M/GIST1M;
- high-dimensional edge directions exhibit measurable low-rank or other geometric structure;
- exact high-dimensional distance calculations constitute a larger performance cost than on SIFT1M.

The experiment cannot establish:

- that a particular edge quantizer is accurate;
- that a particular margin guarantees end-to-end search equivalence;
- a safe number of bits;
- Recall preservation under approximation;
- a realized lookup/decode break-even point;
- an end-to-end speedup.

All final reports must retain this distinction.

## 13. Risks and mitigations

### 13.1 Storage amplification

Each sampled edge direction has 960 components. Geometry CSV can be substantially larger than on SIFT1M even with fewer queries.

Mitigation: retain bounded reservoir sampling, measure actual pilot bytes, convert incrementally to Parquet, and enforce the storage gate.

### 13.2 Query-count mismatch

GIST1M has 1,000 standard queries, not 10,000.

Mitigation: use all 1,000 exactly once, reject invalid ranges at submission, and use query-level confidence intervals.

### 13.3 Index load and memory cost

The raw base matrix alone contains 960 million float32 values. Trace mode also loads base vectors for geometry.

Mitigation: request memory based on pilot peak RSS, avoid duplicate base copies, reuse one index, and account for index-load overhead when selecting shard count.

### 13.4 Dataset-source instability

The original dataset host or archive layout may change.

Mitigation: record the resolved URL and checksums, validate by content rather than archive path alone, and support an explicitly recorded mirror.

### 13.5 Misleading cross-dataset ratios

Different dimensions, query counts, DCO counts, and threshold-valid fractions can make pooled percentages appear directly comparable when they are not.

Mitigation: report counts and denominators, separate pooled from per-query statistics, fix metric definitions, and use the same `efSearch` and sampling rules.

### 13.6 Performance noise

The first SIFT1M run showed elevated variability at high `efSearch`.

Mitigation: pin physical cores, warm the index, record system context, increase repeats when CV exceeds 5%, and avoid publication-quality speed claims from unstable measurements.

## 14. Implementation milestones

### Milestone 1: metric and provenance readiness

- correct valid-neutral margin denominators;
- clarify neutral metric names;
- embed experiment and analysis provenance in summaries;
- regenerate corrected SIFT1M compact metrics for comparison.

### Milestone 2: GIST1M dataset support

- add preparation script and Slurm wrapper;
- generate checksum-pinned dataset manifest;
- add the GIST1M experiment config;
- validate native files and ground truth.

### Milestone 3: workflow generalization

- parameterize dataset and run naming;
- add submission preflight checks;
- preserve SIFT1M backward compatibility;
- validate job dependency and completion-marker behavior locally where possible.

### Milestone 4: GIST1M gates

- build trace-on/off binaries from a clean commit;
- build and checksum the GIST1M index;
- pass ten-query search-neutrality correctness;
- pass the `efSearch=400` capacity pilot;
- freeze the resolved full-run configuration.

### Milestone 5: complete GIST1M experiment

- run all four trace arrays;
- run trace-disabled performance baselines;
- aggregate and validate all shards;
- produce per-configuration and cross-`efSearch` reports.

### Milestone 6: cross-dataset feasibility report

- compare corrected SIFT1M and GIST1M metrics;
- report search-progress behavior and uncertainty;
- identify which observations generalize across datasets;
- state whether evidence supports another observational dataset, a refined trace, or later approximation implementation.

## 15. Definition of done

The GIST1M experiment is complete only when:

1. The dataset manifest validates 1,000,000 base vectors, 1,000 queries, 960 dimensions, top-100 ground truth, and all checksums.
2. A clean, pushed Git commit builds trace-on and trace-off Release binaries on WEIRDO.
3. One checksum-pinned GIST1M index is reused across all `efSearch` values.
4. Trace-on and ordinary search results match exactly on the correctness subset.
5. The maximum-`efSearch` pilot passes storage, time, and memory gates.
6. All 1,000 queries are covered exactly once for every `efSearch`.
7. Aggregated row counts reconcile with per-query summaries and deterministic sampling.
8. Margin percentages use valid-neutral denominators and report the denominator counts.
9. Rejected-at-evaluation and fully-state-neutral metrics are separately named.
10. Trace-enabled and trace-disabled Recall agree for every configuration.
11. Performance variability is measured and unstable results are not overstated.
12. A provenance-complete GIST1M summary is generated.
13. A direct SIFT1M-versus-GIST1M report is generated using consistent definitions.
14. Conclusions remain observational and make no unmeasured quantization or speedup claim.

## 16. Immediate next steps

Implement only the readiness and pilot path first:

1. Correct the margin denominator and neutral-ratio names in analysis output.
2. Recompute the corrected compact SIFT1M metrics from existing aggregated artifacts.
3. Add GIST1M preparation and manifest generation.
4. Add `gist1m_first_full.json` with `query_count=1000`.
5. Generalize the submission entry point and add preflight validation.
6. Run dataset validation and ten-query correctness on WEIRDO.
7. Run the 50-query `efSearch=400` capacity pilot.
8. Review the pilot projection and freeze the full-run configuration.

Do not submit the complete trace matrix until steps 1-8 pass.

## 17. References

- P. H. Chen et al., *FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search*, WWW 2023: <https://doi.org/10.1145/3543507.3583318>
- FINGER author PDF: <https://assets.amazon.science/74/56/d9c1550a4afab9020fb063336769/finger-fast-inference-for-graph-based-approximate-nearest-neighbor-search.pdf>
- ANN-Benchmarks dataset definitions and protocol: <https://github.com/erikbern/ann-benchmarks>
- Existing project-wide plan: `HNSW_REAL_DATA_TRACE_EXPERIMENT_IMPLEMENTATION_PLAN.md`
- Existing operational runbook: `REAL_DATA_TRACE_RUNBOOK.md`
- Completed SIFT1M summary: `sift1m-first-full-20260716-summary/ANALYSIS.md`
