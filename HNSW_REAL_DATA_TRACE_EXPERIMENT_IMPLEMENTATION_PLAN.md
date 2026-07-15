# HNSW Real-Dataset Trace Experiment Implementation Plan

## 1. Purpose

This plan describes the work required to move the existing baseline-trace pipeline from a local synthetic correctness test to reproducible, server-scale experiments on real approximate-nearest-neighbour datasets.

The experiment is intended to characterize the distribution and search relevance of HNSW edge geometry, especially the feasibility of replacing some exact query-to-neighbour distance-comparison operations (DCOs) with an edge-quantized approximation or lower-cost lookup.

The implementation must preserve baseline HNSW search semantics. Trace runs are observational experiments and must not modify candidate ordering, result ordering, visited-set behaviour, termination conditions, or returned neighbours.

## 2. Current baseline

The repository already provides:

- optional trace instrumentation guarded by `HNSWLIB_ENABLE_BASELINE_TRACE`;
- level-0 per-query and per-DCO counters;
- threshold, margin, state-neutral, final-top-k, and edge-geometry fields;
- deterministic DCO sampling;
- CSV trace writers;
- a synthetic high-dimensional C++ runner;
- trace-on versus ordinary-search correctness checks;
- Python validation, Parquet conversion, statistical analysis, PCA, and plots;
- a validated local Windows/MSYS2 toy pipeline.

The current runner generates all vectors in memory, rebuilds an index for every run, computes ground truth by brute force, writes one CSV file per table, and does not support Slurm query sharding. It is therefore a correctness gate, not a full real-data experiment driver.

## 3. Scope

### 3.1 In scope

1. Real dataset loading for base vectors, queries, and supplied ground truth.
2. Reproducible index construction, persistence, loading, and validation.
3. Separate correctness, trace-collection, and performance modes.
4. Query-range and Slurm-array sharding.
5. Scalable, restartable trace output.
6. Linux/Slurm build, index, trace, analysis, and aggregation workflows.
7. Experiment configuration and provenance capture.
8. Capacity estimation and staged execution gates.
9. Exact-distance and future edge-lookup/LUT microbenchmarks.
10. A final report evaluating the edge-quantization feasibility conditions.

### 3.2 Out of scope

This phase does not implement the production edge codebook, quantized edge representation, approximate routing rule, or modified HNSW search policy. Those changes begin only after the trace evidence supports feasibility.

GPU execution is also out of scope unless a later implementation has a concrete GPU workload. The current trace experiment is CPU- and storage-oriented.

## 4. Target environment

The experiments run on the ANU School of Computing cluster, primarily on the WEIRDO node through Slurm:

```text
Account:    db4ai
Partition:  db4ai
QOS:        db4ai
Node:       weirdo
CPU:        AMD EPYC 9654, 96 physical cores, 192 logical CPUs
Memory:     approximately 566 GiB node memory
```

Verified native tools include GCC/G++ 13.3, CMake 3.28.3, GNU Make 4.3, Ninja 1.11.1, Python 3.12.3, and Singularity 3.5.3.

The persistent Python environment is:

```bash
source ~/IndividualProject/envs/baseline-trace-py312-v2/bin/activate
```

Heavy compilation, index construction, preprocessing, trace collection, and analysis must run through Slurm. The login node is limited to lightweight file operations, Git operations, submission, monitoring, and log inspection.

## 5. Proposed repository layout

```text
examples/cpp/
  baseline_trace_runner.cpp          existing synthetic correctness runner
  real_data_trace_runner.cpp         real-data correctness and trace driver
  distance_microbenchmark.cpp        exact-distance cost benchmark

hnswlib/
  baseline_trace.h
  baseline_trace_writer.h

scripts/baseline_trace/
  analyze_trace.py
  aggregate_shards.py
  validate_dataset.py
  estimate_trace_capacity.py
  prepare_dataset.sh
  configure_build.sh
  submit_experiment.sh
  slurm/
    environment_probe.slurm
    build_trace.slurm
    build_index.slurm
    correctness_smoke.slurm
    collect_trace_array.slurm
    analyze_trace.slurm
    performance_baseline.slurm
    distance_microbenchmark.slurm

configs/baseline_trace/
  datasets/
    sift1m.json
  experiments/
    sift1m_smoke.json
    sift1m_trace.json
    sift1m_performance.json

results/baseline_trace/<run_id>/
  config.resolved.json
  metadata/
  index/
  raw/
    query_stats/
    dco_trace/
    edge_directions/
  analysis/
  logs/
  status/
```

Generated indexes, traces, datasets, and results remain outside Git.

## 6. Phase 1: real dataset support

### 6.1 Initial dataset

Use SIFT1M as the first real-data integration target because it has established ANN benchmark files and supplied ground truth. It provides a controlled bridge between the synthetic test and higher-dimensional datasets.

SIFT1M is not sufficient as the only feasibility dataset because its dimension is relatively low. After the pipeline is validated, add at least one higher-dimensional real dataset representative of the proposed edge-quantization use case.

### 6.2 Dataset manifest

Each dataset receives a versioned JSON manifest containing:

```json
{
  "dataset": "sift1m",
  "distance_kind": "squared_l2_float32",
  "base_path": "...",
  "query_path": "...",
  "ground_truth_path": "...",
  "base_format": "fvecs",
  "query_format": "fvecs",
  "ground_truth_format": "ivecs",
  "dimension": 128,
  "n_base": 1000000,
  "n_query": 10000,
  "ground_truth_k": 100,
  "checksums": {}
}
```

Paths may be resolved from an environment-specific data root so configuration files do not hardcode Windows paths or a single user's temporary directory.

### 6.3 Vector readers

Implement strict readers for `fvecs`, `bvecs`, and `ivecs` as required. Readers must:

- validate the dimension prefix of every record;
- detect truncated and malformed input;
- reject inconsistent dimensions;
- use overflow-checked size calculations;
- support `start` and `count` ranges where practical;
- convert byte vectors to float32 explicitly when required;
- preserve dataset label ordering;
- report the number of records loaded.

Do not silently reinterpret inner-product data as L2 data.

### 6.4 Ground truth

Real-data runs should read supplied ground truth rather than recompute it by brute force. Validate that:

- the ground-truth query count covers the requested query range;
- the ground-truth K is at least the experiment K;
- all labels are within the base-vector range;
- query IDs remain global across shards.

Brute-force ground truth remains available only for small correctness fixtures.

## 7. Phase 2: index lifecycle

### 7.1 Runner interface

The real-data runner should support at least:

```text
--dataset-config <path>
--mode correctness|trace|performance
--index-path <path>
--build-index
--load-index
--save-index
--query-start <global index>
--query-count <count>
--k <value>
--ef-search <value>
--M <value>
--ef-construction <value>
--seed <value>
--threads <value>
--dco-sample-modulus <value>
--dco-sample-remainder <value>
--collect-geometry true|false
--collect-distance-timing true|false
--output-dir <path>
--run-id <id>
```

Unknown arguments and incompatible combinations must fail with a non-zero exit code and a clear message.

### 7.2 Build once, query many times

Index construction must be separated from trace collection. A single index should be reused across multiple `efSearch` values and trace shards when the dataset, `M`, `efConstruction`, seed, distance, and source commit are identical.

Store an index manifest next to the binary index containing:

- dataset name and checksums;
- dimension and base count;
- distance type;
- `M`, `efConstruction`, and build seed;
- build threads and elapsed time;
- compiler, build type, and flags;
- Git commit and dirty status;
- index byte size and checksum.

The trace runner must validate the manifest before loading the index.

### 7.3 Memory handling

Avoid holding unnecessary duplicate copies of the base dataset. Geometry collection requires access to original vectors, but performance mode may not. Document peak-memory expectations for each mode and fail early when file sizes or vector counts disagree with the manifest.

## 8. Phase 3: execution modes

### 8.1 Correctness mode

Purpose: prove that tracing does not alter search behaviour on real data.

- use a small, deterministic query subset;
- run ordinary and traced searches against the same loaded index;
- compare queue size, label, distance, and ordering;
- verify trace accounting invariants;
- verify sampled geometry identities;
- compute recall from supplied ground truth;
- write a small full-DCO trace for inspection.

No full trace run may proceed until this mode passes.

### 8.2 Trace mode

Purpose: collect the distributions needed for edge-quantization feasibility analysis.

- record every query summary;
- use deterministic DCO sampling when full DCO output is too large;
- optionally collect sampled edge geometry;
- disable per-DCO distance timing unless explicitly needed;
- write one independent output shard per Slurm task;
- treat trace latency as instrumentation cost, not baseline performance.

### 8.3 Performance mode

Purpose: measure trustworthy baseline search performance.

- compile trace support out or use a trace-disabled target;
- perform warm-up queries;
- run repeated measurement rounds;
- write aggregate timing rather than DCO logs;
- fix thread count and CPU binding;
- report QPS, mean latency, P50, P95, and variation;
- run on the same node and index as compared methods.

## 9. Phase 4: scalable trace output

### 9.1 Sharding

Use global query ranges and Slurm array tasks. Each task writes only to its own directory:

```text
raw/query_stats/part-00017.csv
raw/dco_trace/part-00017.csv
raw/edge_directions/part-00017.csv
status/part-00017.complete.json
```

The global query ID must not be renumbered within a shard. DCO sampling must be deterministic from the run seed, global query ID, and DCO index.

### 9.2 Writer improvements

The first server version may retain CSV if it includes:

- large buffered writes;
- no shared output file between tasks or threads;
- explicit file-open and write-error checking;
- periodic safe flushes at query boundaries;
- a schema version in metadata;
- completion markers written only after files close successfully.

For larger runs, add a versioned binary format or incremental conversion to compressed Parquet. The analysis code must not require loading the entire DCO table into memory.

### 9.3 Resume and overwrite protection

- A run ID maps to one resolved configuration.
- Existing completed shards are skipped only if their manifest matches.
- Partial shards are rerun into new temporary paths and atomically renamed on success.
- No command silently overwrites an existing completed run.
- Aggregation verifies that query ranges are complete and non-overlapping.

## 10. Phase 5: Slurm workflow

### 10.1 Resource rules

WEIRDO jobs use:

```bash
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
```

Every job explicitly requests CPUs, memory, and time. Application thread counts must be derived from `SLURM_CPUS_PER_TASK`.

### 10.2 Job chain

```text
environment/build validation
            |
            v
       index build
            |
            v
   real-data correctness smoke
            |
            v
   trace-capacity pilot shard
            |
            v
      trace array job
            |
            v
 aggregation and validation
            |
            v
 analysis and final report
```

Use `sbatch --parsable` and dependency flags so downstream jobs begin only after successful prerequisites. Record all job IDs in the run metadata.

### 10.3 Slurm script requirements

Every script must:

- use `set -euo pipefail`;
- create output and log directories before writing;
- print hostname, timestamp, job ID, commit, and resolved configuration;
- activate the persistent Python environment where needed;
- set a job-local Matplotlib cache;
- set explicit thread counts;
- check executable and input paths;
- propagate non-zero exit codes;
- leave `.out`, `.err`, and machine-readable status metadata.

## 11. Phase 6: capacity planning

Before a full trace, execute a representative pilot shard and estimate:

```text
mean DCOs per query
sampled DCO rows per query
bytes per query-summary row
bytes per sampled DCO row
bytes per edge-direction row
trace wall time per query
peak RSS
projected total output size
projected total CPU time
```

The capacity estimator must refuse or warn about a configuration whose projected output exceeds an explicit run budget. Keep the approximately 100 GB personal quota in mind and avoid duplicating datasets, indexes, or derived traces.

Suggested staged gates:

1. 10-query full-trace correctness run.
2. 100-query full-trace pilot.
3. 1,000-query sampled trace.
4. Full query set only after storage and time projections are accepted.

## 12. Phase 7: experiment matrix

### 12.1 Initial matrix

For the first real dataset, keep index construction fixed and vary only search effort:

```text
M                = 16
efConstruction   = 200
k                = 10
efSearch         = 50, 100, 200, 400
seed             = fixed documented value
```

Run correctness, trace, and performance modes separately. Do not combine instrumentation timing with baseline performance claims.

### 12.2 Later axes

After the first matrix passes, consider:

- multiple datasets and dimensions;
- multiple `M` values;
- multiple recall targets;
- multiple seeds where index randomness materially changes results;
- geometry sampling rates;
- edge strata by layer, degree, length, access frequency, and search progress.

Change one major axis at a time unless a factorial design is explicitly planned.

## 13. Phase 8: edge sampling and analysis

The current first-N-edge sampler is not representative. Replace it with deterministic stratified or reservoir sampling over:

- graph layer;
- node degree;
- edge-length quantile;
- observed query access frequency;
- early and late search stages;
- expanded and non-expanded candidates;
- final-top-k relevance class.

Required analysis includes:

- distributions of `N_dist`, exact-call sources, and duplicate scans;
- state-neutral and candidate-importance ratios;
- absolute and relative margin distributions by class and progress;
- threshold stabilization behaviour;
- triangle-bound opportunity rate;
- edge length, cosine, and margin relationships;
- edge-direction PCA or related compressibility measures;
- recall by `efSearch`;
- sensitivity to DCO sampling rate;
- dataset-to-dataset variation.

Analysis must report counts and denominators, not only percentages.

## 14. Phase 9: cost microbenchmarks

The current break-even plot uses provisional timing constants. Replace them with measurements on WEIRDO.

Implement isolated benchmarks for:

- exact float32 squared-L2 at relevant dimensions;
- future quantized-edge lookup/decode cost;
- LUT construction cost per query;
- memory-access effects for warm and cold data;
- batch size and thread-count sensitivity.

Use release builds, warm-up, fixed CPU binding, repeated rounds, and robust statistics. Record compiler flags and prevent dead-code elimination. The trace feasibility report must not use assumed lookup or LUT costs as measured evidence.

## 15. Provenance

Each resolved run configuration must record:

```text
run ID
dataset name, version, paths, and checksums
index manifest and checksum
Git commit and dirty state
trace schema version
compiler and build flags
hostname and CPU model
Slurm job and array IDs
CPU, memory, time, and binding request
thread count
distance type and dimension
k, M, efConstruction, efSearch, and seed
query range and shard definition
DCO and geometry sampling rules
Python and dependency versions
start/end timestamps
input/output paths
```

The resolved configuration is immutable after the run starts.

## 16. Validation gates

### Gate A: dataset integrity

- expected dimensions and counts match;
- files are not truncated;
- checksums are recorded;
- ground-truth labels are valid.

### Gate B: index integrity

- index manifest matches the dataset and parameters;
- save/load round trip preserves search results;
- index checksum and size are recorded.

### Gate C: search neutrality

- ordinary and traced results match exactly for the correctness subset;
- trace accounting identities pass;
- geometry reconstruction error is within tolerance.

### Gate D: shard integrity

- query ranges are complete and non-overlapping;
- all expected completion markers exist;
- row counts reconcile with query summaries and sampling rules;
- aggregation is deterministic.

### Gate E: feasibility evidence

- sufficient replaceable DCO volume exists;
- safe/neutral cases have useful margin distributions;
- edge representation shows useful compressibility or predictive structure;
- measured lookup and setup costs admit a realistic break-even point;
- recall and latency targets can plausibly be preserved.

Failure at an earlier gate blocks later claims.

## 17. Testing strategy

### Unit tests

- vector-format parsing and malformed-file rejection;
- query-range boundary handling;
- ground-truth parsing;
- index-manifest compatibility checks;
- deterministic shard and DCO sampling;
- writer escaping, missing values, and schema headers;
- completion-marker semantics.

### Integration tests

- small real-format fixture built from known vectors;
- index build/save/load/search round trip;
- ordinary versus trace equivalence;
- two-shard output aggregation equals one-shard output;
- interrupted shard does not appear complete;
- analysis works with sampled and full DCO traces.

### Cluster tests

- trace-enabled and trace-disabled Release builds on WEIRDO;
- 10-query correctness job;
- two-task Slurm array smoke test;
- pilot capacity estimate;
- analysis job reading shard outputs.

## 18. Implementation milestones

### Milestone 1: real-data correctness

- dataset manifest and readers;
- supplied ground-truth reader;
- real-data runner;
- index save/load and manifest;
- SIFT1M small-subset correctness pass.

### Milestone 2: cluster workflow

- Linux configure/build workflow;
- Slurm build and correctness jobs;
- persistent environment activation;
- complete provenance capture.

### Milestone 3: scalable collection

- query sharding;
- independent buffered output shards;
- completion markers and resume logic;
- aggregation and integrity validation;
- capacity estimator.

### Milestone 4: first complete real-data run

- accepted pilot storage/time estimate;
- SIFT1M trace matrix;
- trace-disabled performance baseline;
- automated analysis and report.

### Milestone 5: feasibility evidence

- higher-dimensional dataset;
- representative edge sampling;
- exact-distance and edge-cost microbenchmarks;
- cross-dataset Conditions A-E report;
- go/no-go decision for edge-codebook implementation.

## 19. Definition of done

The real-data trace experiment implementation is complete when:

1. A clean checkout builds trace-enabled and trace-disabled Release binaries on WEIRDO.
2. Dataset and index manifests validate all inputs.
3. A real-data correctness job proves search neutrality.
4. Slurm array tasks produce restartable, non-overlapping trace shards.
5. Aggregation validates counts and produces bounded-memory analysis artifacts.
6. A trace-disabled performance run provides reliable baseline timing.
7. Measured microbenchmarks replace provisional cost assumptions.
8. The full experiment configuration and provenance are reproducible from the repository.
9. At least one standard and one higher-dimensional real dataset are analyzed.
10. The final report evaluates edge-quantization feasibility using explicit evidence and denominators.

## 20. Immediate next step

Implement Milestone 1 only: add the dataset manifest, strict SIFT vector readers, supplied-ground-truth loading, index persistence, and a small real-data correctness mode. Do not begin the full Slurm trace matrix until the real-data correctness gate and a pilot capacity estimate both pass.
