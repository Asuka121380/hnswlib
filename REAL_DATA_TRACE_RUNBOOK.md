# First Complete Real-Dataset Trace Experiment Runbook

This runbook is the operational entry point for the first complete SIFT1M trace experiment described in `HNSW_REAL_DATA_TRACE_EXPERIMENT_IMPLEMENTATION_PLAN.md`.

## Preconditions

- Run commands from the remote repository root on `cluster1-login-1`.
- Do not compile, extract datasets, build indexes, or run experiments on the login node.
- The persistent Python environment must exist at:

  ```text
  ~/IndividualProject/envs/baseline-trace-py312-v2
  ```

- The repository should be under:

  ```text
  ~/IndividualProject/src/hnswlib
  ```

## 1. Prepare SIFT1M

Submit dataset download, extraction, structural validation, and checksum generation to WEIRDO:

```bash
cd ~/IndividualProject/src/hnswlib
mkdir -p ~/IndividualProject/logs

PREP_JOB=$(sbatch --parsable \
  --output="$HOME/IndividualProject/logs/sift1m_prepare_%j.out" \
  --error="$HOME/IndividualProject/logs/sift1m_prepare_%j.err" \
  --export="ALL,REPO_ROOT=$PWD,DATASET_DIR=$HOME/IndividualProject/datasets/sift1m" \
  scripts/baseline_trace/slurm/prepare_sift1m.slurm)

echo "$PREP_JOB"
```

Monitor and inspect it:

```bash
squeue -j "$PREP_JOB"
sacct -j "$PREP_JOB" \
  --format=JobID,JobName,State,Elapsed,ReqCPUS,ReqMem,MaxRSS,ExitCode,NodeList
cat "$HOME/IndividualProject/logs/sift1m_prepare_${PREP_JOB}.out"
cat "$HOME/IndividualProject/logs/sift1m_prepare_${PREP_JOB}.err"
```

Do not continue unless the job completed with exit code `0:0` and printed `prepare_sift1m_ok`.

## 2. Submit the complete experiment

```bash
cd ~/IndividualProject/src/hnswlib

bash scripts/baseline_trace/submit_first_full_experiment.sh \
  --dataset-config "$HOME/IndividualProject/datasets/sift1m/dataset.json"
```

The command creates a unique run directory and submits this dependency chain:

```text
trace-on/off Release build ----+
                               +--> index build/reuse
dataset structure/checksum ----+          |
                                          v
                                 correctness smoke
                                          |
                                          v
                                 maximum-ef pilot
                                          |
                                          v
                                  capacity budget gate
                                          |
                  +-----------------------+-----------------------+
                  v                       v                       v
           trace arrays by ef      trace-off performance   remaining ef values
                  |
                  v
          aggregate and analyze
```

The initial matrix uses `efSearch = 50, 100, 200, 400`, `M = 16`, `efConstruction = 200`, `k = 10`, and seed 42. It traces all 10,000 SIFT1M queries in 20 shards per `efSearch`, with deterministic DCO sampling modulus 10. The maximum-`efSearch` pilot blocks the full arrays if projected trace output exceeds 60 GB.

## 3. Monitor

The submission command prints the run root. Use:

```bash
squeue -u "$USER"
cat <run-root>/submitted_jobs.txt
```

After each job completes, inspect its `.out`, `.err`, and `sacct` record. A downstream job in `DependencyNeverSatisfied` indicates an earlier gate failed; inspect the earliest failed job rather than resubmitting the whole matrix blindly.

## 4. Expected results

For each `efSearch`, the run produces:

```text
<run-root>/ef<value>/
  capacity.json                 present for the maximum efSearch pilot
  raw/shards/part-*/
    metadata.json
    query_stats.csv
    dco_trace.csv
    edge_directions.csv
    complete.json
  aggregated/
    metadata.json
    query_stats.csv
    query_stats.parquet
    dco_trace.parquet
    edge_directions.parquet
  analysis/
    metrics.json
    report.md
    plots/
```

Trace-disabled performance results are under:

```text
<run-root>/performance/ef<value>/performance.csv
```

After every per-`efSearch` analysis and performance job succeeds, the final summary job writes:

```text
<run-root>/summary/
  summary.csv
  summary.json
  report.md
  plots/
```

For a completed run created before the summary job was added, submit it independently:

```bash
cd ~/IndividualProject/src/hnswlib
RUN_ROOT="$HOME/IndividualProject/results/baseline_trace/<run-id>"
PYTHON_ENV="$HOME/IndividualProject/envs/baseline-trace-py312-v2"

SUMMARY_JOB=$(sbatch --parsable \
  --output="$RUN_ROOT/logs/summary_%j.out" \
  --error="$RUN_ROOT/logs/summary_%j.err" \
  --export="ALL,REPO_ROOT=$PWD,RUN_ROOT=$RUN_ROOT,PYTHON_ENV=$PYTHON_ENV" \
  scripts/baseline_trace/slurm/summarize_experiment.slurm)

echo "SUMMARY_JOB=$SUMMARY_JOB"
```

This job reads only the small per-configuration `metrics.json` and `performance.csv` files. It does not
rescan or modify raw trace shards and aggregated Parquet files.

The first complete experiment is operationally successful only when all build, dataset, index, correctness, capacity, trace-array, aggregation, analysis, and performance jobs complete with exit code `0:0`.

## 5. Safety and reproducibility

- The submission tool refuses to overwrite an existing run directory.
- The runner refuses to overwrite completed output directories.
- Existing indexes are reused only when both index and manifest exist; the runner validates manifest parameters before querying.
- Array tasks write temporary shard directories and rename them only after completion.
- Full experiment work remains on compute nodes and uses `--hint=nomultithread`.
- Keep all logs and the resolved configuration stored under the run root.

## 6. Prepare and submit GIST1M

GIST1M is the high-dimensional follow-up experiment. It uses the same native `fvecs`/`ivecs`
formats and squared-L2 runner as SIFT1M, but contains 1,000,000 base vectors at 960 dimensions and
1,000 standard queries.

Submit download, extraction, structural validation, label validation, and checksum generation to
WEIRDO:

```bash
cd ~/IndividualProject/src/hnswlib

GIST_PREP_JOB=$(sbatch --parsable \
  --output="$HOME/IndividualProject/logs/gist1m_prepare_%j.out" \
  --error="$HOME/IndividualProject/logs/gist1m_prepare_%j.err" \
  --export="ALL,REPO_ROOT=$PWD,DATASET_DIR=$HOME/IndividualProject/datasets/gist1m" \
  scripts/baseline_trace/slurm/prepare_gist1m.slurm)

echo "$GIST_PREP_JOB"
```

Do not continue unless the job exits with `0:0` and prints `prepare_gist1m_ok`. Revalidate the
generated manifest independently:

```bash
source ~/IndividualProject/envs/baseline-trace-py312-v2/bin/activate
python scripts/baseline_trace/validate_dataset.py \
  --dataset-config "$HOME/IndividualProject/datasets/gist1m/dataset.json"
```

Before formal submission, require a clean, pushed repository state. The submission tool rejects a
dirty worktree.

```bash
git status --short
git rev-parse HEAD

bash scripts/baseline_trace/submit_first_full_experiment.sh \
  --dataset-config "$HOME/IndividualProject/datasets/gist1m/dataset.json" \
  --experiment-config configs/baseline_trace/experiments/gist1m_first_full.json
```

The initial GIST1M configuration uses all 1,000 independent queries, 20 shards, and a 50-query
maximum-`efSearch` pilot. Do not repeat queries to emulate the 10,000-query SIFT1M workload. Accept
the complete trace matrix only after the capacity job confirms that projected storage is within both
the configured budget and the actual remaining quota.

After both dataset summaries have metric schema version 2, generate the cross-dataset comparison:

```bash
SIFT_RUN_ROOT="$HOME/IndividualProject/results/baseline_trace/sift1m-first-full-20260716"
GIST_RUN_ROOT="$HOME/IndividualProject/results/baseline_trace/<gist-run-id>"
OUTPUT_DIR="$HOME/IndividualProject/results/baseline_trace/sift1m-gist1m-comparison"

sbatch \
  --export="ALL,REPO_ROOT=$PWD,SIFT_RUN_ROOT=$SIFT_RUN_ROOT,GIST_RUN_ROOT=$GIST_RUN_ROOT,OUTPUT_DIR=$OUTPUT_DIR,PYTHON_ENV=$HOME/IndividualProject/envs/baseline-trace-py312-v2" \
  scripts/baseline_trace/slurm/compare_datasets.slurm
```

Legacy SIFT1M metrics must be regenerated with the corrected valid-neutral margin denominator before
the comparison script will accept them.
