# Edge Transform V1 validation tools

This directory implements Gate 0, Phase A, and the offline Phase B of
`C:\ANU\IndividualProject\8.10\EDGE_TRANSFORM_VALIDATION_AND_IMPLEMENTATION_PLAN_20260810.md`.

It deliberately contains no online pruning hook.  Online hnswlib integration
was gated on at least 10% held-out deterministic saving with zero bound
violations. The 2026-08-10 GIST1M Gate B result is `NO_GO_CPP`, so this branch
intentionally contains no sidecar or search hook.

## Commands

Run the million-case mathematical stress test:

```powershell
& .\.venv-msys\bin\python.exe `
  scripts\edge_transform_v1\run_property_tests.py `
  --trials 1000000 `
  --output results\edge_transform_v1\gate_a_property_tests.json
```

Freeze the existing Stage-B development trace:

```powershell
& .\.venv-msys\bin\python.exe `
  scripts\edge_transform_v1\freeze_operational_manifest.py `
  --records C:\ANU\IndividualProject\8.3\20260803-radius-v2-q1000\analysis-stageB-radius-q1000\candidate_metrics.parquet `
  --source-metadata C:\ANU\IndividualProject\8.3\20260803-radius-v2-q1000\stageB-radius-q1000\metadata.json `
  --repo C:\ANU\IndividualProject\Implementation\hnswlib `
  --output-dir results\edge_transform_v1\gate_0
```

The trace stores HNSW internal IDs. Extract the internal-to-external label
mapping from the exact frozen index before replay; directly indexing fvecs by
trace ID is incorrect for a parallel-built index:

```bash
python extract_hnsw_internal_labels.py \
  --index <frozen-index.bin> \
  --output internal_to_label.npy \
  --expected-count 1000000 \
  --require-permutation
```

Run the pivot audit after the raw vectors and mapping are available:

```powershell
& .\.venv-msys\bin\python.exe `
  scripts\edge_transform_v1\pivot_audit.py `
  --records <candidate_metrics.parquet> `
  --base-fvecs <gist_base.fvecs> `
  --query-fvecs <gist_query.fvecs> `
  --internal-to-label <internal_to_label.npy> `
  --split-manifest results\edge_transform_v1\gate_0\query_split_manifest.json `
  --output results\edge_transform_v1\gate_b\pivot_audit.json
```

Run the same-budget Gate-B replay after the raw vectors are available:

```powershell
& .\.venv-msys\bin\python.exe `
  scripts\edge_transform_v1\gate_b_replay.py `
  --records <candidate_metrics.parquet> `
  --base-fvecs <gist_base.fvecs> `
  --query-fvecs <gist_query.fvecs> `
  --internal-to-label <internal_to_label.npy> `
  --split-manifest results\edge_transform_v1\gate_0\query_split_manifest.json `
  --output-dir results\edge_transform_v1\gate_b `
  --m 32 --ksub 256
```

The replay trains four 32-byte representations: identity edge PQ, random/JQ
rotation, uniform Edge-OPQ, and an operationally weighted ETV1 OPQ baseline.
It reports both a shared query-mean pivot and the offline `p=c` oracle.

The tracked formal result is recorded in `GATE_B_RESULT.md`. A Chinese project
copy is also retained outside this Git worktree under the dated `8.10` reports.
