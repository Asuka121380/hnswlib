# EQ-RCP Stage 1 Runbook

## Purpose

Stage 1 freezes the operational learning problem before any quantizer is trained. It converts a frozen search trace into three immutable, path-independent logical tables:

- `queries.parquet`: query-level split ownership;
- `edges.parquet`: deduplicated directed operational edges and exact edge geometry;
- `events.parquet`: one row per operational comparison, including threshold, exact label, `Y = <q-c,v-c>`, margin, and weights.

The source of truth remains the frozen trace, vector files, internal-to-label mapping, and native HNSW index. Delta vectors are reconstructed rather than duplicated.

## Local development materialization

From the repository root:

```powershell
./scripts/eq_rcp/run_stage1_development.ps1
```

The wrapper runs unit tests, builds twice into different physical directories, validates all geometry in the first build, samples the second, and requires identical path-independent semantic hashes.

The local GIST1M configuration intentionally has no native index path. Its adjacency status is therefore `PENDING_INDEX_UNAVAILABLE`; this is permitted only because `dataset_role=development`. Local edge IDs are trace-pair IDs and must not be deployed against another index.

## Formal cluster materialization

1. Copy `configs/eq_rcp/stage1_cluster_template.json` and replace every placeholder.
2. Use a fresh formal query population. Do not relabel the current development queries as final test data.
3. Freeze the native hnswlib index, trace, base/query vectors, mapping, and mapping manifest together.
4. Fill every expected SHA-256 value. The mapping manifest's `index_sha256` must equal the configured and actual native index hash.
5. Keep `dataset_role=formal` and `require_index_adjacency_validation=true`.
6. Run:

```bash
python scripts/eq_rcp/build_operational_dataset.py \
  --config configs/eq_rcp/stage1_cluster.json \
  --output-dir results/eq_rcp/stage1/formal/materialization-a

python scripts/eq_rcp/validate_operational_dataset.py \
  --dataset-dir results/eq_rcp/stage1/formal/materialization-a \
  --geometry-record-limit 0
```

7. Repeat into `materialization-b` on a different physical path and compare `dataset_semantic_sha256` plus all three table `semantic_sha256` values.

## Hard gates

Formal Stage 1 is accepted only when:

- all input content hashes match the frozen manifest;
- every trace internal ID maps to the correct base-vector label;
- every traced directed edge is present in the frozen index and has a recorded neighbor slot;
- recomputed current/candidate distances and the exact identity agree within tolerance;
- query splits are deterministic, disjoint, and query-level;
- no label-only field is classified as an online input;
- two builds have identical semantic hashes;
- `formal_materialization_status` is `PASS`.

`PENDING_INDEX_UNAVAILABLE` is never a formal pass.
