# EQ-RCP Stage 1 Result

Run: `20260810-180305-951`

- Implementation: PASS
- Unit tests: PASS
- Local development materialization: PASS
- Local repeatability: PASS
- Cluster formal materialization: PENDING

## Materialized data

- Queries: 1,000
- Operational events: 211,836
- Deduplicated directed trace edges: 203,740
- Oracle-prunable events: 185,205 (0.8742848241092166)
- Good-candidate events: 26,631
- Dataset semantic SHA-256: `2cca1c36ddba3cbde3ddc5191435a2c55decf5c0cee23015a1a81b2b17e4aebf`
- Query-table semantic SHA-256: `d455007af9fc97cd96879c778153dcb320ef728364e6e0569bd94c92f98b4ed9`
- Edge-table semantic SHA-256: `9f6ced2c6fb77db411765e5dbd5f8bd95cf333eb7200fa9eb131e1c9597a2a4c`
- Event-table semantic SHA-256: `09c121a63dce281f92a8ceec9cbd86816a175c7b98477ec3db891556fe0912c8`

Two independent builds in different physical directories produced all four identical semantic hashes. The dataset hash includes the path-free logical configuration, the three table hashes, the schema, and every frozen input file SHA-256.

## Geometry validation

The first materialization recomputed all 211,836 events from frozen float32 base/query vectors using float64 accumulation.

- Configured absolute tolerance: `1e-5`
- Maximum current-distance error: `1.6575524153239485e-6`
- Maximum candidate-distance error: `2.7125710815312232e-6`
- Maximum exact-identity error: `2.3092638912203256e-14`

## Query-level development split

| Split | Queries | Events |
|---|---:|---:|
| development_quantizer_train | 427 | 90,625 |
| development_decoder_train | 154 | 32,583 |
| development_model_selection | 156 | 32,844 |
| development_risk_calibration | 155 | 33,700 |
| development_final_test | 108 | 22,084 |

Assignment uses `sha256(seed:query_id)-uint64le-v1`, seed `20260810`; assignment SHA-256 is `18e47793a70c07897e87cff9961f8e317753386c10e7b90b1b149bb670e540a4`.

## Remaining formal gate

Local adjacency validation is `PENDING_INDEX_UNAVAILABLE`. The local machine has the mapping manifest and expected index hash but not the 3.99 GB native HNSW index itself. Consequently local `edge_id` values have scope `development_trace_directed_pair` and must not be treated as deployable neighbor-slot IDs.

Formal Stage 1 requires a fresh query population and the exact frozen native HNSW index on the cluster. Every traced edge must be found in its source node's level-0 adjacency list, `neighbor_slot` must be materialized, and `formal_materialization_status` must become `PASS`.
