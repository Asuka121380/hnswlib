# Safe-bound diagnostic implementation status

## Implemented

- Schema-v2 identity, search context, bound reconstruction and radius-component
  fields.
- Observe-only CSV output with reserved nullable cap, blockwise and `LB*`
  columns.
- Current radius decomposition into direction error, stored numeric padding,
  operational float32 L2 padding, and upward-rounding closure.
- Chunked validation and analysis with Parquet, CSV, JSON, Markdown and PNG
  outputs.
- Positive and deliberately broken synthetic Python regression cases.
- C++ checks for component closure, directed rounding and unchanged shadow
  search results.
- Stage-A-first Slurm workflow and reproducibility manifest.

## Verified locally

- `v0_shadow_validation_test`
- `v0_real_pruning_test`
- `v0_query_metadata_test`
- `v0_feature_isolation_test`
- existing schema-v1 shadow analysis regression test
- schema-v2 radius analysis regression test
- Python syntax and Slurm shell syntax

## Intentionally unchanged

- The current V0 lower-bound formula and decision threshold.
- Default HNSW search behavior.
- Real-pruning enablement and control flow.
- Sidecar binary format and bytes per edge.

## Gated next work

Run the 100-query cluster Stage A and inspect the component report. Formalizing
the current-code uncertainty set and implementing certified `LB*`/spherical-cap
candidates begins only after Stage A passes. Blockwise PQ and progressive
residual remain conditional on the theoretical coverage upper bound exceeding
the measured break-even requirement.
