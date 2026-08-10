# EQ-RCP Stage 0 Result

**Date:** 2026-08-10  
**Branch:** `eq-rcp-staged-implementation`  
**Base commit:** `55a6d2e5a1557f181b05f74644e15a01d7b00d69`  
**Decision:** `PASS`

## Scope

Stage 0 establishes only the feature-isolated implementation contract:

- default-off CMake options and dependency validation;
- immutable base enums and the provisional 32-byte-code/48-byte-record layout;
- an EQ-RCP feature-isolation test;
- a feature-off versus skeleton-on semantic probe;
- a reproducible build-matrix and trace-comparison harness.

It does not add an OAE quantizer, sidecar, query LUT, `hnswalg.h` hook, shadow
estimator, or pruning behavior.

## Build matrix

Two valid Release builds were configured with GNU C++ 16.1.0:

| Build | `HNSWLIB_ENABLE_EQ_RCP` | Baseline trace | Result |
|---|---:|---:|---|
| feature-off | `OFF` | `ON` | Pass |
| skeleton-on | `ON` | `ON` | Pass |

The harness asserts the actual cached Boolean value after configuration so a
literal or stale cache value cannot masquerade as an off/on comparison.

Seven invalid configurations were rejected at CMake configure time:

1. shadow without the EQ-RCP root feature;
2. real pruning without the root feature;
3. fine timing without the root feature;
4. offline tools without the root feature;
5. EQ-RCP real pruning together with EQ-RCP shadow;
6. EQ-RCP real pruning together with legacy V0 real pruning;
7. EQ-RCP real pruning together with V0 ratio real pruning.

## Tests

The following passed:

- feature-off `baseline_trace_test`;
- skeleton-on `baseline_trace_test`;
- `eq_rcp_feature_isolation_test`;
- feature-off CTest: 1/1;
- skeleton-on CTest: 2/2;
- `eq_rcp_semantic_comparison_test.py`;
- existing `v0_feature_isolation_test`;
- existing `v0_query_metadata_test`;
- existing `v0_shadow_validation_test`;
- existing `v0_real_pruning_test`.

The baseline trace test reported the same maximum relative geometry error in
both builds: `0.000197553`.

## Semantic parity

The comparison used a deterministic direct-search probe plus a synthetic
baseline trace with:

- 512 base vectors;
- 16 queries;
- dimension 64;
- `K=10`;
- `efSearch=50`;
- `M=16`;
- `efConstruction=100`;
- seed 42;
- full DCO sampling.

Only three timing columns and `build_latency_ns` were excluded. Search results,
visited/control-flow records, thresholds, queue states, distance counts,
geometry, and recall remained part of the comparison.

| Artifact | SHA-256 in both builds | Result |
|---|---|---|
| Semantic search probe | `4a82c48ff84cae06e4f9ecf4290684ca68ec1ac954fef5f8ee74b1e52afacbaa` | Equal |
| Normalized metadata | `5be645d0d2a5214803adacd1642cd0a01461cb95ec8b1fa644cb9c3c9a3575f4` | Equal |
| Query counters without timing | `31ec518410ee052b6a2b51e31a20af081e5b0ddb92a47ac34fc20e8a250304ab` | Equal |
| Full DCO trace | `1256a3bd9b601c38ceedfdd201698444977ec0754af34f4db45723f337a43db3` | Equal |
| Edge geometry sample | `1eb0e87c7cdc7719edaadd42502d5e29ec084d76e2fedd5d7d0fc194d50a24f5` | Equal |

Both builds reported mean recall `1.0` and mean exact-distance count `396`.

## Gate decision

| Condition | Result |
|---|---|
| New features default off | Pass |
| Invalid feature combinations fail closed | Pass |
| Feature-off CTest excludes the EQ-RCP test | Pass |
| Feature-off and skeleton-on search results match | Pass |
| Visited/control-flow trace matches | Pass |
| Threshold and distance counters match | Pass |
| Existing V0 regression tests pass | Pass |
| No search hook or pruning behavior added | Pass |

Stage 0 therefore passes. Stage 1 may construct the unified operational
dataset, but this result does not claim that OAE, progressive coding, or
pruning is effective.

## Reproduction

```powershell
& scripts\eq_rcp\run_stage0_validation.ps1 -Parallel 6
```

Ignored raw outputs are written under `results/eq_rcp/stage0/`; build trees are
under `build-eq-rcp-stage0/`.
