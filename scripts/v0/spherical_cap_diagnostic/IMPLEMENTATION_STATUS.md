# Implementation status

Date: 2026-08-04

## Frozen baseline

- Branch starting point: `374b51dfbb6888a5e7bc87b8313113e5060b5c93`
- Branch: `v0-spherical-cap-diagnostic`
- Existing Release build: `build-v0-shadow`
- Real pruning: OFF
- Baseline tests passed before implementation:
  - `v0_all_edge_encoder_test`
  - `v0_query_lut_test`
  - `v0_bound_evaluator_test`
  - `v0_shadow_validation_test`
- The local Windows checkout does not contain the full GIST1M dataset, HNSW
  index, or `.v0meta` sidecar, so the frozen 100-query shadow replay cannot be
  executed here. The prior manifest locates these inputs on the cluster at:
  - `/home/remote/u7808975/IndividualProject/datasets/gist1m/dataset.json`
  - `/home/remote/u7808975/IndividualProject/datasets/gist1m/indexes/gist1m_M16_efc200_seed42_gitfd11efdb86c6.bin`
  - `/home/remote/u7808975/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta`

## Phase 1

- [x] Compile-time-isolated sampled geometry exporter.
- [x] No sidecar or edge-record schema change.
- [x] No current-LB or search-control-flow decision change.
- [x] Direct sampled reconstruction of \(s\), \(x^\top r\), \(\lVert x\rVert\), and certificate error.
- [x] 80-digit Decimal closed-form evaluator.
- [x] Independent 2D span-reduction support check.
- [x] Synthetic/degenerate analyzer tests.
- [x] Geometry, tightness, coverage, per-query, and safety outputs.
- [ ] Run the sampled GIST1M experiment and record the Phase-1 Go/No-Go result.

The implementation and local validation are complete; the empirical gate is
pending a cluster run using the paths above.

## Locked later phases

Strict interval C++ cap bounds, double-sided LUTs, online cap shadow decisions,
and real pruning are not implemented. They remain gated on a Phase-1
`go_candidate` result and a separate certificate audit.
