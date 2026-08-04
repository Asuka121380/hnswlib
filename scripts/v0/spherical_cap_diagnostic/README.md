# V0 spherical-cap diagnostic (Phase 1)

This directory implements the value-validation stage from
`GIST1M_V0_SPHERICAL_CAP_IMPLEMENTATION_PLAN_20260804.md`.

The implementation is deliberately split in two:

1. `v0_search_runner`, compiled with
   `HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC=ON`, samples bound attempts and
   exports raw edge geometry to `cap_diagnostic_input.csv`.
2. `analyze_spherical_cap.py` evaluates the spherical-cap formula with
   80-digit `Decimal` arithmetic, checks the support with a separate 2D span
   reduction, and produces the Phase-1 coverage/safety report.

The diagnostic does not alter the current LB, HNSW control flow, sidecar
format, or real-pruning decisions. Real pruning must remain OFF.

## Configure and build

```bash
cmake -S . -B build-v0-cap-diagnostic -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON \
  -DHNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC=ON \
  -DHNSWLIB_ENABLE_V0_REAL_PRUNING=OFF
cmake --build build-v0-cap-diagnostic --target v0_search_runner
```

Run `v0_search_runner` in `shadow` mode with the usual dataset/index/sidecar
arguments and a deterministic sample modulus/remainder. The new raw input is
written beside the existing schema-v2 outputs.

## Analyze

```bash
python scripts/v0/spherical_cap_diagnostic/analyze_spherical_cap.py \
  --input RUN_DIR/cap_diagnostic_input.csv \
  --output-dir RUN_DIR/cap_phase1_analysis
```

The decision thresholds are explicit CLI parameters. Defaults are:

- minimum extra cap coverage: `0.001`;
- maximum share of extra prunes from one query: `0.5`;
- minimum proper-cap fraction: `0.1`;
- minimum boundary-branch fraction: `0.1`.

`go_candidate` means that the sampled geometry/value gates passed; it is
permission to audit and plan Phase 2. It is not a safety certificate, does
not include the Phase-2 break-even benchmark, and is not permission to enable
real pruning.

## Tests

```bash
python scripts/v0/spherical_cap_diagnostic/test_analyze_spherical_cap.py
```

See `SCHEMA.md` for field definitions and `IMPLEMENTATION_STATUS.md` for the
current gate status.
