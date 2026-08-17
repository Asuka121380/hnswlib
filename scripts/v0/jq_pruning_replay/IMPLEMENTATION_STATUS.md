# JQ theoretical-pruning replay: implementation status

## Scope and provenance

- Repository branch: `jq-theoretical-pruning-replay`
- Base branch: `v0-edge-quantisation`
- Base commit: `374b51dfbb6888a5e7bc87b8313113e5060b5c93`
- Official JHQ reference commit: `1636e197a36871db4a79d66d22f9f9dd0fa51e31`
- Evaluation target: theoretical pruning rate under the frozen V0 shadow trace.
- This branch deliberately does not modify online HNSW search, add eager/lazy LUTs,
  or claim an end-to-end latency result.

## Implemented components

- Deterministic single-level primary JQ training and encoding for `M=32` and
  `M=120`.
- A behavioral port of the official analytical Gaussian initializer, including
  subspace statistics, robust variance, norm quantiles, fixed directions,
  Gram-Schmidt orthogonalization, radii, dimension adjustment, and noise.
- Conservative direction-error bounds that remain valid under stored float32
  rotations.
- V0-compatible upward cell rounding, upward accumulation, float32 operational
  padding, strict `lower_bound > threshold` pruning, and old 14-column shadow
  trace support.
- Memory-mapped `.fvecs` loading and explicit internal-ID-to-label mapping.
- Frozen input contracts, atomic artifacts, refusal to overwrite complete or
  partial output directories, per-run manifests, and fixed-seed aggregation.
- Unit, parity, admissibility, mapping, and end-to-end fixture tests.

## Verification

The complete test suite was run with bytecode generation disabled:

```text
python -B -m unittest discover -s scripts/v0/jq_pruning_replay -p test_*.py
Ran 12 tests
OK
```

All six full frozen replays processed 211,992 rows, of which 211,836 were valid.
Every run reported:

- 203,893 unique directed edges used to fit the low-cost diagnostic quantizer;
- 153 zero-length unique edges;
- zero invalid-bound events;
- zero lower-bound violations;
- sampled exact-distance parity maximum absolute error of
  `9.5367431640625e-07`;
- exact frozen PQ reference count of 20 would-prune events.

The large inputs were separately SHA-256 verified against `dataset.json`:

- base vectors: `73418110328F5AA522D9F6B0CD9115A6C515DC44E3C48420E506DDEDDBDBDBC0`
- query vectors: `0D1D620049DE12DA455ED7201E97CBAB372C4D54D0E6DEDBC8C503F62C911299`

## Full replay results

| Configuration | Seed | Would prune | Valid events | Theoretical prune rate |
|---|---:|---:|---:|---:|
| M32, 32-byte iso-budget | 1234 | 0 | 211,836 | 0 |
| M32, 32-byte iso-budget | 17 | 0 | 211,836 | 0 |
| M32, 32-byte iso-budget | 42 | 0 | 211,836 | 0 |
| M120, 120-byte diagnostic | 1234 | 47 | 211,836 | 0.000221870 |
| M120, 120-byte diagnostic | 17 | 41 | 211,836 | 0.000193546 |
| M120, 120-byte diagnostic | 42 | 52 | 211,836 | 0.000245473 |

The M120 median theoretical pruning rate is `0.000221870`, or approximately
`0.0222%`. M120 is an upper-bound diagnostic and is not a fair memory-budget
comparison with the 32-byte PQ baseline.

Under the implementation plan's continuation criterion, neither M32 nor M120
reaches the `0.1%` theoretical-pruning threshold. This frozen-trace diagnostic
therefore provides no positive evidence for proceeding to an online HNSW or
lazy-LUT integration solely on the basis of JQ pruning rate.

## Known limitations and deviations

1. The local workspace does not contain the original HNSW index or a full-graph
   edge sidecar. The quantizer is therefore fitted on all nonzero unique directed
   edges present in the frozen trace, not on every edge in the full graph. This
   choice is recorded in each run manifest.
2. The implementation follows the official initializer algorithmically, but
   NumPy's RNG and QR/LAPACK behavior are not bitwise identical to Faiss and the
   C++ standard library. The repository consequently contains a behavioral
   reference rather than a binary official-codebook fixture. See
   `OFFICIAL_REFERENCE.md`.
3. Only the single-level primary JQ path is implemented. JHQ residual hierarchy,
   online graph traversal, eager/lazy LUT construction, and latency measurement
   are intentionally out of scope.
4. These results are a low-cost feasibility diagnostic. The trace-only fitting
   source and non-bitwise initializer parity mean they should not be presented as
   a definitive reproduction of official JQ on the complete graph.
