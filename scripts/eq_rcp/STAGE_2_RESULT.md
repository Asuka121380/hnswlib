# EQ-RCP Stage 2 Result

Validated development runs: `20260810-183251-348` and `20260810-183425-765`.

- Implementation: PASS
- Synthetic unit/integration tests: PASS
- Local GIST1M development ceiling: PASS
- Path-independent repeatability: PASS
- Development diagnostic Gate: FAIL
- Formal Stage 2 Gate: `PENDING_FORMAL_STAGE2`
- Results semantic SHA-256: `a8ca7a8ff7cd5fed11ca63ca7f7752dab72fc5e01564c36d9b1ce39cd6c5bf09`

Both runs produced the same semantic hash in different physical directories. The hash covers canonical model arrays, query selections, logical metrics, per-query results, and Gate output; it excludes physical paths, NPZ container bytes, and training time.

## Development selection

- Quantizer training: 40 queries / 9,174 events
- Flexible-decoder training: 30 queries / 6,326 events
- Independent evaluation: 40 queries / 8,591 events
- Methods: 6
- Seeds: 2 (`17`, `43`)
- Code contract: 32 uint8 subcodes = 32B for every method
- Dimension: 960

This is a deterministic query-level subsample of the Stage 1 development dataset. It is not formal evidence.

## Mean held-out results across seeds

| Method | Projection MSE | Q99 `abs(X^T r)` | Q99.9 | Unique-edge reconstruction MSE | Flexible code-oracle coverage |
|---|---:|---:|---:|---:|---:|
| direct-Delta PQ | 0.067581 | 0.526204 | 0.669772 | 0.691246 | 0.422535 |
| direct-Delta OPQ | 0.051704 | 0.466911 | 0.603346 | 0.600780 | 0.517984 |
| diagonal OAE-PQ | 0.068061 | 0.525848 | 0.660136 | 0.698233 | 0.400477 |
| block OAE-PQ | 0.072094 | 0.536137 | 0.675572 | 0.703901 | 0.398033 |
| full-metric OAE ceiling | 0.079945 | 0.588384 | 0.734110 | 0.805241 | 0.203876 |
| direction-PQ + exact length reference | 0.039074 | 0.411587 | 0.532430 | 0.844230 | 0.494704 |

The direction/gain-shape reference has worse vector reconstruction MSE than OPQ but the best operational projection MSE and Q99. This is direct evidence that reconstruction MSE is not a sufficient model-selection objective in this problem.

## Pre-registered diagnostic Gate

Best MSE baseline: `direct_delta_opq`.

Best structured OAE candidate: `oae_diagonal`.

- Required Q99 reduction: at least 10%; observed `-12.6227%` (OAE is worse).
- Required paired-query CI lower bound: greater than zero; observed 95% CI `[-0.053856, -0.040342]` for baseline-Q99 minus OAE-Q99.
- Required flexible code-oracle coverage gain: at least 2 pp; observed `-11.7507 pp`.
- Required ideal coverage: at least 10%; observed `40.0477%` (only passing substantive criterion).
- Required seed stability: failed; Q99 reductions were `-11.3111%` and `-13.9718%`.

Therefore the development diagnostic Gate is a clear failure for the current global-metric OAE construction.

## Interpretation boundary

This does not authorize `NO_GO_CPP`, because:

- Stage 1 is development-only and lacks formal frozen-index adjacency validation;
- Stage 2 used 40/30/40 query subsamples rather than complete formal splits;
- the current OAE metric is one global visit covariance, without static length/density grouping;
- only the raw-event objective was run; the pre-registered smooth threshold-weight challenger remains untested;
- formal training uses more iterations, three seeds, and a fresh query population.

It does, however, reject the idea that the current global diagonal/block/full OAE metric is already an adequate replacement for OPQ. The next theory-first action should be to diagnose why the global `A=E[XX^T]` surrogate fails and test the gain-shape parameterization suggested by the strong direction+exact-length reference before starting Stage 3 or any C++ sidecar/LUT work.
