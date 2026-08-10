# Edge Transform V1 — Gate B Result

Date: 2026-08-10  
Branch: `edge-transform-v1`  
Decision: `NO_GO_CPP`

## Conclusion

The tested representation—single global query-mean pivot, 32-byte
edge-direction PQ, orthogonal transform, and deterministic block
reconstruction-ball certificates—does not meet the minimum representation
gate for a C++ sidecar or online shadow implementation.

This is not a safety failure: every held-out replay reported zero deterministic
lower-bound violations. The ideal float64 certificate remains too loose to
convert the strong raw estimator signal into useful deterministic pruning.
Under the pre-registered Gate B rules, ideal saving below 10% is a stop
condition, so this branch intentionally does not implement metadata Gate C,
an ETV1 sidecar, query context, or an HNSW search hook.

## Input integrity

- Operational records: 211,836.
- Records SHA-256:
  `d17b5df324de81df44e6ada00a218af3ee4d872f656d0df25804c3a77b65d1a5`.
- Internal-to-label mapping: 1,000,000 entries and a complete permutation of
  `[0, 1000000)`.
- Mapping SHA-256:
  `109438c81fb9ef90eba4ebbf42cfc6ccfbfe11e7e36cbedf862bf5d18913ef57`.
- Frozen index SHA-256:
  `35be7096a3e482b94f429ccf17dbeaaf70732f0e5b7bed690d9a27ecbbdcb1a6`.
- Maximum recomputed current-distance error: `1.91e-6`.
- Maximum recomputed candidate-distance error: `2.86e-6`.
- Recomputed oracle opportunity: `87.4284824%`, exactly matching the trace.

The trace stores HNSW internal IDs. Since the index was built in parallel,
those IDs do not equal fvecs row labels. An initial unmapped diagnostic was
invalidated; all formal results below use the mapping extracted from the exact
frozen index.

## Pivot audit

The metric is `||q-p|| / ||q-c||`.

| Pivot | P10 | P50 | P90 | P99 |
|---|---:|---:|---:|---:|
| Query mean | 0.9126 | 1.0361 | 1.3045 | 3.1239 |
| Zero | 1.8821 | 2.1442 | 2.3474 | 2.5047 |

The query-mean pivot helps only the lower part of the distribution. Its median
factor is already above one and its tail substantially enlarges query-side
uncertainty, so the single-global-pivot assumption fails operationally.

## Formal replay configuration

- Dimension: 960.
- Primary code: PQ32x8 = 32 bytes per edge.
- Training records: 50,000.
- OPQ iterations: 3.
- KMeans iterations: 40.
- Query split: 581 train / 213 validation / 206 test.
- Methods: Identity PQ, Random/JQ, Edge-OPQ, weighted ETV1-OPQ.
- Pivots: query mean and offline `p=c` oracle.
- Status: development-only, because the 1,000 GIST queries have been used in
  earlier method development.

## Test-split results

| Method | Pivot | Saving | Queries with saving | Median query saving | gamma P50 | gamma P90 | Direction MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Identity PQ | Query mean | 0.00693% | 0.971% | 0% | 0.9990 | 1.3058 | 6.434e-4 |
| Random/JQ | Query mean | 0.00693% | 0.971% | 0% | 1.0941 | 1.4681 | 7.307e-4 |
| Edge-OPQ | Query mean | **0.01848%** | **2.913%** | 0% | **0.8115** | **1.0827** | **4.422e-4** |
| Weighted ETV1-OPQ | Query mean | 0.01386% | 2.427% | 0% | 0.8186 | 1.0949 | 4.494e-4 |
| Edge-OPQ | `p=c` oracle | **0.10161%** | **12.136%** | 0% | **0.7876** | **0.8399** | 4.422e-4 |
| Weighted ETV1-OPQ | `p=c` oracle | 0.07390% | 9.223% | 0% | 0.7952 | 0.8468 | 4.494e-4 |

All method/pivot branches have zero deterministic violations. The maximum
test exact-distance alignment error is `2.71e-6`.

## Gate decision

| Condition | Gate | Actual | Result |
|---|---:|---:|---|
| Deterministic violations | 0 | 0 | Pass |
| ETV1/global deterministic saving | >=10% | 0.01386% | Fail |
| Queries with nonzero saving | >=75% | 2.427% | Fail |
| Median per-query saving | >=5% | 0% | Fail |
| ETV1 absolute gain over Edge-OPQ | >=2 pp | -0.0040 pp | Fail |
| Paired-query bootstrap CI lower bound | >0 | -0.0119 pp | Fail |

Even the non-shareable `p=c` oracle reaches only 0.10161%, approximately 98
times below the 10% minimum. The failure is therefore not merely a poor global
pivot. Under the current 32-byte direction-PQ plus reconstruction-ball
certificate, quantization uncertainty remains too large.

Edge-OPQ reduces direction MSE by about 31% relative to Identity PQ, and lowers
median gamma from about 0.999 to 0.811, but saving remains at a few ten-
thousandths. This confirms that lower reconstruction MSE does not imply a
threshold crossing. Earlier counterfactual analysis indicated that typical
effective radius must reach roughly 0.451 of the current radius; even the
`p=c` Edge-OPQ oracle has median gamma 0.788.

Weighted ETV1-OPQ does not improve Edge-OPQ on MSE, gamma, saving, or the
paired-query bootstrap comparison. The present pruning-aware surrogate is not
retained.

## Research boundary

Do not proceed with outward metadata quantization, an ETV1 C++ sidecar, query
LUT engineering, online shadow/real pruning, or lazy residual performance
work for this representation.

Any continuation of the edge mainline should first construct a different
representation/certificate and demonstrate at least 10% ideal float64 saving
on the same frozen stream. The next deterministic question is the
representation-optimal support over the PQ code cell: unit-sphere geometry,
block constraints, and PQ Voronoi half-spaces. If that `LB*` also fails, the
32-byte representation must change rather than the inequality implementation.

The full local replay outputs are intentionally ignored by Git under
`results/edge_transform_v1/`.
