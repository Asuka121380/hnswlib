# EQ-RCP Stage 2.1 Result

Implementation and local GIST1M development pilot completed on 2026-08-10.

- Synthetic unit/integration test: PASS
- Original Stage 2 regression test: PASS
- Deterministic repeated synthetic semantic hash: PASS
- GIST1M pilot validator: PASS (27 method/seed pairs)
- Pilot decision: `PENDING_FORMAL_STAGE2_1`
- Pilot-selected representation: `gain_shape_opq`
- Pilot semantic SHA-256: `3e1991b160391a55fb09edf06664399073136441eca3b5c004d364259b8b6799`
- Result directory: `results/eq_rcp/stage2_1/development/20260810-191822-959`

The pilot used 40 quantizer-training, 30 decoder-training, and 40 independent
model-selection queries from the existing development Stage 1 materialization.
It used three seeds and a fixed 32-byte edge-code budget. It is not formal Gate
evidence and cannot authorize C++ integration.

## Mean held-out Q99 projection error

| Representation | Q99 |
|---|---:|
| raw direct-Delta OPQ | 0.478853 |
| norm-corrected OPQ | 0.369760 |
| direction PQ + exact length | 0.409539 |
| gain-shape OPQ + exact length | 0.356867 |
| grouped diagonal OAE | 0.513459 |
| grouped block OAE | 0.523117 |
| operational-refined OPQ | 0.496953 |

Relative to raw OPQ, norm correction reduced Q99 by 22.78%. Gain-shape OPQ
reduced it by 25.47%, with positive improvement on every seed and a paired-query
95% CI of `[0.07038, 0.09364]` for raw-OPQ Q99 minus challenger Q99.

The matched OPQ contrast indicates that exact length explains most of the gain;
gain-shape parameterization contributes a further roughly 3.5% Q99 reduction
relative to norm-corrected OPQ. This is development evidence only.

Grouped OAE and the current one-step operational refinement did not pass the
representation Gate. The refinement path remains protected by held-out rollback;
its failure does not invalidate the exact-length/gain-shape candidates.

## Five-way full-development Gate

Validated run: `results/eq_rcp/stage2_1/development_full/20260810-200725-326`.

- Five-way split: 427 quantizer-train / 154 decoder-train / 156 model-selection /
  155 risk-calibration / 108 untouched final-test queries
- Methods/seeds: 9 methods x 3 seeds; all 27 selection pairs completed
- Model-selection freeze: `gain_shape_opq`
- Calibration: all selected/raw-OPQ scalar and code-oracle cutoffs passed the
  95% one-sided event-level risk UCB at target 1%
- Validator: PASS
- Semantic SHA-256: `85b33580c8c1be019061544a274bf4336be2c7c3e871442367a3de795600bd48`
- Gate decision: `NO_GO_REPRESENTATION` under the frozen Gate wording

On untouched final-test queries, gain-shape OPQ versus raw OPQ achieved:

| Metric | raw OPQ | gain-shape OPQ |
|---|---:|---:|
| projection Q99 | 0.579125 | 0.421240 |
| projection Q999 | 0.881525 | 0.659875 |
| threshold-near Q99 | 0.534440 | 0.371169 |
| calibrated scalar coverage | 11.82% | 28.54% |
| calibrated scalar false-prune over good | 1.0535% | 1.0649% |
| code-aware oracle coverage | 49.18% | 53.11% |
| code-aware oracle false-prune over good | 0.7214% | 0.9619% |

Gain-shape passed Q99 reduction (27.26%), paired-query CI, Q999, threshold-near,
coverage, code-information, metadata, and all-seed stability. The only failed
criterion was the frozen scalar estimator's final empirical false-prune rate:
1.0649% versus the 1% target. Seed-level scalar risks were 0.7901%, 1.2367%, and
1.1680%.

This run must not authorize Stage 3 under the pre-registered Gate. It localizes
the remaining failure to estimator/calibration transfer rather than quantized-edge
projection quality: the independently calibrated code-aware estimator stayed below
1% while retaining 53.11% coverage. Any safety-margin or estimator change requires
a newly registered protocol and fresh untouched queries; the consumed final-test
split cannot be reused for tuning.

An earlier full run at `20260810-194653-861` is retained as an invalid engineering
diagnostic: it incorrectly enforced empirical risk during model selection and
therefore froze raw OPQ before independent calibration. It is not research evidence.
