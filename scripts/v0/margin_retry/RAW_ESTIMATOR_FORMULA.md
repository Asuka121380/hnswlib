# V0 Raw Estimator Formula and Stage 3 Contract

The active gate uses the existing V0 approximate squared-distance value without the strict lower-bound error-radius subtraction.

For current expanded node `c`, candidate neighbor `n`, and query `q`, the stored edge metadata gives edge length `l`, anchor projection `a`, and a PQ reconstruction of the unit edge direction. With `s_qc = ||q-c||^2`, V0 computes:

```text
u_upper       = PQ-LUT upper-rounded inner product
anchor_lower  = nextDown(a)
residual      = addUp(u_upper, -anchor_lower)
l2_lower      = multiplyDown(l, l)
cross_upper   = multiplyUp(2*l, residual)
base_lower    = addDown(s_qc, l2_lower)
raw_estimate  = addDown(base_lower, -cross_upper)
```

The active first-visit rule is:

```text
raw_estimate > beta * current_squared_threshold
```

The value is only a gate. It is never inserted into the candidate or result queues.

## Correctness-first implementation

`EdgeQuantV0QueryContext::evaluateRaw()` currently calls the existing strict `evaluate()` and returns exactly its `status` and `approximate_squared_distance`. Therefore Stage 3 parity is bit-for-bit by construction and is regression-tested across valid, exact-only, zero-length, invalid-distance, infinite, and NaN inputs.

Run metadata records:

```text
raw_fast_path=false
```

This implementation still computes strict-only error-radius fields and is not eligible for a QPS/latency claim. A dedicated raw fast path is deferred to Stage 7, after fixed-`efSearch` active Recall/DCO viability is established.

## Fail-closed cases

Invalid metadata, exact-only edges, zero-length edges, invalid current distances, and numeric failure all fall back to the exact distance. The active estimator is also bypassed until the result heap contains `efSearch` entries.
