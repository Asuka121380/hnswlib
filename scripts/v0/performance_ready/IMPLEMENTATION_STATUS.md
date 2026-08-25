# V0 Performance Readiness Implementation Status

Date: 2026-08-25

Branch: `v0-margin-retry-performance-ready`

Kernel: `raw_fast_v1`

## Current decision

The mandatory code-path and benchmark-infrastructure implementation is
complete. The repository is **not yet cleared for formal Stage 7 results**:
the GIST1M fast/reference replay, beta recalibration, held-out Recall-DCO gate,
prefetch A/B selection, A-A order test, and component break-even run require
the frozen index/sidecar/query artifacts and must be executed before selecting
one of the plan's final readiness decisions.

## Implemented contract

- M1: independent `EdgeQuantV0ApproxQueryContext::evaluateRawFast`; the old
  strict evaluator remains the reference oracle.
- M2: native-endian immutable codebook plus ordinary-arithmetic full LUT and
  unchecked contiguous-code lookup.
- M3: compile-time retry/no-retry, prefetch, and metrics specializations;
  threshold scaling is updated only when `lowerBound` changes.
- M4: visited-list-owned generation-tagged retry state with O(1) normal query
  reset and explicit generation-wrap clearing.
- M5: one checked fast edge span per expanded node and sequential unchecked
  records in the neighbor loop.
- M6: strict/reference and instrumentation-free performance build contracts;
  baseline and active timing use one binary and therefore the same exact-L2
  kernel and flags.
- M7: isolated single-method runner, latency samples, warmup/repeats, affinity,
  checksums, load-time fields, seeded balanced process schedule, environment
  manifest, and baseline A-A mode.
- M8: legacy-vector and gate-aware sidecar-first prefetch policies with result
  parity coverage. Policy selection remains a data gate.

The metrics runner additionally reports fast/reference valid pairs, status
disagreement, absolute/relative differences, gate-decision disagreement, and
near-threshold disagreement. The performance API instantiates the same search
with these metrics compiled out.

## Verified builds

- `build-v0-performance-ready`: Release, `-O3 -march=native`, strict FP OFF,
  trace/shadow/validation OFF, comparable flags ON.
- `build-v0-approx-reference`: Release, strict FP ON, shadow and approximate
  shadow ON.
- `build-v0-off`: V0 disabled; baseline example and search test compile.

Validation completed locally:

- performance build: 24/24 tests passed;
- reference build: 25/25 tests passed;
- `raw_fast_v1` forbidden-operation source audit passed;
- GNU 16.1.0 vectorization report confirms the fast LUT coordinate loop at
  `edge_quant_v0.h:309` uses 32-byte vectors with an unrolled vector loop and
  a 16-byte vectorized epilogue;
- `git diff --check` reported no whitespace errors (only expected Windows
  line-ending notices).

## Required data-gate sequence

1. Run the strict/reference metrics runner on the frozen calibration,
   validation, and held-out splits; inspect the new fast/reference fields.
2. Recompute the dense beta frontier for `raw_fast_v1`; do not reuse the old
   beta as a formal operating point.
3. Freeze separate no-retry/retry beta values before touching final performance
   queries and confirm at least 15% held-out exact-DCO reduction.
4. Run `v0_fast_kernel_microbenchmark` on the real GIST sidecar and feed its
   JSON plus the active metrics summary to `evaluate_break_even.py`.
5. Use development queries to select legacy or gate-aware prefetch, including
   available hardware counters.
6. Run `run_latin_square.py --aa-baseline` and reject stable order bias before
   running the three-method schedule.
7. Repeat the complete readiness run independently, compare top-k checksums,
   and then issue `READY_FOR_STAGE7`, `ONE_BOUNDED_OPTIMIZATION_REQUIRED`, or
   `NOT_PERFORMANCE_VIABLE_YET`.

Cold-start OS page-cache control, CPU frequency locking, power policy, and NUMA
binding remain environment responsibilities; the orchestrator requires these
policies to be named in its manifest rather than silently assuming them.
