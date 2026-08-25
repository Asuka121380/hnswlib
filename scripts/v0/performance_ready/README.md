# V0 performance-ready runner

The performance build deliberately keeps the V0 feature compiled in for every
method. `baseline`, `approx-no-retry`, and `approx-retry` are selected at run
time in the same executable, so compiler flags, index loading, query loading,
timing, and result serialization are identical. The baseline method does not
load or access a sidecar.

Before collecting timing data, validate the fresh build directory:

```text
python scripts/v0/performance_ready/check_build_contract.py \
  --build-dir build-v0-performance-ready --contract performance
```

Also run `audit_fast_path_source.py --header hnswlib/edge_quant_v0.h`.
It fails if strict-only rounding, padding, checked codebook decoding, or the
strict bound result type leaks back into the `raw_fast_v1` LUT/evaluator.

Run one method per process. Use the same query slice, `k`, `ef-search`, warmup,
repeat count, and optional CPU pin for every method. Approximate methods require
an explicitly calibrated `--beta`; the runner rejects beta values below one.
The JSON is written only after all timed repetitions finish.

```text
v0_performance_runner --method baseline --index-path INDEX \
  --query-path QUERY.fvecs --dimension 128 --query-count 10000 \
  --output baseline.json --repeats 5

v0_performance_runner --method approx-no-retry --index-path INDEX \
  --sidecar-path INDEX.v0eq --query-path QUERY.fvecs --dimension 128 \
  --query-count 10000 --beta CALIBRATED_BETA --output approx.json \
  --repeats 5 --prefetch gate
```

`raw_fast_v1` changes the numerical kernel, so beta values calibrated for the
strict/reference estimator must not be reused for formal performance claims.

For the formal warm-cache block experiment, use `run_latin_square.py`. It
freezes a seeded balanced six-order schedule before the first process starts,
runs each method in an isolated process, hashes the runner/CMake cache/results,
and updates `manifest.json` after every completed position. Use
`--aa-baseline` first to detect a stable order effect without changing the
schedule machinery.

`v0_fast_kernel_microbenchmark` produces the component costs required by the
plan. Combine its JSON with an active metrics `summary.json` using
`evaluate_break_even.py`; the script applies the frozen 25% safety-headroom
gate and exits nonzero when the estimated overhead exceeds the allowance.
