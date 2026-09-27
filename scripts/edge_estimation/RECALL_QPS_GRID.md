# Unified end-to-end Recall-QPS grid

This experiment compares four live HNSW search paths over the same query range
and Recall@10 definition:

- unmodified HNSW;
- PQ8 active pruning (`approx-no-retry`);
- PQ8 plus QJL residual-threshold pruning;
- OPQ full-graph artifact pruning with batched query rotation.

The OPQ runner includes batched rotation and LUT preparation in total latency.
`opq_batch_size` therefore changes the measured end-to-end QPS rather than being
treated as offline work. Invalid estimates fail open to the exact HNSW distance.

## Build

Configure with the V0 implementations, unified active hook, and BLAS enabled:

```bash
cmake -S . -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON \
  -DHNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR=ON \
  -DHNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING=ON \
  -DHNSWLIB_BUILD_EDGE_ESTIMATION_TOOLS=ON \
  -DHNSWLIB_ENABLE_EDGE_ESTIMATION_ACTIVE=ON \
  -DUQ_WITH_BLAS=ON
cmake --build "$BUILD" --target v0_performance_runner uq_opq_performance_runner --parallel 4
```

For the cluster's wheel-bundled OpenBLAS, additionally pass the already verified
absolute library as `-DUQ_BLAS_LIBRARY="$UQ_BLAS"`.

## Run

Use `run_recall_qps_grid.py` directly, or submit the same contract through
`submit_recall_qps_grid_slurm.sh --help`. The default pilot grid is
`configs/edge_estimation/recall_qps_grid_v1.json`.

The output directory contains:

- `points.csv`: one Recall-QPS point per `(method, beta, efSearch)`;
- `summary.json`: the same data with the complete numeric fields;
- `manifest.json`: configuration, input identities, and the expanded grid;
- `cases/<case>/`: raw performance JSON, result rows, latency rows, command, and log;
- `complete.json`: only written after all points succeed.

`--resume` reuses a case only when both its `performance.json` and `results.csv`
exist. The runner writes `points.csv` after every completed point, so a failed
pilot still preserves all earlier measurements.
