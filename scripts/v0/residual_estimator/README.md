# Residual estimator prototype

This directory contains the A0–D experiment entry points. The existing
`p2-pq-m16-b6` checkout is not modified.

The real run needs an `assets.json` with `index`, `sidecar`, `queries`,
`trace`, `ground_truth`, and `dataset_config` entries. Each entry is either
an absolute path or `{ "path": "...", "sha256": "..." }`. The index and
sidecar are validated against their existing identities; no PQ retraining is
performed. The historical trace is the frozen phase-1
`cap_diagnostic_input.csv`.

Build with `HNSWLIB_ENABLE_EDGE_QUANT_V0=ON`,
`HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON`,
`HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR=ON`, and
`HNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING=ON` for active search. These options
default to OFF. Use separate reference and performance build directories.

Commands below assume the environment provides NumPy, Pandas and PyArrow.
All formal runs require a clean implementation commit descended from the
frozen base commit.

Build the reference and performance binaries from the same commit. Enable
`HNSWLIB_ENABLE_EDGE_QUANT_V0`, `HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING`,
`HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR`, and
`HNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING` in both builds. The reference build
may collect quality metrics; the performance build must follow the existing
performance contract. Check both with
`scripts/v0/performance_ready/check_build_contract.py` before formal timing.

```bash
python scripts/v0/residual_estimator/freeze_contract.py --config configs/v0/residual_estimator/prototype_v1.json --assets assets.json --output "$RUN_ROOT/contract"
python scripts/v0/residual_estimator/build_dataset.py --contract "$RUN_ROOT/contract/manifest.json" --exporter build-reference/v0_residual_exporter --output "$RUN_ROOT/dataset"
python scripts/v0/residual_estimator/encode_sketch.py --contract "$RUN_ROOT/contract/manifest.json" --dataset "$RUN_ROOT/dataset/manifest.json" --output "$RUN_ROOT/sketch"
python scripts/v0/residual_estimator/replay_frontier.py --dataset "$RUN_ROOT/dataset/manifest.json" --sketch "$RUN_ROOT/sketch/manifest.json" --split development,selection --output "$RUN_ROOT/replay-selection"
python scripts/v0/residual_estimator/select_candidates.py --input "$RUN_ROOT/replay-selection/manifest.json" --output "$RUN_ROOT/selection.json"
build-reference/v0_residual_encoder --index INDEX --sidecar SIDECAR --matrix "$RUN_ROOT/sketch/seed-42/k-64/matrix.f32" --output "$RUN_ROOT/companion/seed-42-k64.v0res" --dimension 960 --bits 64 --seed 42 --chunk-edges 256
python scripts/v0/residual_estimator/run_active_matrix.py --stage quality --config configs/v0/residual_estimator/active_quality_v1.json --contract "$RUN_ROOT/contract/manifest.json" --selection "$RUN_ROOT/selection.json" --runner build-reference/v0_search_runner --companion "$RUN_ROOT/companion/seed-42-k64.v0res" --output "$RUN_ROOT/active-quality"
python scripts/v0/residual_estimator/select_matched_recall.py --quality "$RUN_ROOT/active-quality/manifest.json" --config configs/v0/residual_estimator/active_qps_v1.json --output "$RUN_ROOT/matched-cases.json"
python scripts/v0/residual_estimator/run_active_matrix.py --stage qps --config "$RUN_ROOT/matched-cases.json" --contract "$RUN_ROOT/contract/manifest.json" --selection "$RUN_ROOT/selection.json" --runner build-performance/v0_performance_runner --companion "$RUN_ROOT/companion/seed-42-k64.v0res" --output "$RUN_ROOT/active-qps"
python scripts/v0/residual_estimator/summarize_active.py --quality "$RUN_ROOT/active-quality/manifest.json" --qps "$RUN_ROOT/active-qps/manifest.json" --output "$RUN_ROOT/analysis/active-summary.json"
```

Replace `64` with the primary `selected_bits[0]` in `selection.json`. The
encoder accepts `--chunk-edges` from 1 to 4096 and prints record count,
invalid count, elapsed seconds, and output bytes. The selected `matrix.f32`
must match the seed and bit length passed to the encoder.
Pass `--resume` after an interrupted encoder run to continue from its last
complete checkpoint. The encoder verifies the saved index/sidecar identity,
matrix bytes, partial size and checkpoint payload SHA before appending.
If interrupted in the middle of a chunk, the partial file has extra records
and resume refuses it; remove both `.partial` and `.checkpoint` to restart.
For the Slurm C stage, set `RESIDUAL_RESUME=1` to pass `--resume`.

On Slurm, `submit_prototype.sh` accepts `--stage A0|A1|A2|B|C|D-quality|D-match|D-qps`
plus `--partition --qos --node --memory --time --cpus --resource-profile`.
The node option is optional; the others are required. Set
`RESIDUAL_RUN_ROOT`, `RESIDUAL_PYTHON`, `RESIDUAL_REFERENCE_BUILD`, and
`RESIDUAL_PERFORMANCE_BUILD` to absolute paths. A0 also needs
`RESIDUAL_ASSETS`; C needs `RESIDUAL_MATRIX`, `RESIDUAL_COMPANION`,
`RESIDUAL_BITS`, and `RESIDUAL_SEED`; D quality and QPS need
`RESIDUAL_COMPANION`. Submit each stage after its input manifests exist.
The wrapper checks Python dependencies and writes Slurm logs under `logs/`.

The full-graph encoder consumes the frozen `matrix.f32` in one
`sketch/seed-*/k-*` directory. Its `.v0res` output is loaded by both quality
and performance runners. Never mix matrices, codes, scales or offsets from
different seeds or code lengths.

The C++ tests cover file identity and corruption, bit order, the real encoder
on a small graph, invalid-record fallback, and the no-retry search path.
The Python tests cover split isolation, strict ties, event denominators and
paired QPS blocks. Run them with `python -m unittest discover -s tests/python
-p 'v0_residual_*_test.py'`. Historical 1/16 records represent sampled
exposure only. Formal results require cluster assets absent from the local
checkout; `complete.json` is written only after output validation.

The encoder's `.partial` output is incomplete and cannot be loaded. Keep
completed companions in distinct paths per seed and code length. The exporter writes a
fixed-layout `edges.bin`; the dataset stage creates `edges.parquet` and
`events.parquet` for audit and analysis. The kernel
microbenchmark isolates synthetic component costs; matched-recall QPS is the
end-to-end performance measurement.
