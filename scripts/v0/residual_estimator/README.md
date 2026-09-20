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
The quality matrix and matched-recall selection use only development and
selection query IDs from the frozen split. Audit queries are excluded from
that selection; the later full-query QPS run occurs after cases are frozen.
Pass `--resume` after an interrupted encoder run to continue from its last
complete checkpoint. The encoder verifies the saved index/sidecar identity,
matrix bytes, partial size and checkpoint payload SHA before appending.
If interrupted in the middle of a chunk, the partial file has extra records
and resume refuses it; remove both `.partial` and `.checkpoint` to restart.
For the Slurm C stage, set `RESIDUAL_RESUME=1` to pass `--resume`.

On Slurm, `submit_prototype.sh` accepts `--stage A0|A1|A2|B|C|D-quality|D-match|D-qps|D-tune-quality|D-tune-supplement-quality|D-tune-fine-quality|D-tune-select|D-tune-qps|D-target-qps`
plus `--account --partition --qos --node --memory --time --cpus --resource-profile`.
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

The preliminary matched-recall run uses 11 quality cases: baseline ef
395/405/415/425/435, original PQ beta 1.45 at ef 450/465/480, and
128-bit residual-direct at ef 575/600/625. It targets recall@10 0.9555
with maximum spread 0.001. The timed comparison uses only these three
methods, three randomized blocks, five repeats, and 1000 queries per
case. An unmatched quality selection is rejected before QPS. Both
quality and performance binaries use `HNSWLIB_ENABLE_NATIVE_ARCH=OFF`.

The follow-up residual-only tuning uses the same frozen 128-bit companion
and query split. `D-tune-quality` measures 16 threshold cases: ef 500 with
theta 1.08/1.10/1.12/1.14, ef 525 with 1.06/1.08/1.10/1.12, ef 550 with
1.04/1.06/1.08/1.10, and ef 575 with 1.02/1.04/1.06/1.08. It also
remeasures residual-direct at ef 600 as an internal anchor. Ef 475 and
theta 1.26 are excluded. `D-tune-select` retains every threshold case
with measured development+selection recall@10 at least 0.9555 and the
direct anchor; if the anchor or every threshold case misses the floor,
selection fails before timing. `D-tune-qps` measures every retained case
in five randomized blocks, five repeats and 1000 queries per case, then
writes `analysis/residual-tuning-summary.json`. The ranking is by median
QPS within this residual-only run; the old PQ timing is historical context.
Audit query IDs remain excluded from tuning and may be evaluated once the
winner is frozen. Both staged builds must have native architecture OFF.
The quality runner must also advertise `--query-id-file` in `--help`;
the earlier residual reference binary does not support frozen query IDs.
`submit_prototype.sh` checks this before submitting quality jobs.

The optional `D-tune-supplement-quality` stage checks only three additional
points on the same frozen 800 query IDs: ef 475 with theta 1.08/1.10 and
ef 500 with theta 1.04. It writes to `residual-tuning-supplement-quality`
without changing the completed 17-case quality manifest or the existing
selection. Compare these recalls with the earlier cases before freezing a
smaller QPS shortlist.

`D-tune-fine-quality` checks ef 475/theta 1.07 and ef 500/theta 1.05,
using the same frozen query IDs and a separate
`residual-tuning-fine-quality` directory. Select QPS candidates against
the measured baseline ef 405 recall@10 of 0.95575, not against the
residual-direct anchor recall of 0.95550.

`D-target-qps` freezes measured recall from the completed active,
supplement, and fine-quality manifests. It uses the measured baseline ef
405 recall on the same frozen 800 query IDs and accepts cases within
plus or minus 0.001 of that recall. The five cases are baseline ef 405,
original PQ beta 1.45/ef 465, residual threshold ef 500/theta 1.04 and
1.05, and residual-direct ef 600. Selection rejects a case outside the
window. QPS uses five randomized paired blocks, 1000 queries, five
repeats per case, and native architecture OFF, then writes
`analysis/targeted-qps-summary.json`.

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
