# Unified edge-estimation runbook

Configure the portable tool build:

```powershell
cmake -S . -B build-uq -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHNSWLIB_BUILD_EDGE_ESTIMATION_TOOLS=ON -DHNSWLIB_ENABLE_NATIVE_ARCH=OFF
cmake --build build-uq --target uq_native_runner
ctest --test-dir build-uq --output-on-failure -R uq_
python -m unittest discover -s tests/python -p "uq_*_test.py"
```

The capture-enabled build additionally registers `uq_end_to_end_fixture`. It
creates a persisted synthetic index and exercises catalog, real capture,
full-graph packed-PQ train/encode, artifact/hash validation, quality and timing.

Capture requires a separate diagnostic build with
`HNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE=ON` and graph access enabled. Synthetic
mode is reserved for transparency/smoke tests and marks its output
`formal_result_eligible=false`. Real mode requires explicit frozen paths and an
original-query-ID file; it never discovers assets by filename.

```powershell
build-uq-capture/uq_capture --synthetic RUN/dataset
build-uq/uq_native_runner validate RUN/dataset/events.bin RUN/dataset/labels.bin RUN/dataset/query_ranges.bin
python scripts/edge_estimation/run.py validate --events RUN/dataset/events.bin --labels RUN/dataset/labels.bin --ranges RUN/dataset/query_ranges.bin --out RUN/validation
```

Real capture:

```powershell
build-uq-capture/uq_capture --index INDEX --queries QUERIES.fvecs --query-ids query_ids.txt --dimension 960 --k 10 --ef 100 --out RUN/dataset
python scripts/edge_estimation/run.py capture --assets assets.json --config config.json --split split.json --split-name development --runner build-uq-capture/uq_capture --out RUN/dataset
```

Catalog, artifact and native validation:

```powershell
python scripts/edge_estimation/run.py catalog --assets assets.json --dimension 960 --runner build-uq-capture/uq_capture --out RUN/catalog --ledger RUN/ledger.jsonl
python scripts/edge_estimation/run.py train-encode --config config.json --assets assets.json --catalog RUN/catalog --out RUN/artifact --ledger RUN/ledger.jsonl
python scripts/edge_estimation/run.py validate --events RUN/dataset/events.bin --artifact RUN/artifact --queries DATA/queries.fvecs --runner build-uq/uq_native_runner --out RUN/validation --ledger RUN/ledger.jsonl
python scripts/edge_estimation/run.py quality --events RUN/dataset/events.bin --labels RUN/dataset/labels.bin --artifact RUN/artifact --queries DATA/queries.fvecs --runner build-uq/uq_native_runner --alpha 1.0 --out RUN/quality --ledger RUN/ledger.jsonl
```

For formal asset identity, run preflight with `--hash-assets`; without it the
audit deliberately sets `formal_asset_identity_complete=false`. A paired timing
matrix contains `methods` entries with unique `name` and frozen `artifact`, plus
`reference`, `blocks`, `repeats`, `seed` and `isa_profile`, and is passed with
`bench --matrix MATRIX --events ... --queries ... --runner ...`.

`uq_native_runner capabilities` is the native source of truth. OPQ/PRQ/JQ/
RaBitQ/SAQ report a concrete unavailable reason until their dependencies and
adapters are compiled. This is an expected capability result, not a failed PQ
smoke test.

Formal GIST runs must first replace every placeholder in `assets.example.json`,
freeze a split, validate hashes, build full-graph artifacts, and use a clean
commit. Do not publish synthetic, subset, or legacy sparse timing as QPS.
