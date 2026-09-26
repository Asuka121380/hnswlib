# Six-method edge-estimation runbook

## Meaning of “locally complete”

Local completion means the six formal configurations have executable trainer/exporter/native
paths, synthetic correctness evidence, frozen configuration files, failure-path checks, and no
known code placeholder in their static evaluator path. It does **not** mean their GIST1M quality,
speed, or Recall-QPS ranking is known. Those conclusions require cluster stages C1 and C2.

Use the dedicated Faiss environment for training and the capture-enabled C++ build for native
validation. All commands below create new output directories and refuse to overwrite existing
results.

```powershell
$repo = 'PATH_TO_REPOSITORY'
$py = 'PATH_TO_FAISS_PYTHON'
$capture = "$repo/build-uq-capture/uq_capture.exe"
$runner = "$repo/build-uq-capture/uq_native_runner.exe"
$assets = 'PATH_TO_ASSETS_JSON'
$oldSplit = 'PATH_TO_FROZEN_OLD_SPLIT_JSON'
$run = 'PATH_TO_NEW_RUN_DIRECTORY'

Set-Location $repo
& $py scripts/edge_estimation/run.py convert-split --input $oldSplit --query-count 1000 --out "$run/split"
& $py scripts/edge_estimation/run.py preflight --assets $assets --config configs/edge_estimation/methods/pq8.json --stage formal --hash-assets --out "$run/preflight"
& $py scripts/edge_estimation/run.py catalog --assets $assets --dimension 960 --runner $capture --out "$run/catalog"
& $py scripts/edge_estimation/run.py capture --assets $assets --config configs/edge_estimation/methods/pq8.json --split "$run/split/split.json" --split-name development --runner $capture --out "$run/development"
& $py scripts/edge_estimation/run.py validate --events "$run/development/events.bin" --labels "$run/development/labels.bin" --ranges "$run/development/query_ranges.bin" --out "$run/dataset-validation"
```

For each name in `pq8,pq4,opq,prq,jq,rabitq`, run:

```powershell
& $py scripts/edge_estimation/run.py train-encode --trainer-python $py --assets $assets --config "configs/edge_estimation/methods/$name.json" --catalog "$run/catalog" --out "$run/artifact-$name"
& $py scripts/edge_estimation/run.py validate --runner $runner --events "$run/development/events.bin" --queries 'PATH_TO_QUERIES_FVECS' --artifact "$run/artifact-$name" --out "$run/validation-$name"
& $py scripts/edge_estimation/run.py quality --runner $runner --events "$run/development/events.bin" --labels "$run/development/labels.bin" --queries 'PATH_TO_QUERIES_FVECS' --artifact "$run/artifact-$name" --alpha 0.8 --alpha 0.9 --alpha 1.0 --alpha 1.1 --out "$run/quality-$name"
& $py scripts/edge_estimation/run.py bench --runner $runner --events "$run/development/events.bin" --queries 'PATH_TO_QUERIES_FVECS' --artifact "$run/artifact-$name" --validation "$run/validation-$name/validation.json" --quality-report "$run/quality-$name/quality.json" --formal --repeats 5 --out "$run/timing-$name"
```

The formal assets JSON should contain path strings plus an `expected_sha256` object for `index`,
`queries`, `base`, `internal_to_label`, and `ground_truth`. A computed hash with no expected value
is recorded but is not evidence that the asset matches the frozen experiment identity.

## Legacy baseline wrapping

When the historical sidecar exists, wrap it without changing its bytes:

```powershell
& $py scripts/edge_estimation/run.py wrap-legacy --sidecar 'PATH_TO_M32B8_SIDECAR' --catalog "$run/catalog" --out "$run/artifact-legacy-pq"
& $py scripts/edge_estimation/run.py wrap-legacy --sidecar 'PATH_TO_M32B8_SIDECAR' --residual 'PATH_TO_QJL_COMPANION' --catalog "$run/catalog" --out "$run/artifact-legacy-pq-qjl"
```

Then use the same validate/quality/bench commands. The loader checks internal sidecar checksums,
wrapper file hashes, catalog identity, edge count, source/slot identity, and companion identity.

## Cluster sequence

1. Rebuild the exact source revision and run the complete local fixture suite on Linux.
2. Verify all formal asset hashes and their semantic relationship before encoding.
3. Run a small development capture and legacy parity first.
4. Encode all six full-graph artifacts once, then run quality and randomized paired static timing.
5. Select Q* only from C1 evidence. Run IVF+PQ, IVF+Q*, and PQ+QJL+theta as active HNSW searches
   in C2 and compare Recall@10/QPS on held-out queries.

Static replay cannot establish end-to-end Recall or QPS, and local Windows timings must not be
ratioed against Linux cluster timings.
