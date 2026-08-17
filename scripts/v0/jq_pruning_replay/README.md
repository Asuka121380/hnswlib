# JQ theoretical-pruning replay

This directory implements the experiment frozen in
`C:\ANU\IndividualProject\8.17\JQ_THEORETICAL_PRUNING_REPLAY_IMPLEMENTATION_PLAN.md`.
It evaluates one metric: the event-weighted rate at which a conservative JQ
edge-direction bound is greater than the threshold recorded by an unchanged
baseline HNSW search.

## Scope

The implementation is deliberately offline. It does not modify HNSW search
control flow, V0 sidecars, eager LUTs, lazy LUTs, or exact-distance execution.
It implements only the one-level primary JQ quantizer; JHQ residual levels are
out of scope.

The JQ behavior is ported from official JHQ commit
`1636e197a36871db4a79d66d22f9f9dd0fa51e31` (Apache-2.0). The public code uses
Faiss RNG, LAPACK QR, and libstdc++ random distributions. NumPy's RNG/QR are not
bitwise identical, so saved rotation and centroid arrays are part of every
reproducible run. The implementation does not claim bitwise parity with the
official binary.

The analytical initializer is fitted on all non-zero unique directed edges in
the frozen trace sample. This is a low-cost diagnostic substitute for the full
graph-edge population, which is unavailable locally without the original HNSW
index/sidecar. The run manifest records this fit source explicitly; results
must not be described as a full-graph JQ codebook experiment.

## Correctness model

For an original unit edge direction `u`, stored float32 rotation `R`, and
selected rotated centroid vector `y`, the reconstructed original vector is
`R.T @ y`. The replay bounds its error using

```text
||u - R.T y|| <= ||R.T|| ||R u - y|| + ||I - R.T R||.
```

The query-dependent dot product is evaluated from selected centroid cells in
rotation space with the same upward cell and accumulation semantics as V0.
The V0 float32 L2 padding is then applied. Every valid replay event must satisfy
`lower_bound <= shadow_exact_squared_distance` within a frozen numerical
tolerance; any violation aborts the run.

## Tests

From this directory, using the repository pipeline Python:

```powershell
& 'C:\ANU\IndividualProject\ann-dco-reproduction\.venv-msys\bin\python.exe' `
  -m unittest discover -s . -p 'test_*.py'
```

## Smoke replay

```powershell
& 'C:\ANU\IndividualProject\ann-dco-reproduction\.venv-msys\bin\python.exe' `
  run_replay.py `
  --repo 'C:\ANU\IndividualProject\Implementation\hnswlib' `
  --frozen-stage-dir 'C:\ANU\IndividualProject\8.3\20260803-minimal-v3\stageB-q1000' `
  --dataset-dir 'C:\ANU\IndividualProject\datasets\gist1m' `
  --output-dir 'C:\ANU\IndividualProject\Implementation\hnswlib\results\jq_pruning_replay\smoke-m32-s1234' `
  --subspaces 32 `
  --rotation-seed 1234 `
  --max-rows 256
```

Remove `--max-rows` and add `--enforce-frozen-contract` for a full frozen run.
Add `--verify-large-input-sha256` when the full 3.8 GB base-file hash should be
checked against `dataset.json` before running.
Run all three fixed seeds (`1234`, `17`, `42`) for both `M=32` and `M=120`,
then pass the six complete directories to `summarize_replay.py`.

## Interpretation boundary

`M=32` is the 32-byte iso-budget primary result. `M=120` is a 120-byte
subspace-dimension-eight upper-bound diagnostic and must not be presented as a
fair comparison with PQ-32B. A high theoretical pruning rate is only a gate
for a later lazy-LUT cost experiment; this replay provides no latency claim.
