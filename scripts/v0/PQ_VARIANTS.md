# Parameterized PQ variants (P2)

Branch: `p2-pq-m16-b6`, based on P1 commit `5a4e9d0`.
The first experiment is PQ M=16, b=6, with the original HNSW graph unchanged.

## Scope and representation

`train_pq_codebook.py --M-pq M --nbits b` now accepts b=1..8.
The existing V0PQ and sidecar formats retain **one byte per subquantizer**.
Faiss compact codes are converted offline; the online lookup has no unpack step.
M16,b6 means 64 centroids per subspace, 16 code bytes per edge, a 4 KiB LUT,
and a 56-byte full edge record. It does NOT mean 12-byte packed sidecar codes.
Old M32,b8 code paths and artifacts remain valid.

## Prerequisites

- Python with NumPy and CPU Faiss for training and reconstruction checks.
- Rebuild `v0_faiss_edge_encoder` and `v0_sidecar_inspector` from this branch.
  Do not reuse the old b8-only encoder executable.
- A frozen HNSW index and its unit-edge-direction sample plus manifest.
  The manifest must contain the matching `index_sha256`.
- Use a NEW run directory; existing directories and partial runs are refused.
- Commit changes before cluster runs for reproducibility. The pipeline records
  git HEAD, dirty status, executable hashes, input hashes and resolved options.

Build using the project's existing Faiss CMake package or FAISS_INCLUDE_DIR /
FAISS_LIBRARY settings, with HNSWLIB_ENABLE_EDGE_QUANT_V0=ON and
HNSWLIB_BUILD_FAISS_PQ_TOOLS=ON. Keep this build separate from P1 builds.

## One variant

The following paths are placeholders to replace with cluster paths:

```bash
export V0_PYTHON=/path/to/faiss/python
variant_args=(
  --pq-m 16 --nbits 6
  --directions /path/to/directions.f32
  --sample-manifest /path/to/sample_manifest.json
  --index-path /path/to/frozen_index.bin
  --encoder /path/to/new-build/v0_faiss_edge_encoder
  --inspector /path/to/new-build/v0_sidecar_inspector
  --output-root /path/to/new/m16-b6-run
)
bash scripts/v0/submit_pq_variant.sh --dry-run "${variant_args[@]}"
bash scripts/v0/submit_pq_variant.sh \
  --partition testing --qos normal --nodelist gpusrv-2 \
  --cpus-per-task 8 --mem 32G --time 10:00:00 -- "${variant_args[@]}"
```

Dry run validates/hashes inputs and prints the plan but does not create outputs
or submit jobs. These hashes read the full index/sample and can take time.
Submission uses **one sequential Slurm job**, with stage-specific logs. This
simplifies failure handling compared with a multi-job dependency chain.
Scheduler CPU allocation controls training threads; other trainer options include
`--seed`, `--split-seed`, `--iterations`, `--nredo`, and `--validation-fraction`.
`--faiss-version` and `--faiss-source-commit` describe the C++ Faiss dependency;
Python Faiss version is automatically recorded by the trainer.

Training -> codebook validation -> Faiss reconstruction parity -> all-edge
encoding -> C++ graph/sidecar inspection -> Python sidecar validation -> COMPLETE.
Any stage failure stops the pipeline; inspect `FAILED.json` and `logs/`.
This does not auto-resume; retain failed artifacts and use a new run directory.

The new wrapper defaults `--max-points-per-centroid 0` to a derived cap large
enough to avoid Faiss internal training subsampling. The standalone trainer keeps
its old default 256 for backwards compatibility. Fix the source sample, split,
seeds and training budget across variants; do not claim the same effective sample
as a historical b8 run unless its cap is checked. Explicit positive caps are
supported and their expected subsampling is recorded. Peak memory grows with
sample count; the default Slurm allocation is a starting point, not a guarantee.

## Outputs and evaluation

`training/codebook.v0pq`, training metrics, validation errors and reconstruction
report; `encoding/sidecar.v0meta`, encoding metrics and validation report;
`resolved_config.json`, `logs/`, and `COMPLETE` only after successful validation.

Pass the new `encoding/sidecar.v0meta` via `SIDECAR_PATH` to the existing component,
quality and QPS runners. Keep P1's M32,b8 sidecar untouched. Re-sweep efSearch/beta
and measure matched recall; beta=1.45 is not a quality guarantee for the new codebook.
Use the same runtime build for old/new sidecar A/B. No QPS improvement is claimed
from unit tests or theoretical LUT size alone.

## Validation status

Local tests cover b=1..8 packing (including non-byte-aligned rows), invalid codes,
b4/b6/b8 synthetic all-edge encoding including zero edges, M16,b6 LUT/estimator
identity, trainer contracts, dry-run and failure/no-overwrite behavior.
Local Windows environment does not provide Faiss: compilation/execution of the
real Faiss encoder and full train/encode pipeline still require cluster validation.
No cluster jobs or full GIST training have been launched by this implementation.
