# Unified edge-estimation implementation status

This records code and validation gates, not performance conclusions.

| Formal configuration | Trainer/exporter | Native artifact runtime | Remaining cluster gate |
|---|---|---|---|
| PQ32x8 | Faiss 1.15.1 | `uq-pq-packed/1` | frozen GIST artifact and C1 quality/timing |
| PQ64x4 | Faiss 1.15.1, canonical low-nibble-first packing | `uq-pq-packed/1` | frozen GIST artifact and C1 quality/timing |
| OPQ32x8 | Faiss 1.15.1 | `uq-rotated-pq/1` | frozen GIST artifact and C1 quality/timing |
| PRQ16x2x8 | Faiss 1.15.1 | `uq-prq/1` | frozen GIST artifact and C1 quality/timing |
| JQ32x8 | fixed single-level behavioral port | shared `uq-rotated-pq/1` | author-binary oracle is desirable; implementation identity must remain `legacy_jq_behavioral_port` |
| RaBitQ-1bit | Faiss 1.15.1, zero centroid, `qb=0` | `uq-rabitq/1` | frozen GIST artifact and C1 quality/timing |

These are the six formal static configurations. SAQ is the conditional seventh candidate and
remains deliberately unavailable because its fixed-budget estimator and ISA admission have not
passed; it does not block the six-configuration matrix.

The generic native commands `validate-artifact`, `quality`, and `bench-artifact` dispatch once
from `native.cfg`. Query files are validated and loaded outside replay timing; rotation and LUT
construction remain inside `prepareQuery`. Quality output contains aggregate confusion counts,
global one-sided error percentiles, and per-query counts/percentiles. Formal timing requires
separate validation and quality report identities at the Python orchestration layer.

Legacy PQ and PQ+QJL now have versioned artifact wrappers, sidecar/companion hash validation,
stable source-slot/edge checks, query lifecycle wiring, and native dispatch when the corresponding
V0 build options are enabled. `wrap-legacy` creates the wrapper artifact. Actual parity against the
historical M32b8 sidecar and QJL companion remains a cluster asset gate, not unfinished local code.

## Local evidence

- Synthetic persisted-index capture drives all six configurations through real training/export,
  native loading, validation, and quality; representative timing paths are also exercised.
- The JQ path is explicitly a behavioral port and is never reported as the unavailable author
  binary or as full JHQ.
- RaBitQ's exported byte layout and linear IP estimate were checked against the selected Faiss
  provider before the native scalar kernel was admitted.
- Non-identity internal-ID mapping, canonical 4-bit packing, rotated score orientation, query-store
  timing boundary, artifact rejection, D=960 model shapes, split conversion, and expected-hash
  rejection have local regression coverage.
- The default-feature-off build contains no Faiss runtime dependency; Faiss is a trainer-only
  dependency exchanged through versioned little-endian artifacts.

See `RUNBOOK.md` for the exact local-to-cluster handoff and the distinction between local code
completion, C1 static performance validation, and C2 end-to-end search conclusions.
