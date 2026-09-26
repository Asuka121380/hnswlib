# Unified edge-estimation implementation manual

This branch implements the contracts specified by
`C:\ANU\IndividualProject\9.24\2026-09-24_unified_quantization_infrastructure_implementation_manual.md`.
That dated document remains the complete design record. This in-repository
copy records implementation status so planned interfaces are not mistaken for
completed experiments.

Implemented: M0 branch isolation, hash-capable preflight and append-only stage
ledger, M1 canonical UQEV/UQLB/UQQR
formats and state-machine validation, an independent layer-0 edge catalog,
legacy quality-only import with explicit capability restrictions, M2
compile-time capture observer, synthetic transparency check, and explicit
frozen-index/fvec/split capture, M3 score/policy contracts and PQ/PQ+QJL score adapters,
M4 ordered replay timing boundary, randomized paired-block orchestration,
build/source/binary fingerprint checker, and a real bit-packed PQ backend
covering PQ4 and general 1–8 bit subcodes. The packed-PQ path now includes
deterministic training, streaming full-graph encoding, artifact hash/identity
validation, native quality, native timing, and an end-to-end synthetic fixture.
The IVF coarse-plus-residual composition and active no-prune/fallback API
boundaries also exist without changing the production traversal.

Optional backend paths are present for OPQ, PRQ, JQ, RaBitQ and SAQ, but they
intentionally report `unavailable` until their named external source/library
and native adapter are supplied. They never silently fall back to Python or a
different estimator.

The infrastructure is synthetic-complete, not experiment-complete. Formal
work now waits on a filled and frozen `assets.json`, its split and hashes, plus
the requested optional backend dependencies/artifacts. Those inputs are needed
to run real GIST capture, legacy PQ/PQ+QJL parity, the candidate comparison,
Q* selection and Recall–QPS validation. A full active-search traversal migration
remains a later M7 experiment step; its policy boundary exists, but it is not
misrepresented as part of static infrastructure validation.
