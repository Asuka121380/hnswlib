# V0 residual estimator prototype

The implementation follows `C:/ANU/IndividualProject/9.10/EDGE_PQ_RESIDUAL_ESTIMATOR_PROTOTYPE_IMPLEMENTATION_PLAN_2026_09_19.md`.

Implementation is completed before formal A/B/C/D experiments. Offline
frontier and cost results inform the final interpretation; they do not stop
active search or matched-recall experiments.

The active search path is opt-in through
`HNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING`. The original `.v0meta` is unchanged.
Each `.v0res` companion contains one Gaussian matrix and records in the
original sidecar's global edge ordinal order. Load validates index, sidecar,
adjacency, matrix and payload identities before query timing. Query setup
computes `G*q` and a four-bit signed-sum LUT; a first-visit, heap-full
candidate is pruned only when the corrected estimate exceeds the exact heap
threshold. Exact distances remain in the result and candidate queues.

The implementation branch is `p3-residual-estimator-prototype` in a separate
worktree. Local regression covers enabled and disabled builds, C++ query/IO
and active-search tests, a synthetic index-to-encoder-to-search integration
test, and Python replay/paired-QPS tests. The frozen GIST1M index, production
M32b8 sidecar and ground truth are not present locally, so formal A/B/C/D
experiments have not run. The encoder can resume from verified complete chunk
checkpoints; interruption within a chunk requires a fresh encode. Production
numerical parity and performance remain unmeasured without frozen assets.

See `scripts/v0/residual_estimator/README.md` for commands and stage inputs.
