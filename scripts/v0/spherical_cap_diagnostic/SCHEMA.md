# Spherical-cap Phase-1 schemas

## `cap_diagnostic_input.csv` (schema version 2)

This file is produced by the C++ observe-only runner for deterministically
sampled bound attempts.

| Field | Meaning |
|---|---|
| `query_id`, `current_node_id`, `candidate_id` | Stable sampled attempt identity. |
| `edge_length` | Stored edge length \(\ell\). |
| `direction_error` | Stored direction-error certificate \(\varepsilon\). |
| `threshold` | Search rejection threshold \(\tau\). |
| `current_lb` | Existing strict floating-point V0 LB; unchanged by this feature. |
| `exact_squared_distance` | Operational exact candidate squared distance computed by HNSW. |
| `geometric_squared_distance` | Diagnostic squared distance recomputed directly from query/candidate coordinates with long-double accumulation. This closes against the exported edge geometry and does not participate in search decisions. |
| `reconstruction_norm` | \(s=\lVert r\rVert\), reconstructed from the sampled PQ code and codebook with long-double accumulation. |
| `x_norm` | \(n=\lVert q-c\rVert\), recomputed directly from vectors. |
| `x_dot_r` | \(x^\top r\), recomputed directly from vectors and reconstructed PQ coordinates. |
| `true_edge_norm` | \(\lVert v-c\rVert\), recomputed directly. |
| `x_dot_true_direction` | \(x^\top (v-c)/\lVert v-c\rVert\). |
| `actual_direction_error` | \(\lVert (v-c)/\lVert v-c\rVert-r\rVert\). |
| `certificate_slack` | \(\varepsilon-\text{actual_direction_error}\); negative values are certificate failures. |
| `diagnostic_valid` | Whether all raw reconstruction values were finite and the true edge was nonzero. |
| `raw_would_prune` | Whether the uncorrected approximate squared distance exceeds \(\tau\); diagnostic only. |

The schema also carries run identity, graph layer/status, `ef_search`, current
distance, anchor projection, and current/oracle prune flags for joins and
regression checks.

## `cap_candidate_table.csv.gz` (schema version 1)

The analyzer preserves all input fields and adds:

- `kappa`, descriptive `cap_half_angle_radians`, `cap_kind`, `cap_branch`;
- `x_dot_r_hat`, `cap_support`, and independently computed `span_support`;
- `current_ball_support`, `current_ball_lb`;
- `cap_lb_high_precision`, `cap_would_prune`, `cap_gain`;
- `cap_safety_slack`, `gap_recovery`, `support_contraction`;
- explicit certificate, support-oracle, true-direction-support, LB-violation,
  cap-below-current, and false-prune flags;
- `cap_valid`.

Kinds are `empty`, `singleton`, `proper`, `full_sphere`, `s_zero`, or
`numeric_failure`. Branches are `inside`, `boundary`, or `fallback`.

## Report artifacts

- `summary.json`: Phase-1 decision, gates, coverage, geometry, tolerances.
- `cap_geometry_summary.csv`: P0/P1/P10/P50/P90/P99/P100 distributions.
- `cap_coverage_by_query.csv`: per-query coverage and extra-prune counts.
- `cap_data_quality.json`: certificate and safety failure counts.

All comparison tolerances and the Decimal precision are recorded in
`summary.json`.
