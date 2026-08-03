# V0 shadow schema v2

Schema v2 decomposes the current conservative radius while preserving the
schema-v1 current-bound semantics.

## Identity and search state

`schema_version`, `run_id`, `query_id`, `current_node_id`, `candidate_id`,
`graph_layer`, `bound_status`, `ef_search`, `current_squared_distance`, and
`threshold`.

The current implementation only evaluates layer 0, so `graph_layer` is zero.

## Bound reconstruction fields

`edge_length`, `direction_error`, `anchor_projection`,
`anchor_projection_lower`, `query_direction_inner_product_upper`,
`residual_direction_inner_product_upper`, `length_squared_lower`,
`cross_term_upper`, `base_plus_length_lower`,
`approximate_squared_distance`, and `current_distance_root_upper`.

All `*_upper` and `*_lower` values are the actual directed-rounding results
used by `EdgeQuantV0QueryContext::evaluate()`.

## Radius components

```text
direction_error_radius
stored_numeric_padding
operational_l2_padding
rounding_closure_padding
error_radius
```

The first three values are semantic components. `rounding_closure_padding` is
the non-negative difference introduced by the two upward-rounded additions:

```text
error_radius = direction_error_radius
             + stored_numeric_padding
             + operational_l2_padding
             + rounding_closure_padding
```

The equality is validated within CSV round-trip tolerance. The exact online
value remains the recorded `error_radius`; the decomposition never replaces
the production computation.

## Current method and exact validation

`lower_bound`, `shadow_exact_squared_distance`, `would_prune`,
`lower_bound_valid`, `lower_bound_violation`, and `false_prune` retain their
schema-v1 meaning. `current_lb` and `current_would_prune` are explicit
schema-v2 aliases, and `oracle_would_prune` is the exact-distance opportunity
flag. The analyzer requires the aliases and decisions to agree exactly.

Reserved nullable fields are `cap_lb`, `cap_would_prune`, `blockwise_lb`,
`blockwise_would_prune`, `repr_lb_star`, and
`repr_lb_star_would_prune`. Empty means the method was not evaluated; it must
not be interpreted as a zero lower bound.

## Derived analysis fields

```text
margin              = exact_distance - threshold
raw_margin          = approximate_distance - threshold
current_prune_gap   = lower_bound - threshold
safety_slack        = exact_distance - lower_bound
radius_margin_ratio = error_radius / max(abs(margin), eps)
```

`safety_slack` replaces the ambiguous historical name `safe_gap`.

## Safety gates

- all numeric fields for valid bounds are finite;
- radius components close to `error_radius`;
- the recorded lower bound can be reconstructed;
- `would_prune` equals `lower_bound > threshold`;
- summary mismatch, lower-bound violation, false-prune, and shadow-mode exact
  distance saved counts are all zero.

The summary and per-query metrics also contain full, unsampled
`raw_prunable` and `oracle_prunable` counters. Together with `bound_pruned`,
these provide raw/oracle/current coverage over the same valid bound attempts.
