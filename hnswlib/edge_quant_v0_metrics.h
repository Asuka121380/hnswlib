#pragma once

#include <cstdint>

namespace hnswlib {

struct V0QueryMetrics {
    // In observe-only searches, bound_pruned counts decisions that would
    // prune. In a real-pruning search it counts decisions actually taken,
    // and therefore equals exact_distance_saved.
    uint64_t bound_evaluated = 0;
    uint64_t bound_pruned = 0;
    uint64_t raw_prunable = 0;
    uint64_t oracle_prunable = 0;
    uint64_t exact_fallback = 0;
    uint64_t exact_only_fallback = 0;
    uint64_t exact_distance_saved = 0;
    uint64_t lower_bound_violation = 0;
    uint64_t false_prune = 0;
    uint64_t exact_distance_computed = 0;
    uint64_t expanded_nodes = 0;
    uint64_t edge_scans = 0;
    uint64_t duplicate_encounters = 0;

    void reset() {
        *this = V0QueryMetrics();
    }
};

#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
static const uint32_t V0_SHADOW_SCHEMA_VERSION = 2U;

struct V0ShadowRecord {
    uint64_t query_id = 0;
    uint64_t current_node_id = 0;
    uint64_t candidate_id = 0;
    uint64_t expansion_index = 0;
    uint64_t current_node_degree = 0;
    uint64_t candidate_degree = 0;
    uint32_t graph_layer = 0;
    uint8_t bound_status = 0;
    double current_squared_distance = 0.0;
    double threshold = 0.0;
    double edge_length = 0.0;
    double direction_error = 0.0;
    double anchor_projection = 0.0;
    double anchor_projection_lower = 0.0;
    double query_direction_inner_product_upper = 0.0;
    double residual_direction_inner_product_upper = 0.0;
    double length_squared_lower = 0.0;
    double cross_term_upper = 0.0;
    double base_plus_length_lower = 0.0;
    double approximate_squared_distance = 0.0;
    double current_distance_root_upper = 0.0;
    double direction_error_radius = 0.0;
    double stored_numeric_padding = 0.0;
    double operational_l2_padding = 0.0;
    double rounding_closure_padding = 0.0;
    double error_radius = 0.0;
    double lower_bound = 0.0;
    double shadow_exact_squared_distance = 0.0;
    bool would_prune = false;
    bool oracle_would_prune = false;
    bool lower_bound_valid = false;
    bool lower_bound_violation = false;
    bool false_prune = false;
};

#ifdef HNSWLIB_ENABLE_V0_APPROX_SHADOW
struct V0ShadowExpansionRecord {
    uint64_t query_id = 0;
    uint64_t current_node_id = 0;
    uint64_t expansion_index = 0;
    uint64_t current_node_degree = 0;
};

struct V0ShadowDuplicateRecord {
    uint64_t query_id = 0;
    uint64_t current_node_id = 0;
    uint64_t candidate_id = 0;
    uint64_t expansion_index = 0;
    uint64_t current_node_degree = 0;
};
#endif

// Optional sink for detailed validation. Implementations decide whether to
// stream, sample, or retain records; the search path owns no record storage.
class V0ShadowValidationCollector {
 public:
    virtual ~V0ShadowValidationCollector() {}
    virtual void append(const V0ShadowRecord& record) = 0;
#ifdef HNSWLIB_ENABLE_V0_APPROX_SHADOW
    virtual void beginQuery(uint64_t) {}
    virtual void onExpansion(const V0ShadowExpansionRecord&) {}
    virtual void onDuplicate(const V0ShadowDuplicateRecord&) {}
    virtual void endQuery(uint64_t) {}
#endif
};
#endif

}  // namespace hnswlib
