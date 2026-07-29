#pragma once

#include <cstdint>

namespace hnswlib {

struct V0QueryMetrics {
    // In observe-only searches, bound_pruned counts decisions that would
    // prune. In a real-pruning search it counts decisions actually taken,
    // and therefore equals exact_distance_saved.
    uint64_t bound_evaluated = 0;
    uint64_t bound_pruned = 0;
    uint64_t exact_fallback = 0;
    uint64_t exact_only_fallback = 0;
    uint64_t exact_distance_saved = 0;
    uint64_t lower_bound_violation = 0;
    uint64_t false_prune = 0;

    void reset() {
        *this = V0QueryMetrics();
    }
};

#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
struct V0ShadowRecord {
    uint64_t query_id = 0;
    uint64_t current_node_id = 0;
    uint64_t candidate_id = 0;
    uint8_t bound_status = 0;
    double current_squared_distance = 0.0;
    double threshold = 0.0;
    double approximate_squared_distance = 0.0;
    double error_radius = 0.0;
    double lower_bound = 0.0;
    double shadow_exact_squared_distance = 0.0;
    bool would_prune = false;
    bool lower_bound_valid = false;
    bool lower_bound_violation = false;
    bool false_prune = false;
};

// Optional sink for detailed validation. Implementations decide whether to
// stream, sample, or retain records; the search path owns no record storage.
class V0ShadowValidationCollector {
 public:
    virtual ~V0ShadowValidationCollector() {}
    virtual void append(const V0ShadowRecord& record) = 0;
};
#endif

}  // namespace hnswlib
