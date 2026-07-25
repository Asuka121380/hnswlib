#pragma once

#include <cstdint>

namespace hnswlib {

struct V0QueryMetrics {
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

}  // namespace hnswlib
