#pragma once

#include <cstdint>

#include "policy.h"

namespace hnswlib {
namespace edge_estimation {

struct ActivePruneRequest {
    uint64_t query_id;
    uint32_t source_id;
    uint32_t target_id;
    uint32_t neighbor_slot;
    uint32_t source_degree;
    double d_current;
    double threshold;
    bool threshold_valid;

    ActivePruneRequest()
        : query_id(0U), source_id(kInvalidNodeId), target_id(kInvalidNodeId),
          neighbor_slot(kInvalidNodeId), source_degree(0U), d_current(0.0),
          threshold(0.0), threshold_valid(false) {}
};

// Type-erased active-search boundary used by optional unified estimators.
// Implementations must fail open: an invalid estimate returns fallback=true
// and prune=false so the exact HNSW distance path remains authoritative.
class ActiveEdgePruner {
 public:
    virtual ~ActiveEdgePruner() {}
    virtual PolicyDecision evaluate(const ActivePruneRequest& request) = 0;
};

inline ActiveEdgePruner*& activeEdgePrunerSlot() {
    static thread_local ActiveEdgePruner* pruner = NULL;
    return pruner;
}

inline ActiveEdgePruner* activeEdgePruner() {
    return activeEdgePrunerSlot();
}

class ScopedActiveEdgePruner {
 public:
    explicit ScopedActiveEdgePruner(ActiveEdgePruner* pruner)
        : previous_(activeEdgePrunerSlot()) {
        activeEdgePrunerSlot() = pruner;
    }
    ~ScopedActiveEdgePruner() { activeEdgePrunerSlot() = previous_; }

 private:
    ScopedActiveEdgePruner(const ScopedActiveEdgePruner&);
    ScopedActiveEdgePruner& operator=(const ScopedActiveEdgePruner&);
    ActiveEdgePruner* previous_;
};

// Shared active-search boundary.  HNSW traversal remains exact whenever a
// backend cannot produce a valid score; a no-prune policy is a parity oracle.
class NoPrunePolicy {
 public:
    PolicyDecision decide(const EdgeScore&, double, bool) const {
        return PolicyDecision{false, false};
    }
};

template <typename Policy>
inline PolicyDecision decideActiveEdge(
    const Policy& policy,
    const EdgeScore& score,
    double threshold,
    bool threshold_valid) {
    if (!score.valid()) return PolicyDecision{false, true};
    return policy.decide(score, threshold, threshold_valid);
}

}  // namespace edge_estimation
}  // namespace hnswlib
