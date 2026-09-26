#pragma once

#include "policy.h"

namespace hnswlib {
namespace edge_estimation {

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
