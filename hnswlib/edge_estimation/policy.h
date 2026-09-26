#pragma once

#include <cmath>
#include <stdexcept>

#include "types.h"

namespace hnswlib {
namespace edge_estimation {

struct PolicyDecision {
    bool prune;
    bool fallback;
};

class ScaledThresholdPolicy {
 public:
    explicit ScaledThresholdPolicy(double alpha) : alpha_(alpha) {
        if (!std::isfinite(alpha_) || alpha_ < 0.0) {
            throw std::invalid_argument("alpha must be finite and non-negative");
        }
    }

    PolicyDecision decide(
        const EdgeScore& score,
        double threshold,
        bool threshold_valid) const {
        if (!threshold_valid || !std::isfinite(threshold) || threshold < 0.0 ||
            !score.valid() || !std::isfinite(score.squared_distance)) {
            return PolicyDecision{false, true};
        }
        return PolicyDecision{
            score.squared_distance > alpha_ * threshold,
            false};
    }

 private:
    double alpha_;
};

}  // namespace edge_estimation
}  // namespace hnswlib
