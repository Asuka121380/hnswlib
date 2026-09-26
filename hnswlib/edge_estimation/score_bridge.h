#pragma once

#include <cmath>

#include "types.h"

namespace hnswlib {
namespace edge_estimation {

inline EdgeScore bridgeDotToSquaredDistance(
    double d_current,
    double edge_length,
    const DotEstimate& estimate) {
    if (!estimate.valid()) {
        return EdgeScore(0.0, estimate.status);
    }
    if (edge_length == 0.0) {
        return EdgeScore(0.0, EstimateStatus::ZeroLength);
    }
    if (!std::isfinite(d_current) || !std::isfinite(edge_length) ||
        !std::isfinite(estimate.value) || edge_length < 0.0) {
        return EdgeScore(0.0, EstimateStatus::NonFinite);
    }
    const double value = d_current + edge_length * edge_length -
        2.0 * edge_length * estimate.value;
    if (!std::isfinite(value)) {
        return EdgeScore(0.0, EstimateStatus::NumericOverflow);
    }
    return EdgeScore(value, EstimateStatus::Valid);
}

}  // namespace edge_estimation
}  // namespace hnswlib
