#pragma once

#include <cmath>

#include "types.h"

namespace hnswlib {
namespace edge_estimation {

// IVF is a representation composition, not a second evaluator.  The coarse
// and residual terms are computed by their own frozen models and combined here.
inline DotEstimate composeIvfResidualDot(
    const DotEstimate& coarse,
    const DotEstimate& residual) {
    if (!coarse.valid()) return coarse;
    if (!residual.valid()) return residual;
    const double value = coarse.value + residual.value;
    return std::isfinite(value)
        ? DotEstimate(value, EstimateStatus::Valid)
        : DotEstimate(0.0, EstimateStatus::NumericOverflow);
}

}  // namespace edge_estimation
}  // namespace hnswlib
