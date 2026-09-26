#pragma once

#if !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) || !defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
#error "pq_qjl_legacy.h requires V0 PQ and residual estimator support"
#endif

#include <cmath>

#include "pq_legacy.h"
#include "hnswlib/edge_quant_v0_residual.h"

namespace uq {

// The companion reader supplies packed signs, float32 scale, and float32
// offset. This adapter deliberately invokes the existing correction method
// instead of translating the correction through a dot-product abstraction.
class PqQjlLegacyScoreKernel {
 public:
    PqQjlLegacyScoreKernel(
        const hnswlib::EdgeQuantV0ApproxQueryContext& pq_query,
        const hnswlib::V0ResidualQueryContext& residual_query)
        : pq_query_(pq_query), residual_query_(residual_query) {}

    hnswlib::edge_estimation::EdgeScore score(
        const hnswlib::V0EdgeRecordView& record,
        double d_current,
        const uint8_t* packed_signs,
        float scale,
        float offset) const {
        const hnswlib::V0RawEstimateResult raw =
            pq_query_.evaluateRawFast(record, d_current);
        const hnswlib::edge_estimation::EstimateStatus status =
            legacyStatus(raw.status);
        if (status != hnswlib::edge_estimation::EstimateStatus::Valid)
            return hnswlib::edge_estimation::EdgeScore(0.0, status);
        const double corrected = residual_query_.correct(
            raw.approximate_squared_distance, packed_signs, scale, offset);
        if (!std::isfinite(corrected))
            return hnswlib::edge_estimation::EdgeScore(
                0.0, hnswlib::edge_estimation::EstimateStatus::NonFinite);
        return hnswlib::edge_estimation::EdgeScore(
            corrected, hnswlib::edge_estimation::EstimateStatus::Valid);
    }

 private:
    const hnswlib::EdgeQuantV0ApproxQueryContext& pq_query_;
    const hnswlib::V0ResidualQueryContext& residual_query_;
};

}  // namespace uq
