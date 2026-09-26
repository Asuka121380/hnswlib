#pragma once

#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
#error "pq_legacy.h requires HNSWLIB_ENABLE_EDGE_QUANT_V0"
#endif

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_estimation/types.h"

namespace uq {

inline hnswlib::edge_estimation::EstimateStatus legacyStatus(
    hnswlib::V0BoundStatus status) {
    using hnswlib::edge_estimation::EstimateStatus;
    switch (status) {
        case hnswlib::V0BoundStatus::Valid: return EstimateStatus::Valid;
        case hnswlib::V0BoundStatus::ZeroLength: return EstimateStatus::ZeroLength;
        case hnswlib::V0BoundStatus::ExactOnly: return EstimateStatus::LegacyExactOnly;
        case hnswlib::V0BoundStatus::NumericFailure: return EstimateStatus::NonFinite;
        default: return EstimateStatus::UnsupportedRecord;
    }
}

// Thin score-level adapter: expression order and float LUT accumulation remain
// entirely inside the production raw_fast_v1 implementation.
class PqLegacyScoreKernel {
 public:
    explicit PqLegacyScoreKernel(
        const hnswlib::EdgeQuantV0ApproxQueryContext& query)
        : query_(query) {}

    hnswlib::edge_estimation::EdgeScore score(
        const hnswlib::V0EdgeRecordView& record,
        double d_current) const {
        const hnswlib::V0RawEstimateResult raw =
            query_.evaluateRawFast(record, d_current);
        return hnswlib::edge_estimation::EdgeScore(
            raw.approximate_squared_distance, legacyStatus(raw.status));
    }

 private:
    const hnswlib::EdgeQuantV0ApproxQueryContext& query_;
};

}  // namespace uq
