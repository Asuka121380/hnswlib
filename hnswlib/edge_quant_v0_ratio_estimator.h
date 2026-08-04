#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "edge_quant_v0_io.h"

namespace hnswlib {

static const char* const V0_RATIO_ESTIMATOR_FORMULA_VERSION =
    "v0_ratio_corrected_pq_distance_v1";

enum class V0RatioEstimatorStatus : uint8_t {
    Valid = 0U,
    ExactOnly = 1U,
    ZeroLength = 2U,
    InvalidCurrentDistance = 3U,
    InvalidEdgeMetadata = 4U,
    NonPositiveReconstructionNorm = 5U,
    NonPositiveDenominator = 6U,
    KappaBelowMinimum = 7U,
    NumericFailure = 8U
};

inline const char* v0RatioFallbackReason(V0RatioEstimatorStatus status) {
    switch (status) {
        case V0RatioEstimatorStatus::Valid: return "none";
        case V0RatioEstimatorStatus::ExactOnly: return "edge_exact_only";
        case V0RatioEstimatorStatus::ZeroLength: return "zero_length_edge";
        case V0RatioEstimatorStatus::InvalidCurrentDistance:
            return "invalid_current_distance";
        case V0RatioEstimatorStatus::InvalidEdgeMetadata:
            return "invalid_edge_metadata";
        case V0RatioEstimatorStatus::NonPositiveReconstructionNorm:
            return "non_positive_reconstruction_norm";
        case V0RatioEstimatorStatus::NonPositiveDenominator:
            return "non_positive_denominator";
        case V0RatioEstimatorStatus::KappaBelowMinimum:
            return "kappa_below_minimum";
        case V0RatioEstimatorStatus::NumericFailure:
            return "numeric_failure";
    }
    return "unknown";
}

struct V0RatioEstimate {
    V0RatioEstimatorStatus status =
        V0RatioEstimatorStatus::InvalidEdgeMetadata;
    double current_distance_norm = 0.0;
    double edge_length = 0.0;
    double direction_error = 0.0;
    double reconstruction_norm_squared = 0.0;
    double reconstruction_norm = 0.0;
    double query_dot_reconstruction = 0.0;
    double current_dot_reconstruction = 0.0;
    double x_dot_reconstruction = 0.0;
    double denominator_base = 0.0;
    double kappa_meta = 0.0;
    double rho_hat_raw = 0.0;
    double rho_hat_ratio = 0.0;
    double rho_hat_ratio_clipped = 0.0;
    double estimated_squared_distance = 0.0;

    bool eligible() const {
        return status == V0RatioEstimatorStatus::Valid;
    }
};

// Query-independent codeword-norm table. It is constructed once per loaded
// sidecar and adds no per-edge storage.
class V0RatioCodebookNormLut {
 public:
    explicit V0RatioCodebookNormLut(const V0SidecarView& sidecar)
        : pq_m_(sidecar.header().pq_m),
          pq_ksub_(sidecar.header().pq_ksub),
          pq_dsub_(sidecar.header().pq_dsub) {
        validateLayout(sidecar.header());
        values_.resize(
            static_cast<size_t>(pq_m_) * pq_ksub_, 0.0);
        for (uint32_t m = 0U; m < pq_m_; ++m) {
            for (uint32_t centroid = 0U;
                 centroid < pq_ksub_;
                 ++centroid) {
                double norm_squared = 0.0;
                for (uint32_t d = 0U; d < pq_dsub_; ++d) {
                    const size_t offset =
                        (static_cast<size_t>(m) * pq_ksub_ + centroid) *
                            pq_dsub_ +
                        d;
                    const double value = static_cast<double>(
                        sidecar.codebookCentroid(offset));
                    norm_squared += value * value;
                }
                if (!std::isfinite(norm_squared) || norm_squared < 0.0) {
                    throw std::runtime_error(
                        "V0 ratio codeword norm is invalid");
                }
                values_[index(m, centroid)] = norm_squared;
            }
        }
    }

    double codewordNormSquared(size_t m, size_t centroid) const {
        if (m >= pq_m_ || centroid >= pq_ksub_) {
            throw std::out_of_range(
                "V0 ratio norm LUT index is out of range");
        }
        return values_[index(
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(centroid))];
    }

    double reconstructionNormSquared(
        const V0EdgeRecordView& edge) const {
        if (edge.codeSize() != pq_m_) {
            throw std::runtime_error(
                "V0 ratio edge code size does not match norm LUT");
        }
        double total = 0.0;
        for (uint32_t m = 0U; m < pq_m_; ++m) {
            const uint32_t centroid = edge.code(m);
            if (centroid >= pq_ksub_) {
                throw std::runtime_error(
                    "V0 ratio edge code is outside norm LUT");
            }
            total += values_[index(m, centroid)];
        }
        if (!std::isfinite(total) || total < 0.0) {
            throw std::runtime_error(
                "V0 ratio reconstruction norm overflowed");
        }
        return total;
    }

    size_t tableBytes() const {
        return values_.size() * sizeof(double);
    }

 private:
    static void validateLayout(const V0SidecarHeader& header) {
        if (header.dimension == 0U || header.pq_m == 0U ||
            header.pq_ksub == 0U || header.pq_dsub == 0U ||
            header.pq_code_size != header.pq_m ||
            static_cast<uint64_t>(header.pq_m) * header.pq_dsub !=
                header.dimension) {
            throw std::runtime_error(
                "V0 ratio norm LUT received an invalid PQ layout");
        }
    }

    size_t index(uint32_t m, uint32_t centroid) const {
        return static_cast<size_t>(m) * pq_ksub_ + centroid;
    }

    uint32_t pq_m_;
    uint32_t pq_ksub_;
    uint32_t pq_dsub_;
    std::vector<double> values_;
};

class V0RatioQueryDotLut {
 public:
    V0RatioQueryDotLut(
        const float* query,
        const V0SidecarView& sidecar)
        : pq_m_(sidecar.header().pq_m),
          pq_ksub_(sidecar.header().pq_ksub),
          pq_dsub_(sidecar.header().pq_dsub) {
        if (query == NULL) {
            throw std::invalid_argument(
                "V0 ratio query LUT requires a query vector");
        }
        values_.resize(
            static_cast<size_t>(pq_m_) * pq_ksub_, 0.0);
        for (uint32_t m = 0U; m < pq_m_; ++m) {
            const size_t query_offset = static_cast<size_t>(m) * pq_dsub_;
            for (uint32_t centroid = 0U;
                 centroid < pq_ksub_;
                 ++centroid) {
                double dot = 0.0;
                for (uint32_t d = 0U; d < pq_dsub_; ++d) {
                    const double q = static_cast<double>(
                        query[query_offset + d]);
                    const size_t codebook_offset =
                        (static_cast<size_t>(m) * pq_ksub_ + centroid) *
                            pq_dsub_ +
                        d;
                    const double r = static_cast<double>(
                        sidecar.codebookCentroid(codebook_offset));
                    dot += q * r;
                }
                if (!std::isfinite(dot)) {
                    throw std::runtime_error(
                        "V0 ratio query dot-product overflowed");
                }
                values_[index(m, centroid)] = dot;
            }
        }
    }

    double innerProduct(const V0EdgeRecordView& edge) const {
        if (edge.codeSize() != pq_m_) {
            throw std::runtime_error(
                "V0 ratio edge code size does not match query LUT");
        }
        double total = 0.0;
        for (uint32_t m = 0U; m < pq_m_; ++m) {
            const uint32_t centroid = edge.code(m);
            if (centroid >= pq_ksub_) {
                throw std::runtime_error(
                    "V0 ratio edge code is outside query LUT");
            }
            total += values_[index(m, centroid)];
        }
        if (!std::isfinite(total)) {
            throw std::runtime_error(
                "V0 ratio query lookup overflowed");
        }
        return total;
    }

    size_t tableBytes() const {
        return values_.size() * sizeof(double);
    }

 private:
    size_t index(uint32_t m, uint32_t centroid) const {
        return static_cast<size_t>(m) * pq_ksub_ + centroid;
    }

    uint32_t pq_m_;
    uint32_t pq_ksub_;
    uint32_t pq_dsub_;
    std::vector<double> values_;
};

class V0RatioEstimatorQueryContext {
 public:
    V0RatioEstimatorQueryContext(
        const float* query,
        const V0SidecarView& sidecar,
        const V0RatioCodebookNormLut& norm_lut)
        : query_lut_(query, sidecar), norm_lut_(norm_lut) {}

    V0RatioEstimate evaluate(
        const V0EdgeRecordView& edge,
        double current_squared_distance,
        double kappa_min) const {
        V0RatioEstimate result;
        if (!std::isfinite(current_squared_distance) ||
            current_squared_distance < 0.0 ||
            !std::isfinite(kappa_min) || kappa_min <= 0.0) {
            result.status =
                V0RatioEstimatorStatus::InvalidCurrentDistance;
            return result;
        }
        const uint8_t flags = edge.flags();
        if ((flags & V0_ZERO_LENGTH_EDGE) != 0U) {
            result.status = V0RatioEstimatorStatus::ZeroLength;
            return result;
        }
        if ((flags & (V0_EDGE_EXACT_ONLY | V0_RESERVED_INVALID)) != 0U) {
            result.status = V0RatioEstimatorStatus::ExactOnly;
            return result;
        }

        result.edge_length = edge.edgeLength();
        result.direction_error = edge.directionError();
        result.current_dot_reconstruction = edge.anchorProjection();
        if (!std::isfinite(result.edge_length) ||
            result.edge_length <= 0.0 ||
            !std::isfinite(result.direction_error) ||
            result.direction_error < 0.0 ||
            !std::isfinite(result.current_dot_reconstruction)) {
            result.status =
                V0RatioEstimatorStatus::InvalidEdgeMetadata;
            return result;
        }

        try {
            result.reconstruction_norm_squared =
                norm_lut_.reconstructionNormSquared(edge);
            result.query_dot_reconstruction =
                query_lut_.innerProduct(edge);
        } catch (const std::exception&) {
            result.status =
                V0RatioEstimatorStatus::InvalidEdgeMetadata;
            return result;
        }
        result.reconstruction_norm =
            std::sqrt(result.reconstruction_norm_squared);
        result.current_distance_norm =
            std::sqrt(current_squared_distance);
        if (!(result.reconstruction_norm > 0.0)) {
            result.status = V0RatioEstimatorStatus::
                NonPositiveReconstructionNorm;
            return result;
        }
        if (!(result.current_distance_norm > 0.0)) {
            result.status =
                V0RatioEstimatorStatus::InvalidCurrentDistance;
            return result;
        }

        result.x_dot_reconstruction =
            result.query_dot_reconstruction -
            result.current_dot_reconstruction;
        result.denominator_base =
            1.0 + result.reconstruction_norm_squared -
            result.direction_error * result.direction_error;
        if (!std::isfinite(result.denominator_base) ||
            !(result.denominator_base > 0.0)) {
            result.status =
                V0RatioEstimatorStatus::NonPositiveDenominator;
            return result;
        }
        result.kappa_meta = result.denominator_base /
            (2.0 * result.reconstruction_norm);
        if (!std::isfinite(result.kappa_meta)) {
            result.status = V0RatioEstimatorStatus::NumericFailure;
            return result;
        }
        if (result.kappa_meta < kappa_min) {
            result.status =
                V0RatioEstimatorStatus::KappaBelowMinimum;
            return result;
        }

        result.rho_hat_raw = result.x_dot_reconstruction /
            (result.current_distance_norm *
             result.reconstruction_norm);
        result.rho_hat_ratio = result.rho_hat_raw /
            result.kappa_meta;
        result.rho_hat_ratio_clipped = std::max(
            -1.0, std::min(1.0, result.rho_hat_ratio));
        result.estimated_squared_distance =
            current_squared_distance +
            result.edge_length * result.edge_length -
            2.0 * result.current_distance_norm *
                result.edge_length * result.rho_hat_ratio_clipped;
        if (!std::isfinite(result.x_dot_reconstruction) ||
            !std::isfinite(result.rho_hat_raw) ||
            !std::isfinite(result.rho_hat_ratio) ||
            !std::isfinite(result.rho_hat_ratio_clipped) ||
            !std::isfinite(result.estimated_squared_distance)) {
            result.status = V0RatioEstimatorStatus::NumericFailure;
            return result;
        }
        result.status = V0RatioEstimatorStatus::Valid;
        return result;
    }

 private:
    V0RatioQueryDotLut query_lut_;
    const V0RatioCodebookNormLut& norm_lut_;
};

}  // namespace hnswlib
