#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "edge_quant_v0_graph_access.h"
#include "edge_quant_v0_io.h"
#include "edge_quant_v0_sampler.h"

namespace hnswlib {

namespace edge_quant_v0_query_detail {

inline double nextUp(double value) {
    return std::nextafter(
        value, std::numeric_limits<double>::infinity());
}

inline double nextDown(double value) {
    return std::nextafter(
        value, -std::numeric_limits<double>::infinity());
}

inline double addUp(double left, double right) {
    return nextUp(left + right);
}

inline double addDown(double left, double right) {
    return nextDown(left + right);
}

inline double multiplyUp(double left, double right) {
    return nextUp(left * right);
}

inline double multiplyDown(double left, double right) {
    return nextDown(left * right);
}

// hnswlib's operational squared-L2 distance is accumulated in float32.
// A mathematical lower bound must therefore also cover the possibility that
// the operational result rounded below the real squared distance. The
// standard gamma_n model is applied with a deliberately conservative
// operation budget for both the current and target distance computations.
inline double floatSquaredL2PaddingUpper(
    double observed_current_squared_distance,
    double edge_length,
    uint32_t dimension) {
    const double operation_count =
        8.0 * static_cast<double>(dimension) + 64.0;
    const double scaled_epsilon =
        operation_count *
        static_cast<double>(
            std::numeric_limits<float>::epsilon());
    if (!std::isfinite(scaled_epsilon) ||
        scaled_epsilon >= 1.0) {
        throw std::runtime_error(
            "V0 float32 L2 rounding model is invalid");
    }
    const double gamma =
        nextUp(scaled_epsilon / (1.0 - scaled_epsilon));
    const double current_upper =
        nextUp(
            observed_current_squared_distance /
            (1.0 - gamma));
    const double target_norm_upper =
        addUp(
            nextUp(std::sqrt(current_upper)),
            nextUp(edge_length));
    const double target_squared_upper =
        multiplyUp(target_norm_upper, target_norm_upper);
    return multiplyUp(gamma, target_squared_upper);
}

inline float castFloatUp(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "V0 query LUT contains a non-finite dot product");
    }
    float result = static_cast<float>(value);
    if (!std::isfinite(static_cast<double>(result))) {
        throw std::runtime_error(
            "V0 query LUT dot product is not representable as float32");
    }
    if (static_cast<double>(result) < value) {
        result = std::nextafter(
            result, std::numeric_limits<float>::infinity());
    }
    if (!std::isfinite(static_cast<double>(result)) ||
        static_cast<double>(result) < value) {
        throw std::runtime_error(
            "V0 query LUT could not round a dot product upward");
    }
    return result;
}

}  // namespace edge_quant_v0_query_detail

// Query-global inner-product table. Each cell is rounded toward +infinity,
// so summing the cells selected by an edge code gives an upper bound for
// q^T u_hat. This is the direction needed by the squared-L2 lower bound.
class V0QueryLut {
 public:
    V0QueryLut(
        const float* query,
        const V0SidecarView& sidecar)
        : dimension_(sidecar.header().dimension),
          pq_m_(sidecar.header().pq_m),
          pq_ksub_(sidecar.header().pq_ksub),
          pq_dsub_(sidecar.header().pq_dsub) {
        validateLayout(sidecar.header());
        if (query == NULL) {
            throw std::invalid_argument(
                "V0 query LUT requires a query vector");
        }
        for (uint32_t d = 0U; d < dimension_; ++d) {
            if (!std::isfinite(static_cast<double>(query[d]))) {
                throw std::invalid_argument(
                    "V0 query LUT received a non-finite query");
            }
        }

        const size_t table_size =
            static_cast<size_t>(pq_m_) *
            static_cast<size_t>(pq_ksub_);
        table_.resize(table_size);
        for (uint32_t subquantizer = 0U;
             subquantizer < pq_m_;
             ++subquantizer) {
            const size_t query_offset =
                static_cast<size_t>(subquantizer) * pq_dsub_;
            for (uint32_t centroid = 0U;
                 centroid < pq_ksub_;
                 ++centroid) {
                const size_t centroid_offset =
                    (static_cast<size_t>(subquantizer) * pq_ksub_ +
                     centroid) *
                    pq_dsub_;
                double dot_upper = 0.0;
                for (uint32_t coordinate = 0U;
                     coordinate < pq_dsub_;
                     ++coordinate) {
                    const double product =
                        static_cast<double>(
                            query[query_offset + coordinate]) *
                        static_cast<double>(
                            sidecar.codebookCentroid(
                                centroid_offset + coordinate));
                    if (!std::isfinite(product)) {
                        throw std::runtime_error(
                            "V0 query LUT dot product overflowed");
                    }
                    dot_upper =
                        edge_quant_v0_query_detail::addUp(
                            dot_upper, product);
                    if (!std::isfinite(dot_upper)) {
                        throw std::runtime_error(
                            "V0 query LUT accumulation overflowed");
                    }
                }
                table_[tableIndex(subquantizer, centroid)] =
                    edge_quant_v0_query_detail::castFloatUp(
                        dot_upper);
            }
        }
    }

    uint32_t dimension() const {
        return dimension_;
    }

    uint32_t subquantizerCount() const {
        return pq_m_;
    }

    uint32_t centroidCountPerSubquantizer() const {
        return pq_ksub_;
    }

    uint32_t subvectorDimension() const {
        return pq_dsub_;
    }

    size_t tableBytes() const {
        return table_.size() * sizeof(float);
    }

    float value(
        size_t subquantizer,
        size_t centroid) const {
        if (subquantizer >= pq_m_ || centroid >= pq_ksub_) {
            throw std::out_of_range(
                "V0 query LUT index is out of range");
        }
        return table_[tableIndex(
            static_cast<uint32_t>(subquantizer),
            static_cast<uint32_t>(centroid))];
    }

    double innerProductUpper(
        const V0EdgeRecordView& edge) const {
        if (edge.codeSize() != pq_m_) {
            throw std::runtime_error(
                "V0 edge code size does not match query LUT");
        }
        double result = 0.0;
        for (uint32_t subquantizer = 0U;
             subquantizer < pq_m_;
             ++subquantizer) {
            const uint32_t centroid = edge.code(subquantizer);
            if (centroid >= pq_ksub_) {
                throw std::runtime_error(
                    "V0 edge code is outside the query LUT");
            }
            result = edge_quant_v0_query_detail::addUp(
                result,
                static_cast<double>(
                    table_[tableIndex(
                        subquantizer, centroid)]));
            if (!std::isfinite(result)) {
                throw std::runtime_error(
                    "V0 query LUT lookup overflowed");
            }
        }
        return result;
    }

 private:
    void validateLayout(
        const V0SidecarHeader& header) const {
        if (header.dimension == 0U ||
            header.pq_m == 0U ||
            header.pq_ksub == 0U ||
            header.pq_ksub > 256U ||
            header.pq_dsub == 0U ||
            header.pq_code_size != header.pq_m ||
            static_cast<uint64_t>(header.pq_m) *
                static_cast<uint64_t>(header.pq_dsub) !=
                header.dimension) {
            throw std::runtime_error(
                "V0 query LUT received an invalid PQ layout");
        }
        const uint64_t table_size =
            static_cast<uint64_t>(header.pq_m) *
            static_cast<uint64_t>(header.pq_ksub);
        if (table_size >
            static_cast<uint64_t>(
                std::numeric_limits<size_t>::max() /
                sizeof(float))) {
            throw std::overflow_error(
                "V0 query LUT size overflows size_t");
        }
    }

    size_t tableIndex(
        uint32_t subquantizer,
        uint32_t centroid) const {
        return static_cast<size_t>(subquantizer) * pq_ksub_ +
            centroid;
    }

    uint32_t dimension_;
    uint32_t pq_m_;
    uint32_t pq_ksub_;
    uint32_t pq_dsub_;
    std::vector<float> table_;
};

enum class V0BoundStatus : uint8_t {
    Valid = 0U,
    ExactOnly = 1U,
    ZeroLength = 2U,
    ReservedInvalid = 3U,
    InvalidCurrentDistance = 4U,
    InvalidEdgeMetadata = 5U,
    NumericFailure = 6U
};

struct V0BoundResult {
    V0BoundStatus status;
    double edge_length;
    double direction_error;
    double anchor_projection;
    double anchor_projection_lower;
    double query_direction_inner_product_upper;
    double residual_direction_inner_product_upper;
    double length_squared_lower;
    double cross_term_upper;
    double base_plus_length_lower;
    double approximate_squared_distance;
    double current_distance_root_upper;
    double direction_error_radius;
    double stored_numeric_padding;
    double operational_l2_padding;
    double rounding_closure_padding;
    double error_radius;
    double lower_bound;

    V0BoundResult()
        : status(V0BoundStatus::InvalidEdgeMetadata),
          edge_length(0.0),
          direction_error(0.0),
          anchor_projection(0.0),
          anchor_projection_lower(0.0),
          query_direction_inner_product_upper(0.0),
          residual_direction_inner_product_upper(0.0),
          length_squared_lower(0.0),
          cross_term_upper(0.0),
          base_plus_length_lower(0.0),
          approximate_squared_distance(0.0),
          current_distance_root_upper(0.0),
          direction_error_radius(0.0),
          stored_numeric_padding(0.0),
          operational_l2_padding(0.0),
          rounding_closure_padding(0.0),
          error_radius(0.0),
          lower_bound(0.0) {}

    bool valid() const {
        return status == V0BoundStatus::Valid;
    }

    bool requiresExactFallback() const {
        return !valid();
    }

    // A mathematical decision helper only. Milestone 8 does not connect
    // this decision to HNSW control flow; shadow validation must pass before
    // a caller is allowed to skip an exact distance.
    bool provesFartherThan(double exact_threshold) const {
        return valid() &&
            std::isfinite(exact_threshold) &&
            exact_threshold >= 0.0 &&
            lower_bound > exact_threshold;
    }
};

// Per-query state only. It owns no index or sidecar storage and therefore
// remains thread-local; the caller keeps the immutable metadata alive.
// Evaluation is side-effect free and cannot prune or mutate HNSW state.
class EdgeQuantV0QueryContext {
 public:
    EdgeQuantV0QueryContext(
        const float* query,
        const V0SidecarView& sidecar)
        : lut_(query, sidecar) {}

    const V0QueryLut& lut() const {
        return lut_;
    }

    V0BoundResult evaluate(
        const V0EdgeRecordView& edge,
        double exact_current_squared_distance) const {
        V0BoundResult result;
        if (!std::isfinite(exact_current_squared_distance) ||
            exact_current_squared_distance < 0.0) {
            result.status =
                V0BoundStatus::InvalidCurrentDistance;
            return result;
        }

        const uint8_t flags = edge.flags();
        if ((flags & V0_RESERVED_INVALID) != 0U) {
            result.status = V0BoundStatus::ReservedInvalid;
            return result;
        }
        if ((flags & V0_ZERO_LENGTH_EDGE) != 0U) {
            result.status = V0BoundStatus::ZeroLength;
            return result;
        }
        if ((flags & V0_EDGE_EXACT_ONLY) != 0U) {
            result.status = V0BoundStatus::ExactOnly;
            return result;
        }

        const double length = edge.edgeLength();
        const double direction_error = edge.directionError();
        const double anchor = edge.anchorProjection();
        const double stored_padding = edge.numericPadding();
        if (!std::isfinite(length) || length <= 0.0 ||
            !std::isfinite(direction_error) ||
            direction_error < 0.0 ||
            !std::isfinite(anchor) ||
            !std::isfinite(stored_padding) ||
            stored_padding < 0.0) {
            result.status = V0BoundStatus::InvalidEdgeMetadata;
            return result;
        }
        result.edge_length = length;
        result.direction_error = direction_error;
        result.anchor_projection = anchor;
        result.stored_numeric_padding = stored_padding;

        try {
            result.query_direction_inner_product_upper =
                lut_.innerProductUpper(edge);
        } catch (const std::exception&) {
            result.status = V0BoundStatus::InvalidEdgeMetadata;
            return result;
        }

        const double anchor_lower =
            edge_quant_v0_query_detail::nextDown(anchor);
        result.anchor_projection_lower = anchor_lower;
        result.residual_direction_inner_product_upper =
            edge_quant_v0_query_detail::addUp(
                result.query_direction_inner_product_upper,
                -anchor_lower);

        const double length_squared_lower =
            edge_quant_v0_query_detail::multiplyDown(
                length, length);
        result.length_squared_lower = length_squared_lower;
        const double twice_length = 2.0 * length;
        const double cross_term_upper =
            edge_quant_v0_query_detail::multiplyUp(
                twice_length,
                result.residual_direction_inner_product_upper);
        result.cross_term_upper = cross_term_upper;
        const double base_plus_length_lower =
            edge_quant_v0_query_detail::addDown(
                exact_current_squared_distance,
                length_squared_lower);
        result.base_plus_length_lower = base_plus_length_lower;
        const double approximate_lower =
            edge_quant_v0_query_detail::addDown(
                base_plus_length_lower,
                -cross_term_upper);

        const double current_distance_root_upper =
            edge_quant_v0_query_detail::nextUp(
                std::sqrt(exact_current_squared_distance));
        result.current_distance_root_upper =
            current_distance_root_upper;
        double direction_error_radius_upper =
            edge_quant_v0_query_detail::multiplyUp(
                twice_length,
                current_distance_root_upper);
        direction_error_radius_upper =
            edge_quant_v0_query_detail::multiplyUp(
                direction_error_radius_upper,
                direction_error);
        result.direction_error_radius =
            direction_error_radius_upper;
        double error_radius_upper =
            edge_quant_v0_query_detail::addUp(
                direction_error_radius_upper,
                stored_padding);
        const double operational_l2_padding =
            edge_quant_v0_query_detail::
                floatSquaredL2PaddingUpper(
                    exact_current_squared_distance,
                    length,
                    lut_.dimension());
        result.operational_l2_padding =
            operational_l2_padding;
        error_radius_upper =
            edge_quant_v0_query_detail::addUp(
                error_radius_upper,
                operational_l2_padding);
        const long double component_sum =
            static_cast<long double>(direction_error_radius_upper) +
            static_cast<long double>(stored_padding) +
            static_cast<long double>(operational_l2_padding);
        const long double closure =
            static_cast<long double>(error_radius_upper) -
            component_sum;
        result.rounding_closure_padding =
            closure > 0.0L ? static_cast<double>(closure) : 0.0;
        const double lower_bound =
            edge_quant_v0_query_detail::addDown(
                approximate_lower,
                -error_radius_upper);

        if (!std::isfinite(
                result.residual_direction_inner_product_upper) ||
            !std::isfinite(approximate_lower) ||
            !std::isfinite(direction_error_radius_upper) ||
            direction_error_radius_upper < 0.0 ||
            !std::isfinite(operational_l2_padding) ||
            operational_l2_padding < 0.0 ||
            !std::isfinite(result.rounding_closure_padding) ||
            result.rounding_closure_padding < 0.0 ||
            !std::isfinite(error_radius_upper) ||
            error_radius_upper < 0.0 ||
            !std::isfinite(lower_bound)) {
            result.status = V0BoundStatus::NumericFailure;
            return result;
        }

        result.status = V0BoundStatus::Valid;
        result.approximate_squared_distance =
            approximate_lower;
        result.error_radius = error_radius_upper;
        result.lower_bound =
            lower_bound > 0.0 ? lower_bound : 0.0;
        return result;
    }

 private:
    V0QueryLut lut_;
};

}  // namespace hnswlib
