#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "edge_quant_v0_io.h"

namespace hnswlib {

// Raw Phase-1 geometry only. This type is not a lower bound and must never be
// used to make a search decision.
struct V0SphericalCapDiagnosticInput {
    double reconstruction_norm = 0.0;
    double x_norm = 0.0;
    double x_dot_reconstruction = 0.0;
    double true_edge_norm = 0.0;
    double x_dot_true_direction = 0.0;
    double geometric_squared_distance = 0.0;
    double actual_direction_error = 0.0;
    bool valid = false;
};

inline V0SphericalCapDiagnosticInput
computeV0SphericalCapDiagnosticInput(
    const float* query,
    const float* current,
    const float* candidate,
    const V0SidecarView& sidecar,
    const V0EdgeRecordView& edge) {
    V0SphericalCapDiagnosticInput result;
    if (query == NULL || current == NULL || candidate == NULL) {
        return result;
    }
    const V0SidecarHeader& header = sidecar.header();
    if (header.pq_dsub == 0U ||
        header.pq_m * header.pq_dsub != header.dimension ||
        edge.codeSize() != header.pq_m) {
        return result;
    }

    long double x_squared = 0.0L;
    long double reconstruction_squared = 0.0L;
    long double x_dot_reconstruction = 0.0L;
    long double true_edge_squared = 0.0L;
    long double x_dot_true_edge = 0.0L;
    long double geometric_squared_distance = 0.0L;
    for (uint32_t coordinate = 0U;
         coordinate < header.dimension;
         ++coordinate) {
        const uint32_t subquantizer = coordinate / header.pq_dsub;
        const uint32_t subcoordinate = coordinate % header.pq_dsub;
        const uint32_t centroid = edge.code(subquantizer);
        if (centroid >= header.pq_ksub) {
            return result;
        }
        const size_t centroid_offset =
            (static_cast<size_t>(subquantizer) * header.pq_ksub + centroid) *
                header.pq_dsub +
            subcoordinate;
        const long double reconstruction = static_cast<long double>(
            sidecar.codebookCentroid(centroid_offset));
        const long double x =
            static_cast<long double>(query[coordinate]) -
            static_cast<long double>(current[coordinate]);
        const long double true_edge =
            static_cast<long double>(candidate[coordinate]) -
            static_cast<long double>(current[coordinate]);
        const long double query_minus_candidate = x - true_edge;
        x_squared += x * x;
        reconstruction_squared += reconstruction * reconstruction;
        x_dot_reconstruction += x * reconstruction;
        true_edge_squared += true_edge * true_edge;
        x_dot_true_edge += x * true_edge;
        geometric_squared_distance +=
            query_minus_candidate * query_minus_candidate;
    }

    const long double x_norm = std::sqrt(x_squared);
    const long double reconstruction_norm =
        std::sqrt(reconstruction_squared);
    const long double true_edge_norm = std::sqrt(true_edge_squared);
    if (!(true_edge_norm > 0.0L)) {
        return result;
    }
    // Compute ||u_true-r|| directly instead of using 1+s^2-2<u,r>;
    // the latter loses accuracy when the reconstruction is already close.
    long double error_squared = 0.0L;
    for (uint32_t coordinate = 0U;
         coordinate < header.dimension;
         ++coordinate) {
        const uint32_t subquantizer = coordinate / header.pq_dsub;
        const uint32_t subcoordinate = coordinate % header.pq_dsub;
        const uint32_t centroid = edge.code(subquantizer);
        const size_t centroid_offset =
            (static_cast<size_t>(subquantizer) * header.pq_ksub + centroid) *
                header.pq_dsub +
            subcoordinate;
        const long double reconstruction = static_cast<long double>(
            sidecar.codebookCentroid(centroid_offset));
        const long double true_direction =
            (static_cast<long double>(candidate[coordinate]) -
             static_cast<long double>(current[coordinate])) /
            true_edge_norm;
        const long double difference = true_direction - reconstruction;
        error_squared += difference * difference;
    }
    const long double actual_error = std::sqrt(error_squared);

    result.reconstruction_norm = static_cast<double>(reconstruction_norm);
    result.x_norm = static_cast<double>(x_norm);
    result.x_dot_reconstruction =
        static_cast<double>(x_dot_reconstruction);
    result.true_edge_norm = static_cast<double>(true_edge_norm);
    result.x_dot_true_direction =
        static_cast<double>(x_dot_true_edge / true_edge_norm);
    result.geometric_squared_distance =
        static_cast<double>(geometric_squared_distance);
    result.actual_direction_error = static_cast<double>(actual_error);
    result.valid =
        std::isfinite(result.reconstruction_norm) &&
        std::isfinite(result.x_norm) &&
        std::isfinite(result.x_dot_reconstruction) &&
        std::isfinite(result.true_edge_norm) &&
        std::isfinite(result.x_dot_true_direction) &&
        std::isfinite(result.geometric_squared_distance) &&
        std::isfinite(result.actual_direction_error);
    return result;
}

}  // namespace hnswlib
