#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hnswlib/edge_quant_v0_numeric.h"
#include "v0_sidecar_test_utils.h"

namespace {

double squaredDistance(
    const float* left,
    const float* right,
    size_t dimension) {
    double result = 0.0;
    for (size_t coordinate = 0U;
         coordinate < dimension;
         ++coordinate) {
        const double difference =
            static_cast<double>(left[coordinate]) -
            static_cast<double>(right[coordinate]);
        result += difference * difference;
    }
    return result;
}

hnswlib::V0EdgeRecord makeRecord(
    const std::vector<uint8_t>& code,
    const float* source,
    const float* target,
    const float* reconstruction,
    uint8_t flags) {
    const hnswlib::V0EdgeNumericMetadata numeric =
        hnswlib::computeV0EdgeNumericMetadata(
            source, target, reconstruction, 4U);
    hnswlib::V0EdgeRecord record;
    record.code = code;
    record.flags = flags;
    record.edge_length = numeric.edge_length;
    record.direction_error = numeric.direction_error;
    record.anchor_projection = numeric.anchor_projection;
    record.numeric_padding = numeric.numeric_padding;
    return record;
}

void writeBoundSidecar(
    const std::string& path,
    const float* source0,
    const float* target0,
    const float* source1,
    const float* target1) {
    hnswlib::V0SidecarWriteSpec spec;
    spec.dimension = 4U;
    spec.pq_m = 2U;
    spec.pq_nbits = 1U;
    spec.pq_ksub = 2U;
    spec.pq_dsub = 2U;
    spec.node_count = 1U;
    spec.directed_edge_count = 5U;
    spec.training_metadata_json =
        "{\"purpose\":\"milestone8_bound_test\"}";
    const float codebook[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
        0.5f, 0.5f
    };
    spec.codebook_centroids.assign(
        codebook, codebook + 8U);
    spec.node_offsets.push_back(0U);
    spec.node_offsets.push_back(5U);
    spec.base_index_sha256 = v0_test::digestOf("bound-index");
    spec.adjacency_sha256 =
        v0_test::digestOf("bound-adjacency");

    const float reconstruction0[] = {
        1.0f, 0.0f, 0.0f, 0.0f
    };
    const float reconstruction1[] = {
        0.0f, 0.0f, 0.5f, 0.5f
    };
    const std::vector<uint8_t> code0 = {1U, 0U};
    const std::vector<uint8_t> code1 = {0U, 1U};

    const hnswlib::V0EdgeRecord valid_exact =
        makeRecord(
            code0,
            source0,
            target0,
            reconstruction0,
            0U);
    const hnswlib::V0EdgeRecord valid_imperfect =
        makeRecord(
            code1,
            source1,
            target1,
            reconstruction1,
            0U);
    hnswlib::V0EdgeRecord more_conservative =
        valid_imperfect;
    more_conservative.direction_error += 0.25;
    more_conservative.numeric_padding = 0.125;
    hnswlib::V0EdgeRecord exact_only = valid_exact;
    exact_only.flags = hnswlib::V0_EDGE_EXACT_ONLY;
    hnswlib::V0EdgeRecord zero_length;
    zero_length.code = code0;
    zero_length.flags =
        hnswlib::V0_EDGE_EXACT_ONLY |
        hnswlib::V0_ZERO_LENGTH_EDGE;

    hnswlib::V0SidecarWriter writer(path, spec);
    writer.writeEdgeRecord(valid_exact);
    writer.writeEdgeRecord(valid_imperfect);
    writer.writeEdgeRecord(more_conservative);
    writer.writeEdgeRecord(exact_only);
    writer.writeEdgeRecord(zero_length);
    writer.finalize();
}

void testBoundEvaluator() {
    const std::string path =
        "v0_bound_evaluator_test.v0meta";
    v0_test::FileCleanup cleanup(path);

    const float source0[] = {
        1.0f, -2.0f, 0.5f, 3.0f
    };
    const float target0[] = {
        3.0f, -2.0f, 0.5f, 3.0f
    };
    const float source1[] = {
        -1.0f, 0.5f, 2.0f, -3.0f
    };
    const float target1[] = {
        -1.0f, 2.0f, 4.0f, -3.0f
    };
    writeBoundSidecar(
        path, source0, target0, source1, target1);

    hnswlib::V0OwnedSidecar owned =
        hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView sidecar = owned.view();
    const float query[] = {
        4.0f, -2.0f, 0.5f, 3.0f
    };
    const hnswlib::EdgeQuantV0QueryContext context(
        query, sidecar);

    const double source0_distance =
        squaredDistance(query, source0, 4U);
    const double target0_distance =
        squaredDistance(query, target0, 4U);
    const hnswlib::V0BoundResult exact_result =
        context.evaluate(
            sidecar.edgeRecord(0U), source0_distance);
    v0_test::require(
        exact_result.valid(),
        "exact-reconstruction edge did not produce a bound");
    v0_test::require(
        exact_result.lower_bound <= target0_distance,
        "exact-reconstruction lower bound exceeds exact distance");
    v0_test::require(
        exact_result.approximate_squared_distance <=
            target0_distance &&
        target0_distance -
            exact_result.approximate_squared_distance < 1e-4,
        "upward LUT rounding produced an invalid distance estimate");
    v0_test::require(
        exact_result.provesFartherThan(0.5),
        "valid bound did not prove a smaller threshold");
    v0_test::require(
        !exact_result.provesFartherThan(
            target0_distance),
        "strict bound incorrectly proved the exact distance threshold");

    const double source1_distance =
        squaredDistance(query, source1, 4U);
    const double target1_distance =
        squaredDistance(query, target1, 4U);
    const hnswlib::V0BoundResult imperfect_result =
        context.evaluate(
            sidecar.edgeRecord(1U), source1_distance);
    v0_test::require(
        imperfect_result.valid(),
        "imperfect-reconstruction edge did not produce a bound");
    v0_test::require(
        imperfect_result.error_radius > 0.0,
        "imperfect reconstruction has no error radius");
    v0_test::require(
        imperfect_result.lower_bound <= target1_distance,
        "imperfect-reconstruction lower bound exceeds exact distance");

    const hnswlib::V0BoundResult more_conservative =
        context.evaluate(
            sidecar.edgeRecord(2U), source1_distance);
    v0_test::require(
        more_conservative.valid() &&
        more_conservative.error_radius >
            imperfect_result.error_radius &&
        more_conservative.lower_bound <=
            imperfect_result.lower_bound,
        "larger error metadata did not make the bound conservative");

    const hnswlib::V0BoundResult exact_only =
        context.evaluate(
            sidecar.edgeRecord(3U), source0_distance);
    v0_test::require(
        exact_only.status == hnswlib::V0BoundStatus::ExactOnly &&
        exact_only.requiresExactFallback(),
        "exact-only edge did not require fallback");

    const hnswlib::V0BoundResult zero_length =
        context.evaluate(
            sidecar.edgeRecord(4U), source0_distance);
    v0_test::require(
        zero_length.status ==
            hnswlib::V0BoundStatus::ZeroLength &&
        zero_length.requiresExactFallback(),
        "zero-length edge did not require fallback");

    const hnswlib::V0BoundResult invalid_distance =
        context.evaluate(
            sidecar.edgeRecord(0U),
            std::numeric_limits<double>::infinity());
    v0_test::require(
        invalid_distance.status ==
            hnswlib::V0BoundStatus::InvalidCurrentDistance &&
        invalid_distance.requiresExactFallback(),
        "non-finite current distance did not require fallback");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    return 2;
#else
    testBoundEvaluator();
    return 0;
#endif
}
