#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "v0_sidecar_test_utils.h"

namespace {

void requireParity(
    const hnswlib::EdgeQuantV0QueryContext& context,
    const hnswlib::V0EdgeRecordView& edge,
    double current_squared_distance) {
    const hnswlib::V0BoundResult bound =
        context.evaluate(edge, current_squared_distance);
    const hnswlib::V0RawEstimateResult raw =
        context.evaluateRaw(edge, current_squared_distance);
    v0_test::require(
        raw.status == bound.status,
        "raw and strict evaluator statuses differ");
    v0_test::require(
        raw.approximate_squared_distance ==
            bound.approximate_squared_distance,
        "raw estimate is not bit-identical to strict approximate distance");
    v0_test::require(
        raw.valid() == bound.valid() &&
            raw.requiresExactFallback() ==
                bound.requiresExactFallback(),
        "raw fallback semantics differ from strict evaluator");
}

void testRawEstimatorParity() {
    const std::string path = "v0_raw_estimator_test.v0meta";
    v0_test::FileCleanup cleanup(path);
    v0_test::writeTinySidecar(path);
    const hnswlib::V0OwnedSidecar owned =
        hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView sidecar = owned.view();
    const float query[] = {0.25f, -0.5f, 1.0f, 2.0f};
    const hnswlib::EdgeQuantV0QueryContext context(query, sidecar);

    const double distances[] = {
        0.0,
        1.0,
        19.25,
        std::numeric_limits<double>::infinity(),
        -1.0,
        std::numeric_limits<double>::quiet_NaN()
    };
    for (size_t edge = 0U; edge < 3U; ++edge) {
        for (size_t i = 0U;
             i < sizeof(distances) / sizeof(distances[0]);
             ++i) {
            requireParity(
                context,
                sidecar.edgeRecord(edge),
                distances[i]);
        }
    }
}

void testRawFastV1() {
    const std::string path = "v0_raw_fast_v1_test.v0meta";
    v0_test::FileCleanup cleanup(path);
    v0_test::writeTinySidecar(path);
    const hnswlib::V0OwnedSidecar owned =
        hnswlib::loadV0Sidecar(path);
    const hnswlib::V0SidecarView sidecar = owned.view();
    std::vector<float> native_codebook(
        static_cast<size_t>(
            sidecar.header().codebook.size / sizeof(float)));
    for (size_t i = 0U; i < native_codebook.size(); ++i) {
        native_codebook[i] = sidecar.codebookCentroid(i);
    }

    const float query[] = {0.25f, -0.5f, 1.0f, 2.0f};
    const hnswlib::EdgeQuantV0ApproxQueryContext context(
        query, sidecar.header(), native_codebook.data());
    const hnswlib::V0RawEstimateResult valid =
        context.evaluateRawFast(sidecar.edgeRecord(0U), 19.25);
    v0_test::require(valid.valid(), "raw_fast_v1 rejected valid edge");
    v0_test::require(
        std::fabs(valid.approximate_squared_distance - (-2.3125)) <
            1e-6,
        "raw_fast_v1 formula mismatch");

    const hnswlib::V0RawEstimateResult exact_only =
        context.evaluateRawFast(sidecar.edgeRecord(1U), 19.25);
    v0_test::require(
        exact_only.status == hnswlib::V0BoundStatus::ExactOnly,
        "raw_fast_v1 did not preserve exact-only fallback");
    const hnswlib::V0RawEstimateResult zero_length =
        context.evaluateRawFast(sidecar.edgeRecord(2U), 19.25);
    v0_test::require(
        zero_length.status == hnswlib::V0BoundStatus::ZeroLength,
        "raw_fast_v1 did not preserve zero-length fallback");
    const hnswlib::V0RawEstimateResult invalid_current =
        context.evaluateRawFast(sidecar.edgeRecord(0U), -1.0);
    v0_test::require(
        invalid_current.status ==
            hnswlib::V0BoundStatus::InvalidCurrentDistance,
        "raw_fast_v1 accepted invalid current distance");
}

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    return 2;
#else
    testRawEstimatorParity();
    testRawFastV1();
    std::cout << "v0_raw_estimator_test_ok" << std::endl;
    return 0;
#endif
}
