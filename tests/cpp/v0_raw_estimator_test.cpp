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

}  // namespace

int main() {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    return 2;
#else
    testRawEstimatorParity();
    std::cout << "v0_raw_estimator_test_ok" << std::endl;
    return 0;
#endif
}
