#include <iostream>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hnswlib/eq_rcp_types.h"
#include "hnswlib/hnswlib.h"

int main() {
#ifndef HNSWLIB_ENABLE_EQ_RCP
    std::cerr << "HNSWLIB_ENABLE_EQ_RCP is required" << std::endl;
    return 2;
#else
    hnswlib::EqRcpLayoutDescriptor layout;
    if (layout.valid()) {
        throw std::runtime_error(
            "A zero-dimensional EQ-RCP layout must be invalid");
    }
    layout.dimension = 960U;
    if (!layout.valid() || layout.codeBytes() != 32U ||
        layout.logical_record_bytes != 44U ||
        layout.edge_record_stride != 48U) {
        throw std::runtime_error("The frozen EQ-RCP layout contract is invalid");
    }

    hnswlib::EqRcpLayoutDescriptor invalid = layout;
    invalid.residual_code_bytes = 7U;
    if (invalid.valid()) {
        throw std::runtime_error("An invalid EQ-RCP code budget was accepted");
    }
    invalid = layout;
    invalid.edge_record_stride = 46U;
    if (invalid.valid()) {
        throw std::runtime_error("An unaligned EQ-RCP record was accepted");
    }

    const size_t dimension = 8U;
    const size_t count = 64U;
    std::vector<float> points(count * dimension);
    for (size_t row = 0; row < count; ++row) {
        for (size_t column = 0; column < dimension; ++column) {
            const int raw = static_cast<int>((row * 17U + column * 29U) % 101U);
            points[row * dimension + column] =
                static_cast<float>(raw - 50) / 17.0f +
                static_cast<float>(row) * 0.001f;
        }
    }

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, count, 12U, 80U, 41U);
    for (size_t row = 0; row < count; ++row) {
        index.addPoint(points.data() + row * dimension, row);
    }
    index.setEf(32U);

    const std::priority_queue<std::pair<float, hnswlib::labeltype> > result =
        index.searchKnn(points.data() + 11U * dimension, 5U);
    if (result.size() != 5U) {
        throw std::runtime_error("Baseline search returned an unexpected size");
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype> > copy = result;
    bool found_self = false;
    while (!copy.empty()) {
        if (copy.top().second == 11U && copy.top().first == 0.0f) {
            found_self = true;
        }
        copy.pop();
    }
    if (!found_self) {
        throw std::runtime_error(
            "Feature-isolation baseline search did not recover the query point");
    }

    std::cout << "eq_rcp_feature_isolation_test_ok" << std::endl;
    return 0;
#endif
}
