#include <iostream>

#include "hnswlib/hnswlib.h"
#include "tools/edge_estimation/backends/pq_legacy.h"
#include "tools/edge_estimation/backends/pq_qjl_legacy.h"

int main() {
    if (uq::legacyStatus(hnswlib::V0BoundStatus::Valid) !=
        hnswlib::edge_estimation::EstimateStatus::Valid) {
        return 1;
    }
    if (uq::legacyStatus(hnswlib::V0BoundStatus::ExactOnly) !=
        hnswlib::edge_estimation::EstimateStatus::LegacyExactOnly) {
        return 1;
    }
    std::cout << "uq_legacy_adapter_compile_test_ok" << std::endl;
    return 0;
}
