#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "tools/edge_estimation/query_store.h"

int main() {
    const char* path = "uq_query_store_test.fvecs";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const uint32_t dimension = 3U;
        const float rows[2][3] = {{1.0f, 2.0f, 3.0f}, {-4.0f, 5.0f, 6.0f}};
        for (size_t row = 0; row < 2U; ++row) {
            output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
            output.write(reinterpret_cast<const char*>(rows[row]), sizeof(rows[row]));
        }
        if (!output) throw std::runtime_error("failed to create query fixture");
    }
    uq::QueryStore store(path, 3U);
    if (std::remove(path) != 0) throw std::runtime_error("query fixture remained open");
    const float* second = store.query(1U);
    if (store.count() != 2U || store.dimension() != 3U || store.bytes() != 24U ||
        std::fabs(second[0] + 4.0f) > 1e-6f || std::fabs(second[2] - 6.0f) > 1e-6f)
        throw std::runtime_error("preloaded query values are incorrect");
    bool rejected = false;
    try { (void)store.query(2U); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("out-of-range query id was accepted");
    return 0;
}
