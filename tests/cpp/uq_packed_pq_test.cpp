#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tools/edge_estimation/backends/pq_packed.h"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    for (uint32_t nbits = 1U; nbits <= 8U; ++nbits) {
        const uint32_t limit = 1U << nbits;
        std::vector<uint8_t> codes(17U);
        for (size_t i = 0; i < codes.size(); ++i)
            codes[i] = static_cast<uint8_t>((i * 13U + 3U) % limit);
        const std::vector<uint8_t> packed = uq::packPqCodes(codes, nbits);
        const uq::PackedPqModel model(static_cast<uint32_t>(codes.size()), nbits);
        require(packed.size() == model.packedCodeBytes(), "packed byte count mismatch");
        for (size_t i = 0; i < codes.size(); ++i)
            require(model.codeAt(&packed[0], packed.size(), static_cast<uint32_t>(i)) == codes[i],
                    "packed code mismatch");
        std::vector<float> lut(codes.size() * limit);
        double expected = 0.0;
        for (size_t sub = 0; sub < codes.size(); ++sub) {
            for (uint32_t code = 0; code < limit; ++code)
                lut[sub * limit + code] = static_cast<float>(sub * 0.25 + code * 0.5);
            expected += lut[sub * limit + codes[sub]];
        }
        const hnswlib::edge_estimation::DotEstimate estimate =
            model.estimate(lut, &packed[0], packed.size());
        require(estimate.valid(), "packed estimate invalid");
        require(std::fabs(estimate.value - expected) < 1e-4, "packed estimate mismatch");
    }
    const std::vector<uint8_t> odd = uq::packPqCodes(std::vector<uint8_t>{1, 2, 3}, 4);
    require(odd.size() == 2U && (odd[1] & 0xf0U) == 0U, "odd nibble padding is non-zero");
    std::cout << "uq_packed_pq_test_ok" << std::endl;
    return 0;
}
