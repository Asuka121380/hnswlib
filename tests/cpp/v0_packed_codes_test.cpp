#include <iostream>
#include <vector>
#include "hnswlib/edge_quant_v0_packed_codes.h"

int main() {
    for (unsigned bits = 1; bits <= 8; ++bits) {
        for (size_t m : {size_t(1), size_t(3), size_t(16), size_t(32)}) {
            const size_t stride = hnswlib::v0PackedCodeSize(m, bits);
            std::vector<uint8_t> bytes(7 * m), packed(7 * stride), decoded(7 * m);
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (i * 17 + 63) % (1U << bits);
            hnswlib::v0PackCodes(bytes.data(), 7, m, bits, packed.data());
            hnswlib::v0UnpackCodes(packed.data(), 7, m, bits, decoded.data());
            if (bytes != decoded) return 1;
        }
    }
    // Explicit reference: 4 six-bit centroids 0,1,2,63 => 0x40,0x20,0xfc.
    uint8_t bytes[] = {0, 1, 2, 63}, packed[3];
    hnswlib::v0PackCodes(bytes, 1, 4, 6, packed);
    if (packed[0] != 0x40 || packed[1] != 0x20 || packed[2] != 0xfc) return 2;
    bytes[3] = 64;
    try { hnswlib::v0PackCodes(bytes, 1, 4, 6, packed); return 3; }
    catch (const std::invalid_argument&) {}
    try { hnswlib::v0PackedCodeSize(16, 9); return 4; }
    catch (const std::invalid_argument&) {}
    std::cout << "v0_packed_codes_test passed\n";
}
