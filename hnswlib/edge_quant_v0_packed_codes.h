#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hnswlib {
// Faiss generic PQ codes are little-endian bit streams, padded per vector.
// V0 sidecars deliberately retain one byte per subquantizer (no online unpack).
inline size_t v0PackedCodeSize(size_t m, unsigned bits) {
    if (!m || !bits || bits > 8 ||
        m > (std::numeric_limits<size_t>::max() - 7) / bits)
        throw std::invalid_argument("invalid PQ packing dimensions");
    return (m * bits + 7) / 8;
}

inline void v0UnpackCodes(const uint8_t* packed, size_t count, size_t m,
                          unsigned bits, uint8_t* bytes) {
    const size_t stride = v0PackedCodeSize(m, bits);
    if (count > std::numeric_limits<size_t>::max() / m)
        throw std::overflow_error("PQ code buffer size overflow");
    for (size_t row = 0; row < count; ++row) {
        for (size_t j = 0; j < m; ++j) {
            const size_t bit = j * bits;
            unsigned value = packed[row * stride + bit / 8] >> (bit % 8);
            if (bit % 8 + bits > 8)
                value |= unsigned(packed[row * stride + bit / 8 + 1]) << (8 - bit % 8);
            bytes[row * m + j] = static_cast<uint8_t>(value & ((1U << bits) - 1));
        }
    }
}

inline void v0PackCodes(const uint8_t* bytes, size_t count, size_t m,
                        unsigned bits, uint8_t* packed) {
    const size_t stride = v0PackedCodeSize(m, bits);
    if (count > std::numeric_limits<size_t>::max() / m)
        throw std::overflow_error("PQ code buffer size overflow");
    for (size_t row = 0; row < count; ++row) {
        for (size_t i = 0; i < stride; ++i) packed[row * stride + i] = 0;
        for (size_t j = 0; j < m; ++j) {
            const unsigned value = bytes[row * m + j];
            if (value >= (1U << bits)) throw std::invalid_argument("PQ code out of range");
            const size_t bit = j * bits;
            packed[row * stride + bit / 8] |= static_cast<uint8_t>(value << (bit % 8));
            if (bit % 8 + bits > 8)
                packed[row * stride + bit / 8 + 1] |= static_cast<uint8_t>(value >> (8 - bit % 8));
        }
    }
}
}  // namespace hnswlib
