#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "hnswlib/edge_estimation/types.h"

namespace uq {

class PackedPqModel {
 public:
    PackedPqModel(uint32_t subquantizers, uint32_t nbits)
        : m_(subquantizers), nbits_(nbits), ksub_(1U << nbits) {
        if (m_ == 0U || nbits_ == 0U || nbits_ > 8U)
            throw std::invalid_argument("invalid packed PQ shape");
    }

    uint32_t subquantizers() const { return m_; }
    uint32_t nbits() const { return nbits_; }
    uint32_t centroidCount() const { return ksub_; }
    size_t packedCodeBytes() const {
        return (static_cast<size_t>(m_) * nbits_ + 7U) / 8U;
    }

    uint32_t codeAt(const uint8_t* packed, size_t packed_size, uint32_t sub) const {
        if (packed == NULL || packed_size != packedCodeBytes() || sub >= m_)
            throw std::invalid_argument("invalid packed PQ record");
        const uint64_t bit_offset = static_cast<uint64_t>(sub) * nbits_;
        const size_t byte_offset = static_cast<size_t>(bit_offset / 8U);
        const uint32_t shift = static_cast<uint32_t>(bit_offset % 8U);
        uint32_t window = packed[byte_offset];
        if (shift + nbits_ > 8U) {
            if (byte_offset + 1U >= packed_size)
                throw std::runtime_error("packed PQ code crosses record boundary");
            window |= static_cast<uint32_t>(packed[byte_offset + 1U]) << 8U;
        }
        return (window >> shift) & (ksub_ - 1U);
    }

    hnswlib::edge_estimation::DotEstimate estimate(
        const std::vector<float>& lut,
        const uint8_t* packed,
        size_t packed_size) const {
        if (lut.size() != static_cast<size_t>(m_) * ksub_)
            return hnswlib::edge_estimation::DotEstimate(
                0.0, hnswlib::edge_estimation::EstimateStatus::UnsupportedRecord);
        float sum = 0.0f;
        try {
            for (uint32_t sub = 0U; sub < m_; ++sub) {
                const uint32_t code = codeAt(packed, packed_size, sub);
                sum += lut[static_cast<size_t>(sub) * ksub_ + code];
            }
        } catch (const std::exception&) {
            return hnswlib::edge_estimation::DotEstimate(
                0.0, hnswlib::edge_estimation::EstimateStatus::InvalidCode);
        }
        if (!std::isfinite(static_cast<double>(sum)))
            return hnswlib::edge_estimation::DotEstimate(
                0.0, hnswlib::edge_estimation::EstimateStatus::NonFinite);
        return hnswlib::edge_estimation::DotEstimate(
            static_cast<double>(sum),
            hnswlib::edge_estimation::EstimateStatus::Valid);
    }

 private:
    uint32_t m_;
    uint32_t nbits_;
    uint32_t ksub_;
};

inline std::vector<uint8_t> packPqCodes(
    const std::vector<uint8_t>& codes,
    uint32_t nbits) {
    if (nbits == 0U || nbits > 8U) throw std::invalid_argument("invalid nbits");
    const uint32_t limit = 1U << nbits;
    std::vector<uint8_t> packed((codes.size() * nbits + 7U) / 8U, 0U);
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] >= limit) throw std::invalid_argument("PQ code out of range");
        const uint64_t bit_offset = static_cast<uint64_t>(i) * nbits;
        const size_t byte_offset = static_cast<size_t>(bit_offset / 8U);
        const uint32_t shift = static_cast<uint32_t>(bit_offset % 8U);
        const uint16_t shifted = static_cast<uint16_t>(codes[i]) << shift;
        packed[byte_offset] |= static_cast<uint8_t>(shifted & 0xffU);
        if (shift + nbits > 8U)
            packed[byte_offset + 1U] |= static_cast<uint8_t>(shifted >> 8U);
    }
    return packed;
}

}  // namespace uq
