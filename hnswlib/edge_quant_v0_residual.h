#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hnswlib {

struct V0ResidualPruningConfig {
    double theta = 1.0;
    bool threshold_mode = false;

    bool valid() const {
        return std::isfinite(theta) && theta > 0.0 &&
            (threshold_mode || theta == 1.0);
    }
};

// A sketch is shared by every occurrence of an edge. Bit j=1 denotes a
// nonnegative projection. The query projection is never quantized.
class V0ResidualQueryContext {
 public:
    V0ResidualQueryContext(const float* query, const float* matrix,
                           uint32_t dimension, uint32_t bits)
        : bits_(bits), projected_(bits), signed_lut_((bits / 4U) * 16U) {
        if (!query || !matrix || !dimension || !bits || bits % 4U ||
            bits > 256U) {
            throw std::invalid_argument("invalid residual query projection");
        }
        for (uint32_t row = 0; row < bits; ++row) {
            float sum = 0.0f;
            const float* weights = matrix + static_cast<size_t>(row) * dimension;
            for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate)
                sum += weights[coordinate] * query[coordinate];
            if (!std::isfinite(sum))
                throw std::runtime_error("nonfinite residual query projection");
            projected_[row] = sum;
        }
        for (uint32_t group = 0; group < bits / 4U; ++group) {
            for (uint32_t pattern = 0; pattern < 16U; ++pattern) {
                float sum = 0.0f;
                for (uint32_t bit = 0; bit < 4U; ++bit)
                    sum += (pattern & (1U << bit) ? 1.0f : -1.0f) *
                        projected_[group * 4U + bit];
                signed_lut_[static_cast<size_t>(group) * 16U + pattern] = sum;
            }
        }
    }

    uint32_t bits() const { return bits_; }

    double signedDot(const uint8_t* packed) const {
        double sum = 0.0;
        for (uint32_t group = 0; group < bits_ / 4U; ++group) {
            const uint8_t byte = packed[group / 2U];
            const uint32_t pattern = (byte >> (4U * (group % 2U))) & 15U;
            sum += signed_lut_[static_cast<size_t>(group) * 16U + pattern];
        }
        return sum;
    }

    double correct(double raw, const uint8_t* packed,
                   float scale, float offset) const {
        return raw - (static_cast<double>(scale) * signedDot(packed) -
                      static_cast<double>(offset));
    }

 private:
    uint32_t bits_;
    std::vector<float> projected_;
    std::vector<float> signed_lut_;
};

// Offline reference for z=(v-c)-ell*hat_u and
// B=ell^2-||v-c||^2+2*ell*(anchor-c.hat_u).
inline double v0ResidualBias(const float* source, const float* target,
                             const float* centroid, const float* direction,
                             uint32_t dimension, double length,
                             double anchor) {
    double delta_norm = 0.0, centroid_dot = 0.0;
    for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double delta = static_cast<double>(target[coordinate]) -
                             source[coordinate];
        delta_norm += delta * delta;
        centroid_dot += static_cast<double>(centroid[coordinate]) *
                        direction[coordinate];
    }
    return length * length - delta_norm +
        2.0 * length * (anchor - centroid_dot);
}

}  // namespace hnswlib
