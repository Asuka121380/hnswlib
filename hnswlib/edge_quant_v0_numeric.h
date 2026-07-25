#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hnswlib {

struct V0EdgeDirectionInfo {
    double edge_length;
    bool zero_length;

    V0EdgeDirectionInfo() : edge_length(0.0), zero_length(false) {}
};

struct V0EdgeNumericMetadata {
    double edge_length;
    double direction_error;
    double anchor_projection;
    double numeric_padding;
    bool zero_length;

    V0EdgeNumericMetadata()
        : edge_length(0.0),
          direction_error(0.0),
          anchor_projection(0.0),
          numeric_padding(0.0),
          zero_length(false) {}
};

inline V0EdgeDirectionInfo fillV0UnitEdgeDirection(
    const float* source,
    const float* target,
    uint32_t dimension,
    float* output) {
    if (source == NULL || target == NULL || output == NULL ||
        dimension == 0U) {
        throw std::invalid_argument(
            "V0 unit-direction input is invalid");
    }
    double squared_length = 0.0;
    for (uint32_t d = 0U; d < dimension; ++d) {
        if (!std::isfinite(static_cast<double>(source[d])) ||
            !std::isfinite(static_cast<double>(target[d]))) {
            throw std::runtime_error(
                "V0 edge contains a non-finite vector coordinate");
        }
        const double difference =
            static_cast<double>(target[d]) -
            static_cast<double>(source[d]);
        squared_length += difference * difference;
    }
    if (!std::isfinite(squared_length)) {
        throw std::runtime_error(
            "V0 edge-length accumulation overflowed");
    }

    V0EdgeDirectionInfo result;
    if (squared_length == 0.0) {
        result.zero_length = true;
        for (uint32_t d = 0U; d < dimension; ++d) {
            output[d] = 0.0f;
        }
        return result;
    }

    result.edge_length = std::sqrt(squared_length);
    if (!std::isfinite(result.edge_length) ||
        result.edge_length <= 0.0) {
        throw std::runtime_error("V0 edge length is invalid");
    }
    const double inverse_length = 1.0 / result.edge_length;
    for (uint32_t d = 0U; d < dimension; ++d) {
        const double direction =
            (static_cast<double>(target[d]) -
             static_cast<double>(source[d])) *
            inverse_length;
        output[d] = static_cast<float>(direction);
        if (!std::isfinite(static_cast<double>(output[d]))) {
            throw std::runtime_error(
                "V0 unit direction is not representable as float32");
        }
    }
    return result;
}

inline V0EdgeNumericMetadata computeV0EdgeNumericMetadata(
    const float* source,
    const float* target,
    const float* reconstructed_direction,
    uint32_t dimension) {
    if (source == NULL || target == NULL ||
        reconstructed_direction == NULL || dimension == 0U) {
        throw std::invalid_argument(
            "V0 numeric metadata input is invalid");
    }

    double squared_length = 0.0;
    for (uint32_t d = 0U; d < dimension; ++d) {
        if (!std::isfinite(static_cast<double>(source[d])) ||
            !std::isfinite(static_cast<double>(target[d])) ||
            !std::isfinite(
                static_cast<double>(reconstructed_direction[d]))) {
            throw std::runtime_error(
                "V0 numeric metadata encountered a non-finite value");
        }
        const double difference =
            static_cast<double>(target[d]) -
            static_cast<double>(source[d]);
        squared_length += difference * difference;
    }
    if (!std::isfinite(squared_length)) {
        throw std::runtime_error(
            "V0 numeric edge-length accumulation overflowed");
    }

    V0EdgeNumericMetadata result;
    if (squared_length == 0.0) {
        result.zero_length = true;
        return result;
    }

    result.edge_length = std::sqrt(squared_length);
    const double inverse_length = 1.0 / result.edge_length;
    double squared_error = 0.0;
    double anchor = 0.0;
    for (uint32_t d = 0U; d < dimension; ++d) {
        const double exact_direction =
            (static_cast<double>(target[d]) -
             static_cast<double>(source[d])) *
            inverse_length;
        const double reconstructed =
            static_cast<double>(reconstructed_direction[d]);
        const double difference = exact_direction - reconstructed;
        squared_error += difference * difference;
        anchor += static_cast<double>(source[d]) * reconstructed;
    }
    if (!std::isfinite(squared_error) || !std::isfinite(anchor)) {
        throw std::runtime_error(
            "V0 numeric metadata accumulation overflowed");
    }
    const double exact_error = std::sqrt(squared_error);
    result.direction_error = std::nextafter(
        exact_error, std::numeric_limits<double>::infinity());
    result.anchor_projection = anchor;
    result.numeric_padding = 0.0;
    if (!std::isfinite(result.direction_error) ||
        result.direction_error < exact_error) {
        throw std::runtime_error(
            "V0 direction error could not be rounded conservatively");
    }
    return result;
}

}  // namespace hnswlib
