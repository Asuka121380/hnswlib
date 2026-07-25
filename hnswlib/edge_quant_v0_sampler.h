#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "edge_quant_v0_checksum.h"
#include "edge_quant_v0_graph_access.h"

namespace hnswlib {

class V0SplitMix64 {
 public:
    explicit V0SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        uint64_t value =
            (state_ += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30U)) *
            0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) *
            0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    // Uniform over [0, inclusive_max], using rejection rather than modulo
    // bias. The algorithm and PRNG are part of sampler format version 1.
    uint64_t uniformInclusive(uint64_t inclusive_max) {
        if (inclusive_max == std::numeric_limits<uint64_t>::max()) {
            return next();
        }
        const uint64_t bound = inclusive_max + 1U;
        const uint64_t threshold =
            static_cast<uint64_t>(-bound) % bound;
        uint64_t value = 0;
        do {
            value = next();
        } while (value < threshold);
        return value % bound;
    }

 private:
    uint64_t state_;
};

struct V0EdgeDirectionSample {
    uint32_t dimension;
    uint64_t requested_sample_count;
    uint64_t directed_edge_count;
    uint64_t valid_edge_count;
    uint64_t zero_length_edge_count;
    uint64_t seed;
    std::vector<float> directions;

    V0EdgeDirectionSample()
        : dimension(0),
          requested_sample_count(0),
          directed_edge_count(0),
          valid_edge_count(0),
          zero_length_edge_count(0),
          seed(0) {}

    size_t sampleCount() const {
        if (dimension == 0U) {
            return 0U;
        }
        return directions.size() / dimension;
    }
};

struct V0DirectionMatrixWriteResult {
    uint64_t byte_count;
    std::array<uint8_t, 32> sha256;

    V0DirectionMatrixWriteResult() : byte_count(0) {
        sha256.fill(0U);
    }
};

inline V0EdgeDirectionSample sampleV0Layer0EdgeDirections(
    const V0Layer0GraphView& graph,
    uint32_t dimension,
    size_t sample_count,
    uint64_t seed) {
    if (dimension == 0U) {
        throw std::invalid_argument(
            "V0 edge sampler dimension must be positive");
    }
    if (graph.dataSize() !=
        static_cast<size_t>(dimension) * sizeof(float)) {
        throw std::invalid_argument(
            "V0 edge sampler dimension does not match graph vector bytes");
    }
    if (sample_count >
        std::numeric_limits<size_t>::max() / dimension) {
        throw std::overflow_error(
            "V0 edge sampler reservoir size overflows memory indexing");
    }

    V0EdgeDirectionSample result;
    result.dimension = dimension;
    result.requested_sample_count =
        static_cast<uint64_t>(sample_count);
    result.seed = seed;
    result.directions.reserve(sample_count * dimension);
    V0SplitMix64 random(seed);

    graph.forEachEdge(
        [&graph, dimension, sample_count, &result, &random](
            tableint source_id,
            tableint target_id,
            size_t) {
            if (result.directed_edge_count ==
                std::numeric_limits<uint64_t>::max()) {
                throw std::overflow_error(
                    "V0 edge sampler directed-edge count overflow");
            }
            ++result.directed_edge_count;

            const float* source = graph.floatVector(source_id);
            const float* target = graph.floatVector(target_id);
            double squared_length = 0.0;
            for (uint32_t d = 0U; d < dimension; ++d) {
                if (!std::isfinite(static_cast<double>(source[d])) ||
                    !std::isfinite(static_cast<double>(target[d]))) {
                    throw std::runtime_error(
                        "V0 edge sampler encountered a non-finite vector");
                }
                const double difference =
                    static_cast<double>(target[d]) -
                    static_cast<double>(source[d]);
                squared_length += difference * difference;
            }
            if (!std::isfinite(squared_length)) {
                throw std::runtime_error(
                    "V0 edge sampler edge-length accumulation overflowed");
            }
            if (squared_length == 0.0) {
                if (result.zero_length_edge_count ==
                    std::numeric_limits<uint64_t>::max()) {
                    throw std::overflow_error(
                        "V0 edge sampler zero-length count overflow");
                }
                ++result.zero_length_edge_count;
                return;
            }
            if (result.valid_edge_count ==
                std::numeric_limits<uint64_t>::max()) {
                throw std::overflow_error(
                    "V0 edge sampler valid-edge count overflow");
            }

            const uint64_t item_index = result.valid_edge_count;
            size_t destination = sample_count;
            if (item_index < static_cast<uint64_t>(sample_count)) {
                destination = static_cast<size_t>(item_index);
                result.directions.resize(
                    (destination + 1U) * dimension);
            } else if (sample_count != 0U) {
                const uint64_t selected =
                    random.uniformInclusive(item_index);
                if (selected < static_cast<uint64_t>(sample_count)) {
                    destination = static_cast<size_t>(selected);
                }
            }

            if (destination < sample_count) {
                const double inverse_length =
                    1.0 / std::sqrt(squared_length);
                const size_t offset = destination * dimension;
                for (uint32_t d = 0U; d < dimension; ++d) {
                    const double direction =
                        (static_cast<double>(target[d]) -
                         static_cast<double>(source[d])) *
                        inverse_length;
                    const float stored = static_cast<float>(direction);
                    if (!std::isfinite(static_cast<double>(stored))) {
                        throw std::runtime_error(
                            "V0 edge sampler produced non-finite direction");
                    }
                    result.directions[offset + d] = stored;
                }
            }
            ++result.valid_edge_count;
        });

    if (result.directed_edge_count !=
        result.valid_edge_count + result.zero_length_edge_count) {
        throw std::logic_error(
            "V0 edge sampler accounting invariant failed");
    }
    return result;
}

inline bool v0SamplerPathExists(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return static_cast<bool>(input);
}

inline void v0SamplerWriteBytes(
    std::ofstream& output,
    const uint8_t* bytes,
    size_t size) {
    if (size == 0U) {
        return;
    }
    if (size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(
            "V0 edge sampler write exceeds stream-size capacity");
    }
    output.write(
        reinterpret_cast<const char*>(bytes),
        static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error(
            "Failed to write V0 direction matrix");
    }
}

inline V0DirectionMatrixWriteResult writeV0DirectionMatrix(
    const std::string& path,
    const V0EdgeDirectionSample& sample) {
    if (sample.dimension == 0U ||
        sample.directions.size() % sample.dimension != 0U) {
        throw std::invalid_argument(
            "V0 direction sample has an invalid matrix shape");
    }
    if (v0SamplerPathExists(path)) {
        throw std::runtime_error(
            "Refusing to overwrite existing V0 direction matrix");
    }
    const std::string partial_path = path + ".partial";
    if (v0SamplerPathExists(partial_path)) {
        throw std::runtime_error(
            "V0 direction matrix partial file already exists");
    }

    std::ofstream output(
        partial_path.c_str(),
        std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Failed to open V0 direction matrix for writing");
    }

    V0DirectionMatrixWriteResult result;
    EdgeQuantV0Sha256 sha;
    try {
        uint8_t bytes[4];
        for (size_t i = 0; i < sample.directions.size(); ++i) {
            uint32_t bits = 0U;
            std::memcpy(&bits, &sample.directions[i], sizeof(bits));
            for (size_t byte = 0; byte < 4U; ++byte) {
                bytes[byte] = static_cast<uint8_t>(
                    (bits >> (byte * 8U)) & 0xffU);
            }
            v0SamplerWriteBytes(output, bytes, sizeof(bytes));
            sha.update(bytes, sizeof(bytes));
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Failed to finalize V0 direction matrix");
        }
        output.close();
        result.byte_count =
            static_cast<uint64_t>(sample.directions.size()) * 4U;
        result.sha256 = sha.final();

        if (std::rename(partial_path.c_str(), path.c_str()) != 0) {
            throw std::runtime_error(
                "Failed to atomically publish V0 direction matrix");
        }
    } catch (...) {
        output.close();
        std::remove(partial_path.c_str());
        throw;
    }
    return result;
}

}  // namespace hnswlib
