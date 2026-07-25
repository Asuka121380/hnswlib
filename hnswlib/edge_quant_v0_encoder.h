#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib.h"
#include "edge_quant_v0_io.h"
#include "edge_quant_v0_numeric.h"

namespace hnswlib {

struct V0AllEdgeEncodingSpec {
    uint32_t dimension;
    uint32_t pq_m;
    uint32_t pq_nbits;
    uint32_t pq_ksub;
    uint32_t pq_dsub;
    size_t block_size;
    std::string output_sidecar;
    std::string training_metadata_json;
    std::vector<float> codebook_centroids;
    V0Sha256Digest base_index_sha256;
    V0Sha256Digest adjacency_sha256;

    V0AllEdgeEncodingSpec()
        : dimension(0U),
          pq_m(0U),
          pq_nbits(0U),
          pq_ksub(0U),
          pq_dsub(0U),
          block_size(0U) {
        base_index_sha256.fill(0U);
        adjacency_sha256.fill(0U);
    }
};

struct V0AllEdgeEncodingMetrics {
    uint64_t node_count;
    uint64_t directed_edge_count;
    uint64_t encoded_edge_count;
    uint64_t zero_length_edge_count;
    uint64_t exact_only_edge_count;
    uint64_t block_count;
    double mean_direction_error;
    double max_direction_error;
    uint64_t sidecar_bytes;
    V0Sha256Digest sidecar_sha256;

    V0AllEdgeEncodingMetrics()
        : node_count(0U),
          directed_edge_count(0U),
          encoded_edge_count(0U),
          zero_length_edge_count(0U),
          exact_only_edge_count(0U),
          block_count(0U),
          mean_direction_error(0.0),
          max_direction_error(0.0),
          sidecar_bytes(0U) {
        sidecar_sha256.fill(0U);
    }
};

inline bool v0EncoderPathExists(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return static_cast<bool>(input);
}

inline std::vector<uint64_t> buildV0NodeOffsets(
    const V0Layer0GraphView& graph) {
    std::vector<uint64_t> offsets(graph.nodeCount() + 1U, 0U);
    uint64_t edge_count = 0U;
    for (size_t node = 0U; node < graph.nodeCount(); ++node) {
        const uint64_t degree = static_cast<uint64_t>(
            graph.neighbors(static_cast<tableint>(node)).size);
        if (degree > std::numeric_limits<uint64_t>::max() - edge_count) {
            throw std::overflow_error("V0 all-edge count overflow");
        }
        edge_count += degree;
        offsets[node + 1U] = edge_count;
    }
    return offsets;
}

template<typename Codec>
V0AllEdgeEncodingMetrics encodeV0AllLayer0Edges(
    const V0Layer0GraphView& graph,
    const V0AllEdgeEncodingSpec& spec,
    Codec& codec) {
    if (spec.dimension == 0U || spec.pq_m == 0U ||
        spec.pq_nbits != 8U ||
        spec.pq_ksub != 256U ||
        spec.pq_dsub == 0U ||
        spec.dimension != spec.pq_m * spec.pq_dsub ||
        spec.block_size == 0U ||
        spec.output_sidecar.empty()) {
        throw std::invalid_argument(
            "V0 all-edge encoding specification is invalid");
    }
    if (graph.dataSize() !=
        static_cast<size_t>(spec.dimension) * sizeof(float)) {
        throw std::invalid_argument(
            "V0 encoder dimension does not match graph vectors");
    }
    if (codec.dimension() != spec.dimension ||
        codec.codeSize() != spec.pq_m) {
        throw std::invalid_argument(
            "V0 codec does not match the encoding specification");
    }
    if (spec.block_size >
        std::numeric_limits<size_t>::max() / spec.dimension) {
        throw std::overflow_error("V0 encoding block is too large");
    }
    if (v0EncoderPathExists(spec.output_sidecar)) {
        throw std::runtime_error(
            "Refusing to overwrite an existing V0 sidecar");
    }
    const std::string partial_path = spec.output_sidecar + ".partial";
    if (v0EncoderPathExists(partial_path)) {
        throw std::runtime_error(
            "V0 sidecar partial file already exists");
    }

    const std::vector<uint64_t> node_offsets =
        buildV0NodeOffsets(graph);
    V0SidecarWriteSpec write_spec;
    write_spec.dimension = spec.dimension;
    write_spec.pq_m = spec.pq_m;
    write_spec.pq_nbits = spec.pq_nbits;
    write_spec.pq_ksub = spec.pq_ksub;
    write_spec.pq_dsub = spec.pq_dsub;
    write_spec.node_count = graph.nodeCount();
    write_spec.directed_edge_count = node_offsets.back();
    write_spec.training_metadata_json =
        spec.training_metadata_json;
    write_spec.codebook_centroids = spec.codebook_centroids;
    write_spec.node_offsets = node_offsets;
    write_spec.base_index_sha256 = spec.base_index_sha256;
    write_spec.adjacency_sha256 = spec.adjacency_sha256;

    V0AllEdgeEncodingMetrics metrics;
    metrics.node_count = graph.nodeCount();
    metrics.directed_edge_count = node_offsets.back();
    long double direction_error_sum = 0.0L;

    std::vector<tableint> sources;
    std::vector<tableint> targets;
    std::vector<uint8_t> zero_length;
    std::vector<float> directions;
    std::vector<uint8_t> codes;
    std::vector<float> decoded;
    sources.reserve(spec.block_size);
    targets.reserve(spec.block_size);
    zero_length.reserve(spec.block_size);
    directions.reserve(spec.block_size * spec.dimension);
    codes.reserve(spec.block_size * spec.pq_m);
    decoded.reserve(spec.block_size * spec.dimension);

    try {
        V0SidecarWriter writer(partial_path, write_spec);
        const std::vector<uint8_t> zero_code(spec.pq_m, 0U);
        const auto flush_block = [&]() {
            const size_t count = sources.size();
            if (count == 0U) {
                return;
            }
            codes.assign(count * spec.pq_m, 0U);
            decoded.assign(count * spec.dimension, 0.0f);
            codec.encode(directions.data(), count, codes.data());
            codec.decode(codes.data(), count, decoded.data());

            for (size_t row_index = 0U;
                 row_index < count;
                 ++row_index) {
                if (zero_length[row_index] != 0U) {
                    writer.writeEdgeRecord(
                        zero_code.data(),
                        zero_code.size(),
                        static_cast<uint8_t>(
                            V0_EDGE_EXACT_ONLY |
                            V0_ZERO_LENGTH_EDGE),
                        0.0,
                        0.0,
                        0.0,
                        0.0);
                    ++metrics.zero_length_edge_count;
                    ++metrics.exact_only_edge_count;
                    continue;
                }
                const V0EdgeNumericMetadata numeric =
                    computeV0EdgeNumericMetadata(
                        graph.floatVector(sources[row_index]),
                        graph.floatVector(targets[row_index]),
                        decoded.data() +
                            row_index * spec.dimension,
                        spec.dimension);
                if (numeric.zero_length) {
                    throw std::logic_error(
                        "V0 edge changed zero-length classification");
                }
                writer.writeEdgeRecord(
                    codes.data() + row_index * spec.pq_m,
                    spec.pq_m,
                    0U,
                    numeric.edge_length,
                    numeric.direction_error,
                    numeric.anchor_projection,
                    numeric.numeric_padding);
                ++metrics.encoded_edge_count;
                direction_error_sum +=
                    static_cast<long double>(
                        numeric.direction_error);
                if (numeric.direction_error >
                    metrics.max_direction_error) {
                    metrics.max_direction_error =
                        numeric.direction_error;
                }
            }
            ++metrics.block_count;
            sources.clear();
            targets.clear();
            zero_length.clear();
            directions.clear();
        };

        for (size_t node = 0U; node < graph.nodeCount(); ++node) {
            const tableint source_id = static_cast<tableint>(node);
            const V0Layer0NeighborSpan span = graph.neighbors(source_id);
            for (size_t slot = 0U; slot < span.size; ++slot) {
                const tableint target_id = span.ids[slot];
                const size_t row = sources.size();
                sources.push_back(source_id);
                targets.push_back(target_id);
                zero_length.push_back(0U);
                directions.resize(
                    (row + 1U) * static_cast<size_t>(spec.dimension));
                const V0EdgeDirectionInfo direction =
                    fillV0UnitEdgeDirection(
                        graph.floatVector(source_id),
                        graph.floatVector(target_id),
                        spec.dimension,
                        directions.data() + row * spec.dimension);
                zero_length[row] = direction.zero_length ? 1U : 0U;

                if (sources.size() == spec.block_size) {
                    flush_block();
                }
            }
        }
        flush_block();
        writer.finalize();

        const V0OwnedSidecar validated =
            loadV0Sidecar(partial_path);
        V0IndexCompatibility expected;
        expected.dimension = spec.dimension;
        expected.node_count = graph.nodeCount();
        expected.directed_edge_count = node_offsets.back();
        expected.base_index_sha256 = spec.base_index_sha256;
        expected.adjacency_sha256 = spec.adjacency_sha256;
        validateV0SidecarCompatibility(
            validated.header(), expected);
        validateV0SidecarGraphLayout(validated.view(), graph);
        const std::vector<uint8_t> serialized_codebook =
            edge_quant_v0_detail::serializeCodebook(
                spec.codebook_centroids);
        if (validated.header().codebook_sha256 !=
            edge_quant_v0_detail::sha256(
                serialized_codebook.data(),
                serialized_codebook.size())) {
            throw std::runtime_error(
                "V0 encoded sidecar codebook fingerprint mismatch");
        }

        if (std::rename(
                partial_path.c_str(),
                spec.output_sidecar.c_str()) != 0) {
            throw std::runtime_error(
                "Failed to atomically publish V0 sidecar");
        }
    } catch (...) {
        std::remove(partial_path.c_str());
        throw;
    }

    if (metrics.encoded_edge_count +
            metrics.zero_length_edge_count !=
        metrics.directed_edge_count) {
        throw std::logic_error(
            "V0 all-edge accounting invariant failed");
    }
    if (metrics.encoded_edge_count != 0U) {
        metrics.mean_direction_error = static_cast<double>(
            direction_error_sum /
            static_cast<long double>(metrics.encoded_edge_count));
    }
    std::ifstream completed(
        spec.output_sidecar.c_str(),
        std::ios::binary | std::ios::ate);
    if (!completed || completed.tellg() < 0) {
        throw std::runtime_error(
            "Failed to determine completed V0 sidecar size");
    }
    metrics.sidecar_bytes =
        static_cast<uint64_t>(completed.tellg());
    completed.close();
    metrics.sidecar_sha256 =
        computeV0FileSha256(spec.output_sidecar);
    return metrics;
}

}  // namespace hnswlib
