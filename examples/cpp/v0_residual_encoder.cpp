#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/resource.h>
#endif

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_quant_v0_residual_io.h"

namespace {
struct Options {
    std::string index, sidecar, matrix, output;
    uint32_t dimension = 0U, bits = 0U, seed = 0U, chunk_edges = 256U;
    bool resume = false;
};

Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help") {
            std::cout << "v0_residual_encoder --index FILE --sidecar FILE "
                         "--matrix FILE --output FILE --dimension N "
                         "--bits N --seed N [--chunk-edges N] [--resume]\n";
            std::exit(0);
        }
        if (key == "--resume") { out.resume = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string value(argv[++i]);
        if (key == "--index") out.index = value;
        else if (key == "--sidecar") out.sidecar = value;
        else if (key == "--matrix") out.matrix = value;
        else if (key == "--output") out.output = value;
        else if (key == "--dimension") out.dimension = std::stoul(value);
        else if (key == "--bits") out.bits = std::stoul(value);
        else if (key == "--seed") out.seed = std::stoul(value);
        else if (key == "--chunk-edges") out.chunk_edges = std::stoul(value);
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (out.index.empty() || out.sidecar.empty() || out.matrix.empty() ||
        out.output.empty() || !out.dimension || !out.chunk_edges ||
        out.chunk_edges > 4096U) {
        throw std::invalid_argument("incomplete residual encoder options");
    }
    hnswlib::v0ResidualStride(out.bits);
    return out;
}

std::vector<float> readMatrix(const Options& options) {
    std::ifstream input(options.matrix.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read residual matrix");
    const uint64_t expected = static_cast<uint64_t>(options.dimension) *
        options.bits * sizeof(float);
    if (static_cast<uint64_t>(input.tellg()) != expected)
        throw std::runtime_error("residual matrix byte count mismatch");
    input.seekg(0);
    std::vector<float> matrix(static_cast<size_t>(options.dimension) * options.bits);
    input.read(reinterpret_cast<char*>(matrix.data()),
               static_cast<std::streamsize>(expected));
    if (!input) throw std::runtime_error("truncated residual matrix");
    for (size_t i = 0; i < matrix.size(); ++i)
        if (!std::isfinite(matrix[i]))
            throw std::runtime_error("nonfinite residual matrix");
    return matrix;
}

struct PendingEdge {
    hnswlib::tableint source = 0U;
    std::vector<double> z;
    std::vector<uint8_t> signs;
    double scale = 0.0, bias = 0.0, signed_centroid = 0.0;
    bool valid = false;
};
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const std::vector<float> matrix = readMatrix(options);
        hnswlib::L2Space space(options.dimension);
        hnswlib::HierarchicalNSW<float> index(&space, options.index);
        index.loadEdgeQuantV0Metadata(options.sidecar);
        const hnswlib::EdgeQuantV0Metadata& meta = index.getEdgeQuantV0Metadata();
        const hnswlib::V0SidecarHeader& header = meta.header();
        if (header.dimension != options.dimension)
            throw std::runtime_error("sidecar dimension mismatch");
        hnswlib::V0ResidualIdentity identity;
        identity.index_sha = index.getV0SerializedIndexFingerprint();
        identity.sidecar_sha = hnswlib::v0ResidualFileSha(options.sidecar);
        identity.adjacency_sha = index.getV0Layer0AdjacencyFingerprint();
        hnswlib::V0ResidualCompanionWriter writer(
            options.output, identity, matrix, options.dimension,
            options.bits, options.seed, header.directed_edge_count,
            options.resume);
        const hnswlib::V0Layer0GraphView graph = index.getV0Layer0GraphView();
        std::vector<float> direction(options.dimension);
        std::vector<PendingEdge> pending;
        pending.reserve(options.chunk_edges);
        uint64_t written = writer.written(), invalid = 0U;
        uint64_t seen = 0U;
        const auto started = std::chrono::steady_clock::now();
        const auto flush = [&]() {
            if (pending.empty()) return;
            // A matrix row stays hot while it is multiplied by the whole
            // chunk. Adjacent edges of one source reuse that source's G*c.
            for (uint32_t row = 0; row < options.bits; ++row) {
                const float* weights = matrix.data() +
                    static_cast<size_t>(row) * options.dimension;
                hnswlib::tableint cached_source =
                    std::numeric_limits<hnswlib::tableint>::max();
                double projected_c = 0.0;
                for (PendingEdge& item : pending) {
                    if (!item.valid) continue;
                    if (item.source != cached_source) {
                        cached_source = item.source;
                        projected_c = 0.0;
                        const float* c = graph.floatVector(item.source);
                        for (uint32_t coordinate = 0;
                             coordinate < options.dimension; ++coordinate)
                            projected_c += static_cast<double>(weights[coordinate]) *
                                c[coordinate];
                    }
                    double projected_z = 0.0;
                    for (uint32_t coordinate = 0;
                         coordinate < options.dimension; ++coordinate)
                        projected_z += static_cast<double>(weights[coordinate]) *
                            item.z[coordinate];
                    const bool positive = projected_z >= 0.0;
                    if (positive)
                        item.signs[row / 8U] |=
                            static_cast<uint8_t>(1U << (row % 8U));
                    item.signed_centroid += positive ? projected_c : -projected_c;
                }
            }
            for (PendingEdge& item : pending) {
                const double offset = item.scale * item.signed_centroid - item.bias;
                const bool valid = item.valid && std::isfinite(offset) &&
                    std::fabs(offset) <= std::numeric_limits<float>::max();
                writer.append(item.signs, static_cast<float>(item.scale),
                              static_cast<float>(offset), valid);
                invalid += !valid;
                ++written;
            }
            pending.clear();
            writer.checkpoint();
        };
        graph.forEachEdge([&](hnswlib::tableint source,
                              hnswlib::tableint target, size_t slot) {
            if (seen++ < writer.written() && pending.empty()) return;
            const hnswlib::V0EdgeRecordView edge = meta.edgeRecord(source, slot);
            const uint8_t flags = edge.flags();
            PendingEdge item;
            item.source = source;
            item.signs.resize(options.bits / 8U, 0U);
            if (flags & (hnswlib::V0_EDGE_EXACT_ONLY |
                         hnswlib::V0_ZERO_LENGTH_EDGE |
                         hnswlib::V0_RESERVED_INVALID)) {
                pending.push_back(std::move(item));
                if (pending.size() == options.chunk_edges) flush();
                return;
            }
            const double length = edge.edgeLengthNativeUnchecked();
            const double anchor = edge.anchorProjectionNativeUnchecked();
            const float* c = graph.floatVector(source);
            const float* v = graph.floatVector(target);
            const uint8_t* code = edge.codeDataUnchecked();
            double delta_norm = 0.0, centroid_dot = 0.0, z_norm = 0.0;
            item.z.resize(options.dimension);
            for (uint32_t coordinate = 0; coordinate < options.dimension; ++coordinate) {
                const uint32_t sub = coordinate / header.pq_dsub;
                const uint32_t within = coordinate % header.pq_dsub;
                const size_t flat = (static_cast<size_t>(sub) * header.pq_ksub +
                    code[sub]) * header.pq_dsub + within;
                direction[coordinate] = meta.nativeCodebookData()[flat];
                const double delta = static_cast<double>(v[coordinate]) - c[coordinate];
                item.z[coordinate] = delta - length * direction[coordinate];
                delta_norm += delta * delta;
                z_norm += item.z[coordinate] * item.z[coordinate];
                centroid_dot += static_cast<double>(c[coordinate]) * direction[coordinate];
            }
            item.bias = length * length - delta_norm +
                2.0 * length * (anchor - centroid_dot);
            item.scale = 2.0 * std::sqrt(z_norm) *
                std::sqrt(3.14159265358979323846 / 2.0) / options.bits;
            item.valid = std::isfinite(length) && length > 0.0 &&
                std::isfinite(anchor) && std::isfinite(item.scale) &&
                std::isfinite(item.bias) &&
                std::fabs(item.scale) <= std::numeric_limits<float>::max();
            pending.push_back(std::move(item));
            if (pending.size() == options.chunk_edges) flush();
        });
        flush();
        writer.finish();
        std::ifstream completed(options.output.c_str(), std::ios::binary | std::ios::ate);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "records=" << written << " invalid=" << invalid
                  << " seconds=" << seconds << " bytes=" << completed.tellg()
                  << " chunk_edges=" << options.chunk_edges;
#if defined(__linux__)
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0)
            std::cout << " peak_rss_kib=" << usage.ru_maxrss;
#endif
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
