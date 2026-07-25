#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_quant_v0_encoder.h"

namespace {

struct Options {
    std::string sidecar_path;
    std::string index_path;
    size_t sample_records;

    Options() : sample_records(5U) {}
};

void printUsage(const char* program) {
    std::cerr
        << "Usage: " << program << "\n"
        << "  --sidecar <v0meta>\n"
        << "  [--index-path <hnsw-index>]\n"
        << "  [--sample-records <non-negative-integer>]\n";
}

uint64_t parseUint64(const std::string& text, const char* option) {
    if (text.empty() || text[0] == '-') {
        throw std::invalid_argument(
            std::string("Invalid unsigned value for ") + option);
    }
    size_t consumed = 0U;
    const unsigned long long value =
        std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument(
            std::string("Invalid unsigned value for ") + option);
    }
    return static_cast<uint64_t>(value);
}

Options parseOptions(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        std::exit(0);
    }
    std::unordered_map<std::string, std::string> values;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            throw std::invalid_argument(
                std::string("Missing value for option ") + argv[i]);
        }
        if (!values.insert(std::make_pair(
                std::string(argv[i]),
                std::string(argv[i + 1]))).second) {
            throw std::invalid_argument("Duplicate inspector option");
        }
    }
    if (values.find("--sidecar") == values.end()) {
        throw std::invalid_argument("Missing required option --sidecar");
    }
    for (std::unordered_map<std::string, std::string>::const_iterator it =
             values.begin();
         it != values.end();
         ++it) {
        if (it->first != "--sidecar" &&
            it->first != "--index-path" &&
            it->first != "--sample-records") {
            throw std::invalid_argument(
                std::string("Unknown option ") + it->first);
        }
    }
    Options result;
    result.sidecar_path = values["--sidecar"];
    if (values.find("--index-path") != values.end()) {
        result.index_path = values["--index-path"];
    }
    if (values.find("--sample-records") != values.end()) {
        const uint64_t parsed =
            parseUint64(
                values["--sample-records"], "--sample-records");
        if (parsed > static_cast<uint64_t>(
                std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument(
                "--sample-records exceeds size_t");
        }
        result.sample_records = static_cast<size_t>(parsed);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    try {
        const Options options = parseOptions(argc, argv);
        const hnswlib::V0OwnedSidecar owned =
            hnswlib::loadV0Sidecar(options.sidecar_path);
        const hnswlib::V0SidecarView sidecar = owned.view();
        const hnswlib::V0SidecarHeader& header = sidecar.header();

        uint64_t exact_only = 0U;
        uint64_t zero_length = 0U;
        double max_error = 0.0;
        long double error_sum = 0.0L;
        for (uint64_t edge = 0U;
             edge < header.directed_edge_count;
             ++edge) {
            const hnswlib::V0EdgeRecordView record =
                sidecar.edgeRecord(static_cast<size_t>(edge));
            if ((record.flags() & hnswlib::V0_EDGE_EXACT_ONLY) != 0U) {
                ++exact_only;
            }
            if ((record.flags() &
                 hnswlib::V0_ZERO_LENGTH_EDGE) != 0U) {
                ++zero_length;
            }
            error_sum += record.directionError();
            max_error =
                std::max(max_error, record.directionError());
        }

        std::cout << std::setprecision(17)
                  << "format_version=" << header.format_version << "\n"
                  << "dimension=" << header.dimension << "\n"
                  << "node_count=" << header.node_count << "\n"
                  << "directed_edge_count="
                  << header.directed_edge_count << "\n"
                  << "M_pq=" << header.pq_m << "\n"
                  << "nbits=" << header.pq_nbits << "\n"
                  << "code_bytes_per_edge="
                  << header.pq_code_size << "\n"
                  << "edge_record_stride="
                  << header.edge_record_stride << "\n"
                  << "sidecar_bytes=" << sidecar.fileSize() << "\n"
                  << "exact_only_edge_count=" << exact_only << "\n"
                  << "zero_length_edge_count=" << zero_length << "\n"
                  << "mean_direction_error="
                  << (header.directed_edge_count == zero_length
                          ? 0.0
                          : static_cast<double>(
                                error_sum /
                                (header.directed_edge_count -
                                 zero_length)))
                  << "\n"
                  << "max_direction_error=" << max_error << "\n"
                  << "base_index_sha256="
                  << hnswlib::edgeQuantV0Sha256Hex(
                        header.base_index_sha256) << "\n"
                  << "adjacency_sha256="
                  << hnswlib::edgeQuantV0Sha256Hex(
                        header.adjacency_sha256) << "\n"
                  << "codebook_sha256="
                  << hnswlib::edgeQuantV0Sha256Hex(
                        header.codebook_sha256) << "\n";

        if (!options.index_path.empty()) {
            hnswlib::L2Space space(header.dimension);
            hnswlib::HierarchicalNSW<float> index(
                &space, options.index_path);
            const hnswlib::V0Layer0GraphView graph =
                index.getV0Layer0GraphView();
            hnswlib::V0IndexCompatibility expected;
            expected.dimension = header.dimension;
            expected.node_count = graph.nodeCount();
            expected.directed_edge_count =
                hnswlib::buildV0NodeOffsets(graph).back();
            expected.base_index_sha256 =
                hnswlib::computeV0FileSha256(options.index_path);
            expected.adjacency_sha256 =
                graph.adjacencyFingerprint();
            hnswlib::validateV0SidecarCompatibility(
                header, expected);
            hnswlib::validateV0SidecarGraphLayout(sidecar, graph);
            std::cout << "index_compatibility=valid\n";
        }

        const size_t samples = std::min(
            options.sample_records,
            static_cast<size_t>(header.directed_edge_count));
        for (size_t edge = 0U; edge < samples; ++edge) {
            const hnswlib::V0EdgeRecordView record =
                sidecar.edgeRecord(edge);
            std::cout << "record[" << edge << "]"
                      << " flags="
                      << static_cast<unsigned int>(record.flags())
                      << " length=" << record.edgeLength()
                      << " epsilon=" << record.directionError()
                      << " anchor=" << record.anchorProjection()
                      << "\n";
        }
        std::cout << "v0_sidecar_inspector_ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v0_sidecar_inspector_error: "
                  << error.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }
#endif
}
