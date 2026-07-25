#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "hnswlib/hnswlib.h"

namespace {

struct Options {
    std::string index_path;
    std::string base_path;
    uint32_t dimension;
    size_t sample_count;
    uint64_t seed;
    std::string output_directions;
    std::string output_manifest;

    Options() : dimension(0), sample_count(0), seed(0) {}
};

void printUsage(const char* program) {
    std::cerr
        << "Usage: " << program << "\n"
        << "  --index-path <hnsw-index>\n"
        << "  [--base-path <base-vector-file-for-provenance>]\n"
        << "  --dimension <positive-integer>\n"
        << "  --sample-count <positive-integer>\n"
        << "  --seed <uint64>\n"
        << "  --output-directions <raw-little-endian-f32>\n"
        << "  --output-manifest <json>\n";
}

uint64_t parseUint64(const std::string& text, const char* option) {
    if (text.empty() || text[0] == '-') {
        throw std::invalid_argument(
            std::string("Invalid unsigned value for ") + option);
    }
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed, 10);
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
        const std::string name(argv[i]);
        if (name.size() < 3U || name.substr(0, 2) != "--") {
            throw std::invalid_argument(
                std::string("Unexpected argument ") + name);
        }
        if (!values.insert(
                std::make_pair(name, std::string(argv[i + 1]))).second) {
            throw std::invalid_argument(
                std::string("Duplicate option ") + name);
        }
    }

    const char* required[] = {
        "--index-path",
        "--dimension",
        "--sample-count",
        "--seed",
        "--output-directions",
        "--output-manifest"
    };
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        if (values.find(required[i]) == values.end()) {
            throw std::invalid_argument(
                std::string("Missing required option ") + required[i]);
        }
    }
    for (std::unordered_map<std::string, std::string>::const_iterator it =
             values.begin();
         it != values.end();
         ++it) {
        const std::string& name = it->first;
        if (name != "--index-path" &&
            name != "--base-path" &&
            name != "--dimension" &&
            name != "--sample-count" &&
            name != "--seed" &&
            name != "--output-directions" &&
            name != "--output-manifest") {
            throw std::invalid_argument(
                std::string("Unknown option ") + name);
        }
    }

    Options options;
    options.index_path = values["--index-path"];
    if (values.find("--base-path") != values.end()) {
        options.base_path = values["--base-path"];
    }
    const uint64_t dimension =
        parseUint64(values["--dimension"], "--dimension");
    const uint64_t sample_count =
        parseUint64(values["--sample-count"], "--sample-count");
    options.seed = parseUint64(values["--seed"], "--seed");
    if (dimension == 0U ||
        dimension > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            "--dimension must fit a positive uint32");
    }
    if (sample_count == 0U ||
        sample_count >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument(
            "--sample-count must fit a positive size_t");
    }
    options.dimension = static_cast<uint32_t>(dimension);
    options.sample_count = static_cast<size_t>(sample_count);
    options.output_directions = values["--output-directions"];
    options.output_manifest = values["--output-manifest"];
    if (options.index_path.empty() ||
        options.output_directions.empty() ||
        options.output_manifest.empty()) {
        throw std::invalid_argument("V0 sampler paths must not be empty");
    }
    if (options.output_directions == options.output_manifest) {
        throw std::invalid_argument(
            "Direction matrix and manifest paths must differ");
    }
    return options;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c =
            static_cast<unsigned char>(value[i]);
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20U) {
                    output
                        << "\\u"
                        << std::hex << std::setw(4)
                        << std::setfill('0')
                        << static_cast<unsigned int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    output << '"';
    return output.str();
}

void writeTextAtomically(
    const std::string& path,
    const std::string& text) {
    if (hnswlib::v0SamplerPathExists(path)) {
        throw std::runtime_error(
            "Refusing to overwrite existing V0 sampler manifest");
    }
    const std::string partial_path = path + ".partial";
    if (hnswlib::v0SamplerPathExists(partial_path)) {
        throw std::runtime_error(
            "V0 sampler manifest partial file already exists");
    }
    std::ofstream output(
        partial_path.c_str(),
        std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Failed to open V0 sampler manifest for writing");
    }
    try {
        output.write(
            text.data(),
            static_cast<std::streamsize>(text.size()));
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Failed to finalize V0 sampler manifest");
        }
        output.close();
        if (std::rename(partial_path.c_str(), path.c_str()) != 0) {
            throw std::runtime_error(
                "Failed to atomically publish V0 sampler manifest");
        }
    } catch (...) {
        output.close();
        std::remove(partial_path.c_str());
        throw;
    }
}

std::string buildManifest(
    const Options& options,
    const hnswlib::V0EdgeDirectionSample& sample,
    const hnswlib::V0DirectionMatrixWriteResult& matrix,
    uint64_t node_count,
    const hnswlib::V0Sha256Digest& index_sha256,
    const hnswlib::V0Sha256Digest& adjacency_sha256,
    const hnswlib::V0Sha256Digest* base_sha256) {
    std::ostringstream json;
    json << "{\n"
         << "  \"format\": \"hnswlib_v0_edge_direction_sample\",\n"
         << "  \"format_version\": 1,\n"
         << "  \"sampling_algorithm\": "
         << "\"reservoir_algorithm_r_splitmix64_v1\",\n"
         << "  \"traversal_order\": \"source_id_then_layer0_slot\",\n"
         << "  \"sampling_population\": "
         << "\"all_nonzero_layer0_directed_edges\",\n"
         << "  \"metric\": \"squared_l2\",\n"
         << "  \"sampled_object\": \"unit_edge_direction\",\n"
         << "  \"vector_source\": \"embedded_hnsw_level0_data\",\n"
         << "  \"base_path_role\": \"optional_provenance_only\",\n"
         << "  \"index_path\": " << jsonEscape(options.index_path) << ",\n"
         << "  \"base_path\": ";
    if (options.base_path.empty()) {
        json << "null,\n";
    } else {
        json << jsonEscape(options.base_path) << ",\n";
    }
    json << "  \"directions_path\": "
         << jsonEscape(options.output_directions) << ",\n"
         << "  \"dtype\": \"float32\",\n"
         << "  \"byte_order\": \"little\",\n"
         << "  \"layout\": \"row_major\",\n"
         << "  \"dimension\": " << sample.dimension << ",\n"
         << "  \"seed\": " << sample.seed << ",\n"
         << "  \"requested_sample_count\": "
         << sample.requested_sample_count << ",\n"
         << "  \"produced_sample_count\": "
         << sample.sampleCount() << ",\n"
         << "  \"node_count\": " << node_count << ",\n"
         << "  \"directed_edge_count\": "
         << sample.directed_edge_count << ",\n"
         << "  \"valid_edge_count\": "
         << sample.valid_edge_count << ",\n"
         << "  \"zero_length_edge_count\": "
         << sample.zero_length_edge_count << ",\n"
         << "  \"directions_bytes\": "
         << matrix.byte_count << ",\n"
         << "  \"index_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(index_sha256))
         << ",\n"
         << "  \"adjacency_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(adjacency_sha256))
         << ",\n"
         << "  \"base_file_sha256\": ";
    if (base_sha256 == NULL) {
        json << "null,\n";
    } else {
        json << jsonEscape(
            hnswlib::edgeQuantV0Sha256Hex(*base_sha256)) << ",\n";
    }
    json << "  \"directions_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(matrix.sha256))
         << "\n"
         << "}\n";
    return json.str();
}

}  // namespace

int main(int argc, char** argv) {
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
    std::cerr << "HNSWLIB_ENABLE_EDGE_QUANT_V0 is required" << std::endl;
    return 2;
#else
    try {
        const Options options = parseOptions(argc, argv);
        if (hnswlib::v0SamplerPathExists(options.output_directions) ||
            hnswlib::v0SamplerPathExists(options.output_manifest)) {
            throw std::runtime_error(
                "Refusing to overwrite existing V0 sampler output");
        }

        hnswlib::L2Space space(options.dimension);
        hnswlib::HierarchicalNSW<float> index(
            &space, options.index_path);
        const hnswlib::V0Layer0GraphView graph =
            index.getV0Layer0GraphView();
        const hnswlib::V0EdgeDirectionSample sample =
            hnswlib::sampleV0Layer0EdgeDirections(
                graph,
                options.dimension,
                options.sample_count,
                options.seed);

        const hnswlib::V0Sha256Digest index_sha256 =
            hnswlib::computeV0FileSha256(options.index_path);
        const hnswlib::V0Sha256Digest adjacency_sha256 =
            graph.adjacencyFingerprint();
        hnswlib::V0Sha256Digest base_sha256;
        const hnswlib::V0Sha256Digest* base_sha256_ptr = NULL;
        if (!options.base_path.empty()) {
            base_sha256 =
                hnswlib::computeV0FileSha256(options.base_path);
            base_sha256_ptr = &base_sha256;
        }

        bool directions_published = false;
        try {
            const hnswlib::V0DirectionMatrixWriteResult matrix =
                hnswlib::writeV0DirectionMatrix(
                    options.output_directions, sample);
            directions_published = true;
            writeTextAtomically(
                options.output_manifest,
                buildManifest(
                    options,
                    sample,
                    matrix,
                    graph.nodeCount(),
                    index_sha256,
                    adjacency_sha256,
                    base_sha256_ptr));
            std::cout
                << "v0_edge_sampler_ok"
                << " nodes=" << graph.nodeCount()
                << " directed_edges=" << sample.directed_edge_count
                << " valid_edges=" << sample.valid_edge_count
                << " zero_length_edges="
                << sample.zero_length_edge_count
                << " samples=" << sample.sampleCount()
                << " directions_sha256="
                << hnswlib::edgeQuantV0Sha256Hex(matrix.sha256)
                << std::endl;
        } catch (...) {
            if (directions_published) {
                std::remove(options.output_directions.c_str());
            }
            throw;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v0_edge_sampler_error: "
                  << error.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }
#endif
}
