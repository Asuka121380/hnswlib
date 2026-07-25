#include <faiss/impl/ProductQuantizer.h>

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
#include "hnswlib/edge_quant_v0_codebook.h"
#include "hnswlib/edge_quant_v0_encoder.h"

namespace {

struct Options {
    std::string index_path;
    std::string base_path;
    std::string codebook_path;
    std::string output_sidecar;
    std::string output_metrics;
    std::string faiss_version;
    std::string faiss_source_commit;
    std::string producer_git_commit;
    size_t block_size;

    Options() : block_size(4096U) {}
};

void printUsage(const char* program) {
    std::cerr
        << "Usage: " << program << "\n"
        << "  --index-path <hnsw-index>\n"
        << "  [--base-path <base-vector-file-for-provenance>]\n"
        << "  --codebook-path <trained-v0pq>\n"
        << "  --output-sidecar <v0meta>\n"
        << "  --output-metrics <json>\n"
        << "  [--block-size <positive-integer>]\n"
        << "  [--faiss-version <text>]\n"
        << "  [--faiss-source-commit <text>]\n"
        << "  [--producer-git-commit <text>]\n";
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
        const std::string name(argv[i]);
        if (name.size() < 3U || name.substr(0U, 2U) != "--") {
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
        "--codebook-path",
        "--output-sidecar",
        "--output-metrics"
    };
    for (size_t i = 0U; i < sizeof(required) / sizeof(required[0]); ++i) {
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
            name != "--codebook-path" &&
            name != "--output-sidecar" &&
            name != "--output-metrics" &&
            name != "--block-size" &&
            name != "--faiss-version" &&
            name != "--faiss-source-commit" &&
            name != "--producer-git-commit") {
            throw std::invalid_argument(
                std::string("Unknown option ") + name);
        }
    }

    Options result;
    result.index_path = values["--index-path"];
    result.codebook_path = values["--codebook-path"];
    result.output_sidecar = values["--output-sidecar"];
    result.output_metrics = values["--output-metrics"];
    if (values.find("--base-path") != values.end()) {
        result.base_path = values["--base-path"];
    }
    if (values.find("--faiss-version") != values.end()) {
        result.faiss_version = values["--faiss-version"];
    }
    if (values.find("--faiss-source-commit") != values.end()) {
        result.faiss_source_commit =
            values["--faiss-source-commit"];
    }
    if (values.find("--producer-git-commit") != values.end()) {
        result.producer_git_commit =
            values["--producer-git-commit"];
    }
    if (values.find("--block-size") != values.end()) {
        const uint64_t parsed =
            parseUint64(values["--block-size"], "--block-size");
        if (parsed == 0U ||
            parsed > static_cast<uint64_t>(
                std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument(
                "--block-size must fit a positive size_t");
        }
        result.block_size = static_cast<size_t>(parsed);
    }
    if (result.index_path.empty() || result.codebook_path.empty() ||
        result.output_sidecar.empty() || result.output_metrics.empty()) {
        throw std::invalid_argument("V0 encoder paths must not be empty");
    }
    if (result.output_sidecar == result.output_metrics) {
        throw std::invalid_argument(
            "V0 sidecar and metrics paths must differ");
    }
    return result;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (size_t i = 0U; i < value.size(); ++i) {
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
                    output << "\\u"
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
    if (hnswlib::v0EncoderPathExists(path)) {
        throw std::runtime_error(
            "Refusing to overwrite existing V0 encoder metrics");
    }
    const std::string partial_path = path + ".partial";
    if (hnswlib::v0EncoderPathExists(partial_path)) {
        throw std::runtime_error(
            "V0 encoder metrics partial file already exists");
    }
    std::ofstream output(
        partial_path.c_str(),
        std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Failed to open V0 encoder metrics");
    }
    try {
        output.write(
            text.data(), static_cast<std::streamsize>(text.size()));
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Failed to finalize V0 encoder metrics");
        }
        output.close();
        if (std::rename(partial_path.c_str(), path.c_str()) != 0) {
            throw std::runtime_error(
                "Failed to atomically publish V0 encoder metrics");
        }
    } catch (...) {
        output.close();
        std::remove(partial_path.c_str());
        throw;
    }
}

class FaissV0Codec {
 public:
    explicit FaissV0Codec(const hnswlib::V0PQCodebook& codebook)
        : pq_(
              codebook.dimension,
              codebook.pq_m,
              codebook.pq_nbits) {
        if (pq_.code_size != codebook.code_size) {
            throw std::runtime_error(
                "Faiss code size does not match V0PQ codebook");
        }
        const size_t values_per_subquantizer =
            static_cast<size_t>(codebook.pq_ksub) *
            codebook.pq_dsub;
        for (uint32_t m = 0U; m < codebook.pq_m; ++m) {
            pq_.set_params(
                codebook.centroids.data() +
                    static_cast<size_t>(m) *
                        values_per_subquantizer,
                static_cast<int>(m));
        }
    }

    uint32_t dimension() const {
        return static_cast<uint32_t>(pq_.d);
    }

    uint32_t codeSize() const {
        return static_cast<uint32_t>(pq_.code_size);
    }

    void encode(
        const float* vectors,
        size_t count,
        uint8_t* codes) const {
        pq_.compute_codes(vectors, codes, count);
    }

    void decode(
        const uint8_t* codes,
        size_t count,
        float* vectors) const {
        pq_.decode(codes, vectors, count);
    }

 private:
    faiss::ProductQuantizer pq_;
};

std::string buildEncodingMetadata(
    const Options& options,
    const hnswlib::V0PQCodebook& codebook) {
    std::ostringstream json;
    json << "{"
         << "\"block_size\":" << options.block_size << ","
         << "\"codebook_file_sha256\":"
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(
                codebook.file_sha256)) << ","
         << "\"codebook_training_metadata\":";
    if (codebook.training_metadata_json.empty()) {
        json << "null";
    } else {
        json << codebook.training_metadata_json;
    }
    json << ",\"encoder\":\"v0_faiss_edge_encoder\""
         << ",\"faiss_source_commit\":"
         << jsonEscape(options.faiss_source_commit)
         << ",\"faiss_version\":"
         << jsonEscape(options.faiss_version)
         << ",\"producer_git_commit\":"
         << jsonEscape(options.producer_git_commit)
         << ",\"traversal_order\":"
         << "\"source_id_then_layer0_slot\""
         << "}";
    return json.str();
}

std::string buildMetrics(
    const Options& options,
    const hnswlib::V0PQCodebook& codebook,
    const hnswlib::V0AllEdgeEncodingMetrics& metrics,
    const hnswlib::V0Sha256Digest& index_sha,
    const hnswlib::V0Sha256Digest& adjacency_sha,
    const hnswlib::V0Sha256Digest* base_sha) {
    std::ostringstream json;
    json << std::setprecision(17)
         << "{\n"
         << "  \"M_pq\": " << codebook.pq_m << ",\n"
         << "  \"adjacency_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(adjacency_sha))
         << ",\n"
         << "  \"base_file_sha256\": ";
    if (base_sha == NULL) {
        json << "null,\n";
    } else {
        json << jsonEscape(
            hnswlib::edgeQuantV0Sha256Hex(*base_sha)) << ",\n";
    }
    json << "  \"block_count\": " << metrics.block_count << ",\n"
         << "  \"block_size\": " << options.block_size << ",\n"
         << "  \"codebook_file_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(
                codebook.file_sha256)) << ",\n"
         << "  \"codebook_path\": "
         << jsonEscape(options.codebook_path) << ",\n"
         << "  \"dimension\": " << codebook.dimension << ",\n"
         << "  \"directed_edge_count\": "
         << metrics.directed_edge_count << ",\n"
         << "  \"encoded_edge_count\": "
         << metrics.encoded_edge_count << ",\n"
         << "  \"exact_only_edge_count\": "
         << metrics.exact_only_edge_count << ",\n"
         << "  \"faiss_source_commit\": "
         << jsonEscape(options.faiss_source_commit) << ",\n"
         << "  \"faiss_version\": "
         << jsonEscape(options.faiss_version) << ",\n"
         << "  \"format\": \"hnswlib_v0_all_edge_encoding_metrics\",\n"
         << "  \"format_version\": 1,\n"
         << "  \"index_path\": "
         << jsonEscape(options.index_path) << ",\n"
         << "  \"index_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(index_sha))
         << ",\n"
         << "  \"max_direction_error\": "
         << metrics.max_direction_error << ",\n"
         << "  \"mean_direction_error\": "
         << metrics.mean_direction_error << ",\n"
         << "  \"nbits\": " << codebook.pq_nbits << ",\n"
         << "  \"node_count\": " << metrics.node_count << ",\n"
         << "  \"output_sidecar\": "
         << jsonEscape(options.output_sidecar) << ",\n"
         << "  \"producer_git_commit\": "
         << jsonEscape(options.producer_git_commit) << ",\n"
         << "  \"sidecar_bytes\": "
         << metrics.sidecar_bytes << ",\n"
         << "  \"sidecar_sha256\": "
         << jsonEscape(hnswlib::edgeQuantV0Sha256Hex(
                metrics.sidecar_sha256)) << ",\n"
         << "  \"status\": \"encoded\",\n"
         << "  \"zero_length_edge_count\": "
         << metrics.zero_length_edge_count << "\n"
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
        if (hnswlib::v0EncoderPathExists(options.output_metrics) ||
            hnswlib::v0EncoderPathExists(
                options.output_metrics + ".partial")) {
            throw std::runtime_error(
                "Refusing to overwrite existing V0 encoder metrics");
        }
        const hnswlib::V0PQCodebook codebook =
            hnswlib::loadV0PQCodebook(options.codebook_path);
        if (codebook.pq_nbits != 8U ||
            codebook.code_size != codebook.pq_m) {
            throw std::runtime_error(
                "Milestone 5 requires one 8-bit code per subquantizer");
        }

        hnswlib::L2Space space(codebook.dimension);
        hnswlib::HierarchicalNSW<float> index(
            &space, options.index_path);
        const hnswlib::V0Layer0GraphView graph =
            index.getV0Layer0GraphView();
        const hnswlib::V0Sha256Digest index_sha =
            hnswlib::computeV0FileSha256(options.index_path);
        const hnswlib::V0Sha256Digest adjacency_sha =
            graph.adjacencyFingerprint();
        hnswlib::V0Sha256Digest base_sha;
        const hnswlib::V0Sha256Digest* base_sha_ptr = NULL;
        if (!options.base_path.empty()) {
            base_sha =
                hnswlib::computeV0FileSha256(options.base_path);
            base_sha_ptr = &base_sha;
        }

        FaissV0Codec codec(codebook);
        hnswlib::V0AllEdgeEncodingSpec spec;
        spec.dimension = codebook.dimension;
        spec.pq_m = codebook.pq_m;
        spec.pq_nbits = codebook.pq_nbits;
        spec.pq_ksub = codebook.pq_ksub;
        spec.pq_dsub = codebook.pq_dsub;
        spec.block_size = options.block_size;
        spec.output_sidecar = options.output_sidecar;
        spec.training_metadata_json =
            buildEncodingMetadata(options, codebook);
        spec.codebook_centroids = codebook.centroids;
        spec.base_index_sha256 = index_sha;
        spec.adjacency_sha256 = adjacency_sha;

        const hnswlib::V0AllEdgeEncodingMetrics metrics =
            hnswlib::encodeV0AllLayer0Edges(graph, spec, codec);
        writeTextAtomically(
            options.output_metrics,
            buildMetrics(
                options,
                codebook,
                metrics,
                index_sha,
                adjacency_sha,
                base_sha_ptr));
        std::cout
            << "v0_faiss_edge_encoder_ok"
            << " nodes=" << metrics.node_count
            << " directed_edges=" << metrics.directed_edge_count
            << " encoded_edges=" << metrics.encoded_edge_count
            << " zero_length_edges="
            << metrics.zero_length_edge_count
            << " sidecar_sha256="
            << hnswlib::edgeQuantV0Sha256Hex(metrics.sidecar_sha256)
            << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v0_faiss_edge_encoder_error: "
                  << error.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    }
#endif
}
