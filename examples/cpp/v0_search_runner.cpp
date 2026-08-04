#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "hnswlib/hnswlib.h"

namespace {

struct DatasetConfig {
    std::string dataset;
    std::string distance_kind;
    std::string query_path;
    std::string ground_truth_path;
    std::string query_format;
    std::string ground_truth_format;
    size_t dimension = 0;
    size_t n_base = 0;
    size_t n_query = 0;
    size_t ground_truth_k = 0;
};

struct Options {
    std::string dataset_config;
    std::string index_path;
    std::string sidecar_path;
    std::string output_dir;
    std::string mode;
    std::string run_id;
    std::string producer_git_commit;
    std::string git_branch;
    std::string working_tree_dirty;
    size_t query_start = 0;
    size_t query_count = 0;
    size_t k = 10;
    size_t ef_search = 200;
    size_t shadow_sample_modulus = 1024;
    size_t shadow_sample_remainder = 0;
};

struct Totals {
    uint64_t query_count = 0;
    uint64_t mismatch_queries = 0;
    uint64_t baseline_latency_ns = 0;
    uint64_t v0_latency_ns = 0;
    uint64_t bound_evaluated = 0;
    uint64_t bound_pruned = 0;
    uint64_t raw_prunable = 0;
    uint64_t oracle_prunable = 0;
    uint64_t exact_fallback = 0;
    uint64_t exact_only_fallback = 0;
    uint64_t exact_distance_saved = 0;
    uint64_t lower_bound_violation = 0;
    uint64_t false_prune = 0;
    uint64_t shadow_records_seen = 0;
    uint64_t shadow_records_written = 0;
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
    uint64_t cap_records_selected = 0;
    uint64_t cap_records_valid = 0;
    uint64_t cap_records_invalid = 0;
    uint64_t cap_certificate_failure = 0;
#endif
    double baseline_recall_sum = 0.0;
    double v0_recall_sum = 0.0;
};

uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20U) {
                    out << "\\u00" << std::hex << std::setw(2)
                        << std::setfill('0') << static_cast<unsigned int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string csvEscape(const std::string& value) {
    bool needs_quotes = false;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == ',' || value[i] == '"' ||
            value[i] == '\n' || value[i] == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) return value;
    std::string escaped;
    escaped.reserve(value.size() + 2U);
    escaped.push_back('"');
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '"') escaped.push_back('"');
        escaped.push_back(value[i]);
    }
    escaped.push_back('"');
    return escaped;
}

std::string readText(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("Cannot read file: " + path);
    }
    return buffer.str();
}

bool fileExists(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return static_cast<bool>(input);
}

uint64_t fileSize(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot inspect file: " + path);
    const std::ifstream::pos_type end = input.tellg();
    if (end < 0) throw std::runtime_error("Cannot determine file size: " + path);
    return static_cast<uint64_t>(end);
}

std::string jsonString(
    const std::string& text,
    const std::string& key) {
    const std::regex expression(
        "\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error(
            "Dataset manifest is missing string field: " + key);
    }
    return match[1].str();
}

size_t jsonSize(const std::string& text, const std::string& key) {
    const std::regex expression(
        "\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error(
            "Dataset manifest is missing integer field: " + key);
    }
    return static_cast<size_t>(std::stoull(match[1].str()));
}

DatasetConfig loadDatasetConfig(const std::string& path) {
    const std::string text = readText(path);
    DatasetConfig config;
    config.dataset = jsonString(text, "dataset");
    config.distance_kind = jsonString(text, "distance_kind");
    config.query_path = jsonString(text, "query_path");
    config.ground_truth_path = jsonString(text, "ground_truth_path");
    config.query_format = jsonString(text, "query_format");
    config.ground_truth_format = jsonString(text, "ground_truth_format");
    config.dimension = jsonSize(text, "dimension");
    config.n_base = jsonSize(text, "n_base");
    config.n_query = jsonSize(text, "n_query");
    config.ground_truth_k = jsonSize(text, "ground_truth_k");
    if (config.distance_kind != "squared_l2_float32") {
        throw std::runtime_error(
            "v0_search_runner only supports squared_l2_float32");
    }
    if (config.query_format != "fvecs" &&
        config.query_format != "bvecs") {
        throw std::runtime_error("query_format must be fvecs or bvecs");
    }
    if (config.ground_truth_format != "ivecs") {
        throw std::runtime_error("ground_truth_format must be ivecs");
    }
    if (config.dimension == 0 || config.n_base == 0 ||
        config.n_query == 0 || config.ground_truth_k == 0) {
        throw std::runtime_error(
            "Dataset manifest counts and dimension must be positive");
    }
    return config;
}

size_t parseSize(const char* value, const std::string& name) {
    try {
        const std::string text(value);
        size_t consumed = 0;
        const unsigned long long parsed =
            std::stoull(text, &consumed);
        if (consumed != text.size()) {
            throw std::runtime_error("trailing characters");
        }
        return static_cast<size_t>(parsed);
    } catch (...) {
        throw std::runtime_error(
            "Invalid value for " + name + ": " + value);
    }
}

void printUsage(std::ostream& out) {
    out
        << "Usage: v0_search_runner\n"
        << "  --dataset-config <dataset.json>\n"
        << "  --index-path <hnsw-index>\n"
        << "  --sidecar-path <v0meta>\n"
        << "  --output-dir <directory>\n"
        << "  --mode <correctness|shadow|prune>\n"
        << "  [--run-id <text>]\n"
        << "  [--query-start <non-negative-integer>]\n"
        << "  [--query-count <non-negative-integer; 0 means remaining>]\n"
        << "  [--k <positive-integer>]\n"
        << "  [--ef-search <positive-integer>]\n"
        << "  [--shadow-sample-modulus <positive-integer>]\n"
        << "  [--shadow-sample-remainder <non-negative-integer>]\n"
        << "  [--producer-git-commit <text>]\n"
        << "  [--git-branch <text>]\n"
        << "  [--working-tree-dirty <true|false|unknown>]\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            printUsage(std::cout);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        const char* value = argv[++i];
        if (key == "--dataset-config") options.dataset_config = value;
        else if (key == "--index-path") options.index_path = value;
        else if (key == "--sidecar-path") options.sidecar_path = value;
        else if (key == "--output-dir") options.output_dir = value;
        else if (key == "--mode") options.mode = value;
        else if (key == "--run-id") options.run_id = value;
        else if (key == "--query-start") {
            options.query_start = parseSize(value, key);
        } else if (key == "--query-count") {
            options.query_count = parseSize(value, key);
        } else if (key == "--k") {
            options.k = parseSize(value, key);
        } else if (key == "--ef-search") {
            options.ef_search = parseSize(value, key);
        } else if (key == "--shadow-sample-modulus") {
            options.shadow_sample_modulus = parseSize(value, key);
        } else if (key == "--shadow-sample-remainder") {
            options.shadow_sample_remainder = parseSize(value, key);
        } else if (key == "--producer-git-commit") {
            options.producer_git_commit = value;
        } else if (key == "--git-branch") {
            options.git_branch = value;
        } else if (key == "--working-tree-dirty") {
            options.working_tree_dirty = value;
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (options.dataset_config.empty() || options.index_path.empty() ||
        options.sidecar_path.empty() || options.output_dir.empty() ||
        options.mode.empty()) {
        throw std::runtime_error(
            "--dataset-config, --index-path, --sidecar-path, "
            "--output-dir, and --mode are required");
    }
    if (options.mode != "correctness" &&
        options.mode != "shadow" &&
        options.mode != "prune") {
        throw std::runtime_error(
            "--mode must be correctness, shadow, or prune");
    }
#ifndef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
    if (options.mode == "shadow") {
        throw std::runtime_error(
            "shadow mode requires HNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON");
    }
#endif
#ifndef HNSWLIB_ENABLE_V0_REAL_PRUNING
    if (options.mode == "prune") {
        throw std::runtime_error(
            "prune mode requires HNSWLIB_ENABLE_V0_REAL_PRUNING=ON");
    }
#endif
    if (options.k == 0 || options.ef_search == 0 ||
        options.shadow_sample_modulus == 0) {
        throw std::runtime_error(
            "k, ef-search, and shadow-sample-modulus must be positive");
    }
    if (options.shadow_sample_remainder >=
        options.shadow_sample_modulus) {
        throw std::runtime_error(
            "shadow-sample-remainder must be less than the modulus");
    }
    return options;
}

void makeOneDirectory(const std::string& path) {
    if (path.empty()) return;
#ifdef _WIN32
    const int result = _mkdir(path.c_str());
#else
    const int result = mkdir(path.c_str(), 0755);
#endif
    if (result != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
}

void makeDirectories(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    std::string current;
    size_t start = 0;
    if (!path.empty() && path[0] == '/') {
        current = "/";
        start = 1;
    } else if (path.size() >= 2 && path[1] == ':') {
        current = path.substr(0, 2);
        start = path.size() >= 3 && path[2] == '/' ? 3 : 2;
    }
    std::istringstream parts(path.substr(start));
    std::string part;
    while (std::getline(parts, part, '/')) {
        if (part.empty()) continue;
        if (!current.empty() && current[current.size() - 1] != '/') {
            current += '/';
        }
        current += part;
        makeOneDirectory(current);
    }
}

void readExact(
    std::ifstream& input,
    char* destination,
    size_t bytes,
    const std::string& path) {
    input.read(destination, static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("Truncated vector file: " + path);
    }
}

std::vector<float> loadVectors(
    const std::string& path,
    const std::string& format,
    size_t dimension,
    size_t start,
    size_t count) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open vector file: " + path);
    const size_t component_bytes =
        format == "fvecs" ? sizeof(float) : sizeof(uint8_t);
    if (dimension >
        (std::numeric_limits<size_t>::max() - sizeof(int32_t)) /
            component_bytes) {
        throw std::runtime_error("Vector record size overflow");
    }
    const size_t record_bytes =
        sizeof(int32_t) + dimension * component_bytes;
    if (start >
        static_cast<size_t>(
            std::numeric_limits<std::streamoff>::max()) /
            record_bytes) {
        throw std::runtime_error("Vector seek offset overflow");
    }
    input.seekg(
        static_cast<std::streamoff>(start * record_bytes),
        std::ios::beg);
    if (!input) throw std::runtime_error("Cannot seek vector file: " + path);
    if (count > std::numeric_limits<size_t>::max() / dimension) {
        throw std::runtime_error("Vector allocation size overflow");
    }
    std::vector<float> vectors(count * dimension);
    std::vector<uint8_t> bytes(dimension);
    for (size_t row = 0; row < count; ++row) {
        int32_t stored_dimension = 0;
        readExact(
            input,
            reinterpret_cast<char*>(&stored_dimension),
            sizeof(stored_dimension),
            path);
        if (stored_dimension <= 0 ||
            static_cast<size_t>(stored_dimension) != dimension) {
            throw std::runtime_error(
                "Inconsistent vector dimension at record " +
                std::to_string(start + row));
        }
        float* output = vectors.data() + row * dimension;
        if (format == "fvecs") {
            readExact(
                input,
                reinterpret_cast<char*>(output),
                dimension * sizeof(float),
                path);
        } else {
            readExact(
                input,
                reinterpret_cast<char*>(bytes.data()),
                dimension,
                path);
            for (size_t column = 0; column < dimension; ++column) {
                output[column] = static_cast<float>(bytes[column]);
            }
        }
    }
    return vectors;
}

std::vector<uint32_t> loadGroundTruth(
    const std::string& path,
    size_t expected_k,
    size_t start,
    size_t count) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Cannot open ground-truth file: " + path);
    }
    const size_t record_bytes =
        sizeof(int32_t) + expected_k * sizeof(int32_t);
    input.seekg(
        static_cast<std::streamoff>(start * record_bytes),
        std::ios::beg);
    if (!input) {
        throw std::runtime_error(
            "Cannot seek ground-truth file: " + path);
    }
    std::vector<uint32_t> result(count * expected_k);
    for (size_t row = 0; row < count; ++row) {
        int32_t stored_k = 0;
        readExact(
            input,
            reinterpret_cast<char*>(&stored_k),
            sizeof(stored_k),
            path);
        if (stored_k <= 0 ||
            static_cast<size_t>(stored_k) != expected_k) {
            throw std::runtime_error(
                "Inconsistent ground-truth K at record " +
                std::to_string(start + row));
        }
        for (size_t column = 0; column < expected_k; ++column) {
            int32_t label = -1;
            readExact(
                input,
                reinterpret_cast<char*>(&label),
                sizeof(label),
                path);
            if (label < 0) {
                throw std::runtime_error("Negative ground-truth label");
            }
            result[row * expected_k + column] =
                static_cast<uint32_t>(label);
        }
    }
    return result;
}

template<typename Queue>
bool queuesExactlyEqual(Queue left, Queue right) {
    if (left.size() != right.size()) return false;
    while (!left.empty()) {
        if (left.top() != right.top()) return false;
        left.pop();
        right.pop();
    }
    return true;
}

template<typename Queue>
std::set<hnswlib::labeltype> labelsFromQueue(Queue queue) {
    std::set<hnswlib::labeltype> labels;
    while (!queue.empty()) {
        labels.insert(queue.top().second);
        queue.pop();
    }
    return labels;
}

template<typename Queue>
double recallAtK(
    const Queue& result,
    const uint32_t* ground_truth,
    size_t k) {
    const std::set<hnswlib::labeltype> labels =
        labelsFromQueue(result);
    size_t matches = 0;
    for (size_t i = 0; i < k; ++i) {
        if (labels.count(
                static_cast<hnswlib::labeltype>(
                    ground_truth[i])) != 0) {
            ++matches;
        }
    }
    return static_cast<double>(matches) /
        static_cast<double>(k);
}

#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
class CsvShadowCollector :
    public hnswlib::V0ShadowValidationCollector {
 public:
    CsvShadowCollector(
        const std::string& path,
        const std::string& run_id,
        size_t ef_search,
        size_t modulus,
        size_t remainder,
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        const std::string& cap_path,
#endif
        Totals* totals)
        : output_(path.c_str()),
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
          cap_output_(cap_path.c_str()),
#endif
          run_id_(run_id),
          ef_search_(ef_search),
          modulus_(modulus),
          remainder_(remainder),
          totals_(totals) {
        if (!output_) {
            throw std::runtime_error(
                "Cannot create shadow_records.csv");
        }
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        if (!cap_output_) {
            throw std::runtime_error(
                "Cannot create cap_diagnostic_input.csv");
        }
        cap_output_
            << "cap_input_schema_version,run_id,query_id,current_node_id,"
            << "candidate_id,graph_layer,bound_status,ef_search,"
            << "current_squared_distance,threshold,edge_length,"
            << "direction_error,anchor_projection,current_lb,"
            << "exact_squared_distance,geometric_squared_distance,"
            << "current_would_prune,"
            << "raw_would_prune,oracle_would_prune,"
            << "reconstruction_norm,x_norm,x_dot_r,"
            << "true_edge_norm,x_dot_true_direction,"
            << "actual_direction_error,certificate_slack,"
            << "diagnostic_valid\n";
#endif
        output_
            << "schema_version,run_id,query_id,current_node_id,"
            << "candidate_id,graph_layer,bound_status,ef_search,"
            << "current_squared_distance,threshold,edge_length,"
            << "direction_error,anchor_projection,"
            << "anchor_projection_lower,"
            << "query_direction_inner_product_upper,"
            << "residual_direction_inner_product_upper,"
            << "length_squared_lower,cross_term_upper,"
            << "base_plus_length_lower,approximate_squared_distance,"
            << "current_distance_root_upper,direction_error_radius,"
            << "stored_numeric_padding,operational_l2_padding,"
            << "rounding_closure_padding,error_radius,lower_bound,"
            << "current_lb,shadow_exact_squared_distance,would_prune,"
            << "current_would_prune,oracle_would_prune,"
            << "lower_bound_valid,lower_bound_violation,false_prune,"
            << "cap_lb,cap_would_prune,blockwise_lb,"
            << "blockwise_would_prune,repr_lb_star,"
            << "repr_lb_star_would_prune\n";
    }

#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
    bool wantsSphericalCapDiagnostic(
        uint64_t query_id,
        uint64_t current_node_id,
        uint64_t candidate_id) const override {
        return isSampled(query_id, current_node_id, candidate_id);
    }
#endif

    void append(const hnswlib::V0ShadowRecord& record) {
        ++totals_->shadow_records_seen;
        const bool sampled = isSampled(
            record.query_id,
            record.current_node_id,
            record.candidate_id);
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        if (record.cap_diagnostic_selected) {
            ++totals_->cap_records_selected;
            if (record.cap_diagnostic_valid) {
                ++totals_->cap_records_valid;
                if (record.cap_certificate_slack < 0.0) {
                    ++totals_->cap_certificate_failure;
                }
            } else {
                ++totals_->cap_records_invalid;
            }
            cap_output_
                << 2U << ','
                << csvEscape(run_id_) << ','
                << record.query_id << ','
                << record.current_node_id << ','
                << record.candidate_id << ','
                << record.graph_layer << ','
                << static_cast<unsigned int>(record.bound_status) << ','
                << ef_search_ << ','
                << std::setprecision(17)
                << record.current_squared_distance << ','
                << record.threshold << ','
                << record.edge_length << ','
                << record.direction_error << ','
                << record.anchor_projection << ','
                << record.lower_bound << ','
                << record.shadow_exact_squared_distance << ','
                << record.cap_geometric_squared_distance << ','
                << (record.would_prune ? 1 : 0) << ','
                << (record.lower_bound_valid &&
                        record.approximate_squared_distance > record.threshold ?
                            1 : 0) << ','
                << (record.oracle_would_prune ? 1 : 0) << ','
                << record.cap_reconstruction_norm << ','
                << record.cap_x_norm << ','
                << record.cap_x_dot_reconstruction << ','
                << record.cap_true_edge_norm << ','
                << record.cap_x_dot_true_direction << ','
                << record.cap_actual_direction_error << ','
                << record.cap_certificate_slack << ','
                << (record.cap_diagnostic_valid ? 1 : 0) << '\n';
            if (!cap_output_) {
                throw std::runtime_error(
                    "Cannot write cap_diagnostic_input.csv");
            }
        }
#endif
        if (!sampled && !record.lower_bound_violation &&
            !record.false_prune) {
            return;
        }
        output_
            << hnswlib::V0_SHADOW_SCHEMA_VERSION << ','
            << csvEscape(run_id_) << ','
            << record.query_id << ','
            << record.current_node_id << ','
            << record.candidate_id << ','
            << record.graph_layer << ','
            << static_cast<unsigned int>(record.bound_status) << ','
            << ef_search_ << ','
            << std::setprecision(17)
            << record.current_squared_distance << ','
            << record.threshold << ','
            << record.edge_length << ','
            << record.direction_error << ','
            << record.anchor_projection << ','
            << record.anchor_projection_lower << ','
            << record.query_direction_inner_product_upper << ','
            << record.residual_direction_inner_product_upper << ','
            << record.length_squared_lower << ','
            << record.cross_term_upper << ','
            << record.base_plus_length_lower << ','
            << record.approximate_squared_distance << ','
            << record.current_distance_root_upper << ','
            << record.direction_error_radius << ','
            << record.stored_numeric_padding << ','
            << record.operational_l2_padding << ','
            << record.rounding_closure_padding << ','
            << record.error_radius << ','
            << record.lower_bound << ','
            << record.lower_bound << ','
            << record.shadow_exact_squared_distance << ','
            << (record.would_prune ? 1 : 0) << ','
            << (record.would_prune ? 1 : 0) << ','
            << (record.oracle_would_prune ? 1 : 0) << ','
            << (record.lower_bound_valid ? 1 : 0) << ','
            << (record.lower_bound_violation ? 1 : 0) << ','
            << (record.false_prune ? 1 : 0)
            << ",,,,,," << '\n';
        if (!output_) {
            throw std::runtime_error(
                "Cannot write shadow_records.csv");
        }
        ++totals_->shadow_records_written;
    }

    void flush() {
        output_.flush();
        if (!output_) {
            throw std::runtime_error(
                "Cannot flush shadow_records.csv");
        }
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        cap_output_.flush();
        if (!cap_output_) {
            throw std::runtime_error(
                "Cannot flush cap_diagnostic_input.csv");
        }
#endif
    }

 private:
    bool isSampled(
        uint64_t query_id,
        uint64_t current_node_id,
        uint64_t candidate_id) const {
        uint64_t key = mix64(query_id);
        key ^= mix64(current_node_id + 0x632be59bd9b4e019ULL);
        key ^= mix64(candidate_id + 0x8cb92baa3f3d8dd7ULL);
        return key % static_cast<uint64_t>(modulus_) ==
            static_cast<uint64_t>(remainder_);
    }

    std::ofstream output_;
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
    std::ofstream cap_output_;
#endif
    std::string run_id_;
    size_t ef_search_;
    size_t modulus_;
    size_t remainder_;
    Totals* totals_;
};
#endif

void addMetrics(
    Totals& totals,
    const hnswlib::V0QueryMetrics& metrics) {
    totals.bound_evaluated += metrics.bound_evaluated;
    totals.bound_pruned += metrics.bound_pruned;
    totals.raw_prunable += metrics.raw_prunable;
    totals.oracle_prunable += metrics.oracle_prunable;
    totals.exact_fallback += metrics.exact_fallback;
    totals.exact_only_fallback += metrics.exact_only_fallback;
    totals.exact_distance_saved += metrics.exact_distance_saved;
    totals.lower_bound_violation += metrics.lower_bound_violation;
    totals.false_prune += metrics.false_prune;
}

void writeMetadata(
    const Options& options,
    const DatasetConfig& dataset,
    const hnswlib::HierarchicalNSW<float>& index,
    size_t query_count) {
    const hnswlib::V0SidecarHeader& header =
        index.getEdgeQuantV0Metadata().header();
    std::ofstream out(
        (options.output_dir + "/metadata.json").c_str());
    if (!out) throw std::runtime_error("Cannot create metadata.json");
    out
        << "{\n"
        << "  \"format\": \"hnswlib_v0_search_run\",\n"
        << "  \"format_version\": 2,\n"
        << "  \"shadow_schema_version\": 2,\n"
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        << "  \"cap_input_schema_version\": 2,\n"
        << "  \"enabled_methods\": [\"current\", \"cap_phase1_export\"],\n"
#else
        << "  \"enabled_methods\": [\"current\"],\n"
#endif
        << "  \"validation_tolerance\": 0.0,\n"
        << "  \"dataset\": \"" << jsonEscape(dataset.dataset) << "\",\n"
        << "  \"mode\": \"" << jsonEscape(options.mode) << "\",\n"
        << "  \"run_id\": \"" << jsonEscape(options.run_id) << "\",\n"
        << "  \"dataset_config\": \""
        << jsonEscape(options.dataset_config) << "\",\n"
        << "  \"index_path\": \"" << jsonEscape(options.index_path)
        << "\",\n"
        << "  \"sidecar_path\": \"" << jsonEscape(options.sidecar_path)
        << "\",\n"
        << "  \"dimension\": " << dataset.dimension << ",\n"
        << "  \"n_base\": " << dataset.n_base << ",\n"
        << "  \"query_start\": " << options.query_start << ",\n"
        << "  \"query_count\": " << query_count << ",\n"
        << "  \"k\": " << options.k << ",\n"
        << "  \"ef_search\": " << options.ef_search << ",\n"
        << "  \"M_pq\": " << header.pq_m << ",\n"
        << "  \"nbits\": " << header.pq_nbits << ",\n"
        << "  \"code_bytes_per_edge\": " << header.pq_code_size << ",\n"
        << "  \"directed_edge_count\": "
        << header.directed_edge_count << ",\n"
        << "  \"index_sha256\": \""
        << index.getV0SerializedIndexFingerprintHex() << "\",\n"
        << "  \"sidecar_sha256\": \""
        << hnswlib::edgeQuantV0Sha256Hex(
               hnswlib::computeV0FileSha256(options.sidecar_path))
        << "\",\n"
        << "  \"codebook_sha256\": \""
        << hnswlib::edgeQuantV0Sha256Hex(header.codebook_sha256)
        << "\",\n"
        << "  \"sidecar_bytes\": " << fileSize(options.sidecar_path)
        << ",\n"
        << "  \"real_pruning_enabled\": "
        << (options.mode == "prune" ? "true" : "false")
        << ",\n"
        << "  \"bound_pruned_semantics\": "
        << (options.mode == "prune" ?
                "\"actual_prune\",\n" :
                "\"would_prune_observe_only\",\n")
        << "  \"shadow_sample_modulus\": "
        << options.shadow_sample_modulus << ",\n"
        << "  \"shadow_sample_remainder\": "
        << options.shadow_sample_remainder << ",\n"
        << "  \"producer_git_commit\": \""
        << jsonEscape(options.producer_git_commit) << "\",\n"
        << "  \"git_branch\": \"" << jsonEscape(options.git_branch)
        << "\",\n"
        << "  \"working_tree_dirty\": \""
        << jsonEscape(options.working_tree_dirty) << "\",\n"
#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
        << "  \"shadow_validation_compiled\": true,\n"
#else
        << "  \"shadow_validation_compiled\": false,\n"
#endif
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        << "  \"spherical_cap_diagnostic_compiled\": true,\n"
#else
        << "  \"spherical_cap_diagnostic_compiled\": false,\n"
#endif
#ifdef HNSWLIB_ENABLE_V0_REAL_PRUNING
        << "  \"real_pruning_compiled\": true\n"
#else
        << "  \"real_pruning_compiled\": false\n"
#endif
        << "}\n";
}

bool writeSummary(
    const Options& options,
    const Totals& totals) {
    const bool valid =
        totals.mismatch_queries == 0 &&
        totals.lower_bound_violation == 0 &&
        totals.false_prune == 0 &&
        totals.raw_prunable <= totals.bound_evaluated &&
        totals.oracle_prunable <= totals.bound_evaluated &&
        (options.mode == "prune" ||
            totals.exact_distance_saved == 0) &&
        (options.mode != "prune" ||
            totals.bound_pruned ==
                totals.exact_distance_saved) &&
        (options.mode != "shadow" || totals.bound_evaluated > 0);
    const double divisor =
        totals.query_count == 0 ?
            1.0 : static_cast<double>(totals.query_count);
    std::ofstream out(
        (options.output_dir + "/summary.json").c_str());
    if (!out) throw std::runtime_error("Cannot create summary.json");
    out
        << "{\n"
        << "  \"format\": \"hnswlib_v0_search_summary\",\n"
        << "  \"format_version\": 2,\n"
        << "  \"shadow_schema_version\": 2,\n"
        << "  \"status\": \"" << (valid ? "valid" : "invalid") << "\",\n"
        << "  \"query_count\": " << totals.query_count << ",\n"
        << "  \"mismatch_queries\": " << totals.mismatch_queries << ",\n"
        << "  \"mean_baseline_recall_at_k\": "
        << std::setprecision(17)
        << totals.baseline_recall_sum / divisor << ",\n"
        << "  \"mean_v0_recall_at_k\": "
        << totals.v0_recall_sum / divisor << ",\n"
        << "  \"baseline_latency_ns\": "
        << totals.baseline_latency_ns << ",\n"
        << "  \"v0_latency_ns\": " << totals.v0_latency_ns << ",\n"
        << "  \"bound_evaluated\": " << totals.bound_evaluated << ",\n"
        << "  \"bound_pruned\": " << totals.bound_pruned << ",\n"
        << "  \"raw_prunable\": " << totals.raw_prunable << ",\n"
        << "  \"oracle_prunable\": " << totals.oracle_prunable << ",\n"
        << "  \"exact_fallback\": " << totals.exact_fallback << ",\n"
        << "  \"exact_only_fallback\": "
        << totals.exact_only_fallback << ",\n"
        << "  \"exact_distance_saved\": "
        << totals.exact_distance_saved << ",\n"
        << "  \"lower_bound_violation\": "
        << totals.lower_bound_violation << ",\n"
        << "  \"false_prune\": " << totals.false_prune << ",\n"
        << "  \"shadow_records_seen\": "
        << totals.shadow_records_seen << ",\n"
        << "  \"shadow_records_written\": "
        << totals.shadow_records_written
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
        << ",\n"
        << "  \"cap_records_selected\": "
        << totals.cap_records_selected << ",\n"
        << "  \"cap_records_valid\": "
        << totals.cap_records_valid << ",\n"
        << "  \"cap_records_invalid\": "
        << totals.cap_records_invalid << ",\n"
        << "  \"cap_certificate_failure\": "
        << totals.cap_certificate_failure << "\n"
#else
        << "\n"
#endif
        << "}\n";
    return valid;
}

void run(const Options& options) {
    const DatasetConfig dataset =
        loadDatasetConfig(options.dataset_config);
    if (options.k > dataset.ground_truth_k) {
        throw std::runtime_error("k exceeds ground-truth K");
    }
    if (options.query_start > dataset.n_query) {
        throw std::runtime_error("query-start exceeds n_query");
    }
    const size_t count =
        options.query_count == 0 ?
            dataset.n_query - options.query_start :
            options.query_count;
    if (count > dataset.n_query - options.query_start ||
        count == 0) {
        throw std::runtime_error(
            "Query range is empty or exceeds the dataset");
    }
    makeDirectories(options.output_dir);
    if (fileExists(options.output_dir + "/complete.json")) {
        throw std::runtime_error(
            "Output directory already contains complete.json");
    }

    const std::vector<float> queries = loadVectors(
        dataset.query_path,
        dataset.query_format,
        dataset.dimension,
        options.query_start,
        count);
    const std::vector<uint32_t> truth = loadGroundTruth(
        dataset.ground_truth_path,
        dataset.ground_truth_k,
        options.query_start,
        count);
    for (size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] >= dataset.n_base) {
            throw std::runtime_error(
                "Ground-truth label exceeds n_base");
        }
    }

    hnswlib::L2Space space(dataset.dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, options.index_path);
    index.setEf(options.ef_search);
    index.loadEdgeQuantV0Metadata(options.sidecar_path);
    const hnswlib::V0SidecarHeader& header =
        index.getEdgeQuantV0Metadata().header();
    if (header.dimension != dataset.dimension ||
        header.node_count != dataset.n_base) {
        throw std::runtime_error(
            "Dataset manifest does not match the validated sidecar");
    }

    Totals totals;
    std::ofstream query_out(
        (options.output_dir + "/query_metrics.csv").c_str());
    if (!query_out) {
        throw std::runtime_error("Cannot create query_metrics.csv");
    }
    query_out
        << "query_id,baseline_recall_at_k,v0_recall_at_k,"
        << "results_equal,baseline_latency_ns,v0_latency_ns,"
        << "bound_evaluated,bound_pruned,raw_prunable,"
        << "oracle_prunable,exact_fallback,"
        << "exact_only_fallback,exact_distance_saved,"
        << "lower_bound_violation,false_prune\n";

#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
    std::unique_ptr<CsvShadowCollector> shadow_collector;
    if (options.mode == "shadow") {
        shadow_collector.reset(new CsvShadowCollector(
            options.output_dir + "/shadow_records.csv",
            options.run_id,
            options.ef_search,
            options.shadow_sample_modulus,
            options.shadow_sample_remainder,
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
            options.output_dir + "/cap_diagnostic_input.csv",
#endif
            &totals));
    }
#endif

    for (size_t local = 0; local < count; ++local) {
        const uint64_t query_id =
            static_cast<uint64_t>(options.query_start + local);
        const float* query =
            queries.data() + local * dataset.dimension;

        const std::chrono::steady_clock::time_point baseline_start =
            std::chrono::steady_clock::now();
        const std::priority_queue<
            std::pair<float, hnswlib::labeltype> > baseline =
            index.searchKnn(query, options.k);
        const std::chrono::steady_clock::time_point baseline_end =
            std::chrono::steady_clock::now();

        hnswlib::V0QueryMetrics metrics;
        const std::chrono::steady_clock::time_point v0_start =
            std::chrono::steady_clock::now();
        std::priority_queue<
            std::pair<float, hnswlib::labeltype> > v0;
#ifdef HNSWLIB_ENABLE_V0_REAL_PRUNING
        if (options.mode == "prune") {
            v0 = index.searchKnnV0Pruned(
                query,
                options.k,
                &metrics,
                nullptr);
        } else
#endif
        {
            v0 = index.searchKnnV0(
                query,
                options.k,
                &metrics,
                nullptr
#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
                , shadow_collector.get()
                , query_id
#endif
            );
        }
        const std::chrono::steady_clock::time_point v0_end =
            std::chrono::steady_clock::now();

        const uint64_t baseline_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                baseline_end - baseline_start).count());
        const uint64_t v0_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                v0_end - v0_start).count());
        const bool equal = queuesExactlyEqual(baseline, v0);
        const uint32_t* query_truth =
            truth.data() + local * dataset.ground_truth_k;
        const double baseline_recall =
            recallAtK(baseline, query_truth, options.k);
        const double v0_recall =
            recallAtK(v0, query_truth, options.k);

        ++totals.query_count;
        totals.mismatch_queries += equal ? 0U : 1U;
        totals.baseline_latency_ns += baseline_ns;
        totals.v0_latency_ns += v0_ns;
        totals.baseline_recall_sum += baseline_recall;
        totals.v0_recall_sum += v0_recall;
        addMetrics(totals, metrics);

        query_out
            << query_id << ','
            << std::setprecision(17) << baseline_recall << ','
            << v0_recall << ','
            << (equal ? 1 : 0) << ','
            << baseline_ns << ','
            << v0_ns << ','
            << metrics.bound_evaluated << ','
            << metrics.bound_pruned << ','
            << metrics.raw_prunable << ','
            << metrics.oracle_prunable << ','
            << metrics.exact_fallback << ','
            << metrics.exact_only_fallback << ','
            << metrics.exact_distance_saved << ','
            << metrics.lower_bound_violation << ','
            << metrics.false_prune << '\n';
        if (!query_out) {
            throw std::runtime_error(
                "Cannot write query_metrics.csv");
        }
    }
    query_out.flush();
    if (!query_out) {
        throw std::runtime_error("Cannot flush query_metrics.csv");
    }
#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
    if (shadow_collector) {
        shadow_collector->flush();
    }
#endif

    writeMetadata(options, dataset, index, count);
    const bool valid = writeSummary(options, totals);
    if (!valid) {
        throw std::runtime_error(
            "V0 validation gates failed; inspect summary.json");
    }
    std::ofstream complete(
        (options.output_dir + "/complete.json").c_str());
    if (!complete) {
        throw std::runtime_error("Cannot create complete.json");
    }
    complete
        << "{\"status\":\"complete\",\"mode\":\""
        << jsonEscape(options.mode)
        << "\",\"query_start\":" << options.query_start
        << ",\"query_count\":" << count << "}\n";
    if (!complete) {
        throw std::runtime_error("Cannot write complete.json");
    }
    std::cout
        << "v0_search_runner_ok"
        << " mode=" << options.mode
        << " queries=" << totals.query_count
        << " bound_evaluated=" << totals.bound_evaluated
        << (options.mode == "prune" ?
                " pruned=" : " would_prune=")
        << totals.bound_pruned
        << " exact_distance_saved="
        << totals.exact_distance_saved
        << " lower_bound_violation="
        << totals.lower_bound_violation
        << " false_prune=" << totals.false_prune
        << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parseOptions(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v0_search_runner: error: "
                  << error.what() << std::endl;
        return 1;
    }
}
