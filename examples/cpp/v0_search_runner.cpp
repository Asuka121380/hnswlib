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
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "hnswlib/hnswlib.h"

#ifndef HNSWLIB_BUILD_TYPE
#define HNSWLIB_BUILD_TYPE "unknown"
#endif
#ifndef HNSWLIB_CXX_COMPILER_ID
#define HNSWLIB_CXX_COMPILER_ID "unknown"
#endif

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
    std::string ratio_selected_path;
    std::string ratio_calibrator_path;
    std::string ratio_operating_point_id;
    bool allow_diagnostic_operating_point = false;
    size_t query_start = 0;
    size_t query_count = 0;
    size_t warmup_query_count = 0;
    size_t repetition = 0;
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
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
    uint64_t ratio_bound_evaluated = 0;
    uint64_t ratio_eligible = 0;
    uint64_t ratio_bound_pruned = 0;
    uint64_t ratio_current_lb_fallback = 0;
    uint64_t ratio_current_lb_fallback_pruned = 0;
    uint64_t ratio_exact_fallback = 0;
    uint64_t ratio_invalid_fallback = 0;
    uint64_t ratio_exact_distance_saved = 0;
    uint64_t ratio_oracle_prunable = 0;
    uint64_t ratio_interval_violation = 0;
    uint64_t ratio_false_prune = 0;
    uint64_t ratio_shadow_records_seen = 0;
    uint64_t ratio_shadow_records_written = 0;
    uint64_t ratio_false_prune_query_exposure = 0;
    uint64_t visited_nodes = 0;
    uint64_t candidate_expansions = 0;
    uint64_t estimator_time_ns = 0;
    uint64_t exact_distance_time_ns = 0;
    uint64_t lost_ground_truth_neighbors = 0;
#endif
#ifdef HNSWLIB_ENABLE_V0_SPHERICAL_CAP_DIAGNOSTIC
    uint64_t cap_records_selected = 0;
    uint64_t cap_records_valid = 0;
    uint64_t cap_records_invalid = 0;
    uint64_t cap_certificate_failure = 0;
#endif
    double baseline_recall_sum = 0.0;
    double v0_recall_sum = 0.0;
    double baseline_recall_at_1_sum = 0.0;
    double v0_recall_at_1_sum = 0.0;
    double worst_v0_recall_at_k = 1.0;
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

uint64_t peakRssBytes() {
#ifdef _WIN32
    return 0U;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0U;
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024U;
#endif
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

bool parseBool(const char* value, const std::string& name) {
    const std::string text(value);
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    throw std::runtime_error(
        "Invalid boolean value for " + name + ": " + text);
}

void printUsage(std::ostream& out) {
    out
        << "Usage: v0_search_runner\n"
        << "  --dataset-config <dataset.json>\n"
        << "  --index-path <hnsw-index>\n"
        << "  --sidecar-path <v0meta>\n"
        << "  --output-dir <directory>\n"
        << "  --mode <correctness|shadow|prune|ratio-shadow|ratio-prune>\n"
        << "  [--ratio-selected-path <selected_operating_points.json>]\n"
        << "  [--ratio-calibrator-path <calibrator.json>]\n"
        << "  [--ratio-operating-point-id <id>]\n"
        << "  [--allow-diagnostic-operating-point <true|false>]\n"
        << "  [--run-id <text>]\n"
        << "  [--query-start <non-negative-integer>]\n"
        << "  [--query-count <non-negative-integer; 0 means remaining>]\n"
        << "  [--warmup-query-count <non-negative-integer>]\n"
        << "  [--repetition <non-negative-integer>]\n"
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
        } else if (key == "--warmup-query-count") {
            options.warmup_query_count = parseSize(value, key);
        } else if (key == "--repetition") {
            options.repetition = parseSize(value, key);
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
        } else if (key == "--ratio-selected-path") {
            options.ratio_selected_path = value;
        } else if (key == "--ratio-calibrator-path") {
            options.ratio_calibrator_path = value;
        } else if (key == "--ratio-operating-point-id") {
            options.ratio_operating_point_id = value;
        } else if (key == "--allow-diagnostic-operating-point") {
            options.allow_diagnostic_operating_point =
                parseBool(value, key);
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
        options.mode != "prune" &&
        options.mode != "ratio-shadow" &&
        options.mode != "ratio-prune") {
        throw std::runtime_error(
            "unsupported --mode value");
    }
#ifndef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
    if (options.mode == "shadow") {
        throw std::runtime_error(
            "shadow mode requires HNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON");
    }
#endif
#ifndef HNSWLIB_ENABLE_V0_RATIO_SHADOW
    if (options.mode == "ratio-shadow") {
        throw std::runtime_error(
            "ratio-shadow mode requires HNSWLIB_ENABLE_V0_RATIO_SHADOW=ON");
    }
#endif
#ifndef HNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING
    if (options.mode == "ratio-prune") {
        throw std::runtime_error(
            "ratio-prune mode requires HNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING=ON");
    }
#endif
    if ((options.mode == "ratio-shadow" ||
         options.mode == "ratio-prune") &&
        (options.ratio_selected_path.empty() ||
         options.ratio_calibrator_path.empty() ||
         options.ratio_operating_point_id.empty())) {
        throw std::runtime_error(
            "ratio modes require selected, calibrator, and operating-point paths/ID");
    }
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

template<typename Queue>
size_t matchCountAtK(
    const Queue& result,
    const uint32_t* ground_truth,
    size_t k) {
    const std::set<hnswlib::labeltype> labels =
        labelsFromQueue(result);
    size_t matches = 0U;
    for (size_t i = 0U; i < k; ++i) {
        matches += labels.count(
            static_cast<hnswlib::labeltype>(ground_truth[i])) != 0U ?
                1U : 0U;
    }
    return matches;
}

template<typename Queue>
double recallAtOne(const Queue& result, const uint32_t* ground_truth) {
    if (result.empty()) return 0.0;
    Queue copy = result;
    hnswlib::labeltype nearest = copy.top().second;
    while (!copy.empty()) {
        nearest = copy.top().second;
        copy.pop();
    }
    return nearest ==
        static_cast<hnswlib::labeltype>(ground_truth[0]) ? 1.0 : 0.0;
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

#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
class CsvRatioShadowCollector :
    public hnswlib::V0RatioShadowCollector {
 public:
    CsvRatioShadowCollector(
        const std::string& path,
        const std::string& run_id,
        size_t ef_search,
        size_t modulus,
        size_t remainder,
        Totals* totals)
        : output_(path.c_str()),
          run_id_(run_id),
          ef_search_(ef_search),
          modulus_(modulus),
          remainder_(remainder),
          totals_(totals) {
        if (!output_) {
            throw std::runtime_error(
                "Cannot create ratio_shadow_records.csv");
        }
        output_
            << "ratio_shadow_schema_version,run_id,query_id,current_node_id,"
            << "candidate_id,graph_layer,ef_search,current_squared_distance,"
            << "threshold,current_lb,current_lb_valid,current_would_prune,edge_length,"
            << "direction_error,reconstruction_norm,"
            << "x_dot_reconstruction,kappa_meta,ratio_eligible,"
            << "ratio_fallback_reason,rho_hat_raw,"
            << "rho_hat_ratio,ratio_estimated_squared_distance,ratio_quantile,"
            << "ratio_lb,ratio_effective_lb,ratio_used_current_fallback,"
            << "ratio_would_prune,exact_squared_distance,oracle_would_prune,"
            << "ratio_interval_violation,ratio_false_prune,calibrator_id\n";
    }

    void append(const hnswlib::V0RatioShadowRecord& record) override {
        ++totals_->ratio_shadow_records_seen;
        if (!isSampled(record) &&
            !record.ratio_interval_violation &&
            !record.ratio_false_prune) {
            return;
        }
        output_
            << hnswlib::V0_RATIO_SHADOW_SCHEMA_VERSION << ','
            << csvEscape(run_id_) << ','
            << record.query_id << ','
            << record.current_node_id << ','
            << record.candidate_id << ','
            << record.graph_layer << ','
            << ef_search_ << ','
            << std::setprecision(17)
            << record.current_squared_distance << ','
            << record.threshold << ','
            << record.current_lb << ','
            << (record.current_lb_valid ? 1 : 0) << ','
            << (record.current_would_prune ? 1 : 0) << ','
            << record.edge_length << ','
            << record.direction_error << ','
            << record.reconstruction_norm << ','
            << record.x_dot_reconstruction << ','
            << record.kappa_meta << ','
            << (record.ratio_eligible ? 1 : 0) << ','
            << csvEscape(record.ratio_fallback_reason) << ','
            << record.rho_hat_raw << ','
            << record.rho_hat_ratio << ','
            << record.ratio_estimated_squared_distance << ','
            << record.ratio_quantile << ','
            << record.ratio_lb << ','
            << record.ratio_effective_lb << ','
            << (record.ratio_used_current_fallback ? 1 : 0) << ','
            << (record.ratio_would_prune ? 1 : 0) << ','
            << record.exact_squared_distance << ','
            << (record.oracle_would_prune ? 1 : 0) << ','
            << (record.ratio_interval_violation ? 1 : 0) << ','
            << (record.ratio_false_prune ? 1 : 0) << ','
            << csvEscape(record.calibrator_id) << '\n';
        if (!output_) {
            throw std::runtime_error(
                "Cannot write ratio_shadow_records.csv");
        }
        ++totals_->ratio_shadow_records_written;
    }

    void flush() {
        output_.flush();
        if (!output_) {
            throw std::runtime_error(
                "Cannot flush ratio_shadow_records.csv");
        }
    }

 private:
    bool isSampled(const hnswlib::V0RatioShadowRecord& record) const {
        uint64_t key = mix64(record.query_id);
        key ^= mix64(record.current_node_id + 0x632be59bd9b4e019ULL);
        key ^= mix64(record.candidate_id + 0x8cb92baa3f3d8dd7ULL);
        return key % static_cast<uint64_t>(modulus_) ==
            static_cast<uint64_t>(remainder_);
    }

    std::ofstream output_;
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

#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
void addRatioMetrics(
    Totals& totals,
    const hnswlib::V0RatioQueryMetrics& metrics) {
    totals.ratio_bound_evaluated += metrics.ratio_bound_evaluated;
    totals.ratio_eligible += metrics.ratio_eligible;
    totals.ratio_bound_pruned += metrics.ratio_bound_pruned;
    totals.ratio_current_lb_fallback +=
        metrics.ratio_current_lb_fallback;
    totals.ratio_current_lb_fallback_pruned +=
        metrics.ratio_current_lb_fallback_pruned;
    totals.ratio_exact_fallback += metrics.ratio_exact_fallback;
    totals.ratio_invalid_fallback += metrics.ratio_invalid_fallback;
    totals.ratio_exact_distance_saved += metrics.exact_distance_saved;
    totals.ratio_oracle_prunable += metrics.oracle_prunable;
    totals.ratio_interval_violation +=
        metrics.ratio_interval_violation;
    totals.ratio_false_prune += metrics.ratio_false_prune;
    totals.ratio_false_prune_query_exposure +=
        metrics.ratio_false_prune != 0U ? 1U : 0U;
    totals.visited_nodes += metrics.visited_nodes;
    totals.candidate_expansions += metrics.candidate_expansions;
    totals.estimator_time_ns += metrics.estimator_time_ns;
    totals.exact_distance_time_ns += metrics.exact_distance_time_ns;
}
#endif

void writeMetadata(
    const Options& options,
    const DatasetConfig& dataset,
    const hnswlib::HierarchicalNSW<float>& index,
    size_t query_count
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
    , const hnswlib::V0RatioCalibrator* ratio_calibrator
#endif
    ) {
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
        << "  \"dataset_config_sha256\": \""
        << hnswlib::edgeQuantV0Sha256Hex(
               hnswlib::computeV0FileSha256(options.dataset_config))
        << "\",\n"
        << "  \"query_dataset_sha256\": \""
        << hnswlib::edgeQuantV0Sha256Hex(
               hnswlib::computeV0FileSha256(dataset.query_path))
        << "\",\n"
        << "  \"ground_truth_sha256\": \""
        << hnswlib::edgeQuantV0Sha256Hex(
               hnswlib::computeV0FileSha256(dataset.ground_truth_path))
        << "\",\n"
        << "  \"index_path\": \"" << jsonEscape(options.index_path)
        << "\",\n"
        << "  \"sidecar_path\": \"" << jsonEscape(options.sidecar_path)
        << "\",\n"
        << "  \"dimension\": " << dataset.dimension << ",\n"
        << "  \"n_base\": " << dataset.n_base << ",\n"
        << "  \"query_start\": " << options.query_start << ",\n"
        << "  \"query_count\": " << query_count << ",\n"
        << "  \"warmup_query_count\": "
        << std::min(options.warmup_query_count, query_count) << ",\n"
        << "  \"repetition\": " << options.repetition << ",\n"
        << "  \"evaluation_order\": \"alternating_by_query_and_repetition\",\n"
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
        << "  \"sidecar_bytes_per_edge\": "
        << (header.directed_edge_count == 0U ? 0.0 :
            static_cast<double>(fileSize(options.sidecar_path)) /
                static_cast<double>(header.directed_edge_count))
        << ",\n"
        << "  \"real_pruning_enabled\": "
        << (options.mode == "prune" || options.mode == "ratio-prune" ?
                "true" : "false")
        << ",\n"
        << "  \"bound_pruned_semantics\": "
        << (options.mode == "prune" || options.mode == "ratio-prune" ?
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
        << "  \"compiler_id\": \"" HNSWLIB_CXX_COMPILER_ID "\",\n"
        << "  \"compiler_version\": \"" << jsonEscape(__VERSION__)
        << "\",\n"
        << "  \"build_type\": \"" HNSWLIB_BUILD_TYPE "\",\n";
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
    if (ratio_calibrator != nullptr) {
        out
            << "  \"ratio_enabled\": true,\n"
            << "  \"ratio_shadow_schema_version\": "
            << hnswlib::V0_RATIO_SHADOW_SCHEMA_VERSION << ",\n"
            << "  \"ratio_selected_path\": \""
            << jsonEscape(ratio_calibrator->selectedPath()) << "\",\n"
            << "  \"ratio_selected_sha256\": \""
            << ratio_calibrator->selectedSha256() << "\",\n"
            << "  \"ratio_calibrator_path\": \""
            << jsonEscape(ratio_calibrator->calibratorPath()) << "\",\n"
            << "  \"ratio_calibrator_id\": \""
            << jsonEscape(ratio_calibrator->operatingPointId()) << "\",\n"
            << "  \"ratio_calibrator_sha256\": \""
            << ratio_calibrator->calibratorSha256() << "\",\n"
            << "  \"ratio_calibration_dataset_sha256\": \""
            << ratio_calibrator->calibrationDatasetSha256() << "\",\n"
            << "  \"ratio_query_split_manifest_sha256\": \""
            << ratio_calibrator->querySplitManifestSha256() << "\",\n"
            << "  \"ratio_estimator_formula_version\": \""
            << ratio_calibrator->formulaVersion() << "\",\n"
            << "  \"ratio_alpha\": " << std::setprecision(17)
            << ratio_calibrator->nominalAlpha() << ",\n"
            << "  \"ratio_calibration_level\": \""
            << ratio_calibrator->calibrationLevel() << "\",\n"
            << "  \"ratio_quantile_decimal\": \""
            << ratio_calibrator->quantileDecimal() << "\",\n"
            << "  \"ratio_quantile_hex\": \""
            << ratio_calibrator->quantileHex() << "\",\n"
            << "  \"ratio_kappa_min\": "
            << ratio_calibrator->kappaMin() << ",\n"
            << "  \"ratio_fallback_policy\": \""
            << ratio_calibrator->fallbackPolicy() << "\",\n"
            << "  \"ratio_diagnostic_only\": "
            << (ratio_calibrator->diagnosticOnly() ? "true" : "false")
            << ",\n"
            << "  \"ratio_trusted\": "
            << (ratio_calibrator->trusted() ? "true" : "false")
            << ",\n"
            << "  \"ratio_norm_lut_bytes\": "
            << index.getV0RatioCodebookNormLut().tableBytes() << ",\n";
    } else {
        out << "  \"ratio_enabled\": false,\n";
    }
#endif
    out
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
        << "  \"real_pruning_compiled\": true,\n"
#else
        << "  \"real_pruning_compiled\": false,\n"
#endif
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        << "  \"ratio_estimator_compiled\": true,\n"
#else
        << "  \"ratio_estimator_compiled\": false,\n"
#endif
#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
        << "  \"ratio_shadow_compiled\": true,\n"
#else
        << "  \"ratio_shadow_compiled\": false,\n"
#endif
#ifdef HNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING
        << "  \"ratio_real_pruning_compiled\": true\n"
#else
        << "  \"ratio_real_pruning_compiled\": false\n"
#endif
        << "}\n";
}

bool writeSummary(
    const Options& options,
    const Totals& totals) {
    bool valid = false;
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
    if (options.mode == "ratio-shadow") {
        valid =
            totals.mismatch_queries == 0U &&
            totals.ratio_bound_evaluated > 0U &&
            totals.ratio_bound_pruned <= totals.ratio_bound_evaluated &&
            totals.ratio_exact_distance_saved == 0U &&
            totals.ratio_shadow_records_seen ==
                totals.ratio_bound_evaluated;
    } else if (options.mode == "ratio-prune") {
        valid =
            totals.ratio_bound_evaluated > 0U &&
            totals.ratio_bound_pruned <= totals.ratio_bound_evaluated &&
            totals.ratio_bound_pruned ==
                totals.ratio_exact_distance_saved &&
            totals.ratio_shadow_records_seen == 0U;
    } else
#endif
    {
        valid =
            totals.mismatch_queries == 0 &&
            totals.lower_bound_violation == 0 &&
            totals.false_prune == 0 &&
            totals.raw_prunable <= totals.bound_evaluated &&
            totals.oracle_prunable <= totals.bound_evaluated &&
            (options.mode == "prune" ||
                totals.exact_distance_saved == 0) &&
            (options.mode != "prune" ||
                totals.bound_pruned == totals.exact_distance_saved) &&
            (options.mode != "shadow" || totals.bound_evaluated > 0);
    }
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
        << "  \"mean_baseline_recall_at_1\": "
        << totals.baseline_recall_at_1_sum / divisor << ",\n"
        << "  \"mean_v0_recall_at_1\": "
        << totals.v0_recall_at_1_sum / divisor << ",\n"
        << "  \"recall_at_k_loss_percentage_points\": "
        << 100.0 * (totals.baseline_recall_sum -
                    totals.v0_recall_sum) / divisor << ",\n"
        << "  \"mismatch_query_fraction\": "
        << static_cast<double>(totals.mismatch_queries) / divisor << ",\n"
        << "  \"worst_v0_recall_at_k\": "
        << totals.worst_v0_recall_at_k << ",\n"
        << "  \"baseline_latency_ns\": "
        << totals.baseline_latency_ns << ",\n"
        << "  \"v0_latency_ns\": " << totals.v0_latency_ns << ",\n"
        << "  \"peak_rss_bytes\": " << peakRssBytes() << ",\n"
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
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        << ",\n"
        << "  \"ratio_bound_evaluated\": "
        << totals.ratio_bound_evaluated << ",\n"
        << "  \"ratio_eligible\": " << totals.ratio_eligible << ",\n"
        << "  \"ratio_bound_pruned\": "
        << totals.ratio_bound_pruned << ",\n"
        << "  \"ratio_current_lb_fallback\": "
        << totals.ratio_current_lb_fallback << ",\n"
        << "  \"ratio_current_lb_fallback_pruned\": "
        << totals.ratio_current_lb_fallback_pruned << ",\n"
        << "  \"ratio_exact_fallback\": "
        << totals.ratio_exact_fallback << ",\n"
        << "  \"ratio_invalid_fallback\": "
        << totals.ratio_invalid_fallback << ",\n"
        << "  \"ratio_exact_distance_saved\": "
        << totals.ratio_exact_distance_saved << ",\n"
        << "  \"ratio_oracle_prunable\": "
        << totals.ratio_oracle_prunable << ",\n"
        << "  \"ratio_interval_violation\": "
        << totals.ratio_interval_violation << ",\n"
        << "  \"ratio_false_prune\": "
        << totals.ratio_false_prune << ",\n"
        << "  \"ratio_false_prune_query_exposure\": "
        << totals.ratio_false_prune_query_exposure << ",\n"
        << "  \"ratio_shadow_records_seen\": "
        << totals.ratio_shadow_records_seen << ",\n"
        << "  \"ratio_shadow_records_written\": "
        << totals.ratio_shadow_records_written << ",\n"
        << "  \"visited_nodes\": " << totals.visited_nodes << ",\n"
        << "  \"candidate_expansions\": "
        << totals.candidate_expansions << ",\n"
        << "  \"estimator_time_ns\": "
        << totals.estimator_time_ns << ",\n"
        << "  \"exact_distance_time_ns\": "
        << totals.exact_distance_time_ns << ",\n"
        << "  \"lost_ground_truth_neighbors\": "
        << totals.lost_ground_truth_neighbors
#endif
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

#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
    std::unique_ptr<hnswlib::V0RatioCalibrator> ratio_calibrator;
    if (options.mode == "ratio-shadow" || options.mode == "ratio-prune") {
        ratio_calibrator.reset(new hnswlib::V0RatioCalibrator(
            hnswlib::V0RatioCalibrator::load(
                options.ratio_selected_path,
                options.ratio_calibrator_path,
                options.ratio_operating_point_id)));
        if (options.mode == "ratio-prune" &&
            ratio_calibrator->diagnosticOnly() &&
            !options.allow_diagnostic_operating_point) {
            throw std::runtime_error(
                "Diagnostic-only ratio operating point requires "
                "--allow-diagnostic-operating-point true");
        }
    }
#endif

    Totals totals;
    std::ofstream query_out(
        (options.output_dir + "/query_metrics.csv").c_str());
    if (!query_out) {
        throw std::runtime_error("Cannot create query_metrics.csv");
    }
    query_out
        << "query_id,baseline_recall_at_k,v0_recall_at_k,"
        << "baseline_recall_at_1,v0_recall_at_1,"
        << "lost_ground_truth_neighbors,"
        << "results_equal,baseline_latency_ns,v0_latency_ns,"
        << "bound_evaluated,bound_pruned,raw_prunable,"
        << "oracle_prunable,exact_fallback,"
        << "exact_only_fallback,exact_distance_saved,"
        << "lower_bound_violation,false_prune,"
        << "ratio_bound_evaluated,ratio_eligible,ratio_bound_pruned,"
        << "ratio_current_lb_fallback,ratio_current_lb_fallback_pruned,"
        << "ratio_exact_fallback,ratio_invalid_fallback,"
        << "ratio_exact_distance_saved,ratio_oracle_prunable,"
        << "ratio_interval_violation,ratio_false_prune,"
        << "visited_nodes,candidate_expansions,estimator_time_ns,"
        << "exact_distance_time_ns\n";

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

#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
    std::unique_ptr<CsvRatioShadowCollector> ratio_shadow_collector;
    if (options.mode == "ratio-shadow") {
        ratio_shadow_collector.reset(new CsvRatioShadowCollector(
            options.output_dir + "/ratio_shadow_records.csv",
            options.run_id,
            options.ef_search,
            options.shadow_sample_modulus,
            options.shadow_sample_remainder,
            &totals));
    }
#endif

    const size_t warmup_count = std::min(options.warmup_query_count, count);
    for (size_t local = 0; local < warmup_count; ++local) {
        const float* query = queries.data() + local * dataset.dimension;
        index.searchKnn(query, options.k);
        hnswlib::V0QueryMetrics warmup_metrics;
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        hnswlib::V0RatioQueryMetrics ratio_warmup_metrics;
#endif
#ifdef HNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING
        if (options.mode == "ratio-prune") {
            index.searchKnnV0RatioPruned(
                query, options.k, *ratio_calibrator,
                &ratio_warmup_metrics, nullptr);
        } else
#endif
#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
        if (options.mode == "ratio-shadow") {
            index.searchKnnV0RatioShadow(
                query, options.k, *ratio_calibrator,
                &ratio_warmup_metrics, nullptr, nullptr, 0U);
        } else
#endif
#ifdef HNSWLIB_ENABLE_V0_REAL_PRUNING
        if (options.mode == "prune") {
            index.searchKnnV0Pruned(
                query, options.k, &warmup_metrics, nullptr);
        } else
#endif
        {
            index.searchKnnV0(query, options.k, &warmup_metrics, nullptr
#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
                , nullptr, 0U
#endif
            );
        }
    }

    for (size_t local = 0; local < count; ++local) {
        const uint64_t query_id =
            static_cast<uint64_t>(options.query_start + local);
        const float* query =
            queries.data() + local * dataset.dimension;

        hnswlib::V0QueryMetrics metrics;
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        hnswlib::V0RatioQueryMetrics ratio_metrics;
#endif
        typedef std::priority_queue<
            std::pair<float, hnswlib::labeltype> > ResultQueue;
        ResultQueue baseline;
        ResultQueue v0;
        std::chrono::steady_clock::time_point baseline_start;
        std::chrono::steady_clock::time_point baseline_end;
        std::chrono::steady_clock::time_point v0_start;
        std::chrono::steady_clock::time_point v0_end;
        const auto run_baseline = [&]() {
            baseline_start = std::chrono::steady_clock::now();
            baseline = index.searchKnn(query, options.k);
            baseline_end = std::chrono::steady_clock::now();
        };
        const auto run_v0 = [&]() {
            v0_start = std::chrono::steady_clock::now();
#ifdef HNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING
            if (options.mode == "ratio-prune") {
                v0 = index.searchKnnV0RatioPruned(
                    query, options.k, *ratio_calibrator,
                    &ratio_metrics, nullptr);
            } else
#endif
#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
            if (options.mode == "ratio-shadow") {
                v0 = index.searchKnnV0RatioShadow(
                    query, options.k, *ratio_calibrator,
                    &ratio_metrics, nullptr,
                    ratio_shadow_collector.get(), query_id);
            } else
#endif
#ifdef HNSWLIB_ENABLE_V0_REAL_PRUNING
            if (options.mode == "prune") {
                v0 = index.searchKnnV0Pruned(
                    query, options.k, &metrics, nullptr);
            } else
#endif
            {
                v0 = index.searchKnnV0(
                    query, options.k, &metrics, nullptr
#ifdef HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
                    , shadow_collector.get(), query_id
#endif
                );
            }
            v0_end = std::chrono::steady_clock::now();
        };
        if (((local + options.repetition) & 1U) == 0U) {
            run_baseline();
            run_v0();
        } else {
            run_v0();
            run_baseline();
        }

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
        const double baseline_recall_at_1 =
            recallAtOne(baseline, query_truth);
        const double v0_recall_at_1 = recallAtOne(v0, query_truth);
        const size_t baseline_match_count =
            matchCountAtK(baseline, query_truth, options.k);
        const size_t v0_match_count =
            matchCountAtK(v0, query_truth, options.k);
        const uint64_t lost_ground_truth_neighbors =
            baseline_match_count > v0_match_count ?
                static_cast<uint64_t>(baseline_match_count - v0_match_count) :
                0U;

        ++totals.query_count;
        totals.mismatch_queries += equal ? 0U : 1U;
        totals.baseline_latency_ns += baseline_ns;
        totals.v0_latency_ns += v0_ns;
        totals.baseline_recall_sum += baseline_recall;
        totals.v0_recall_sum += v0_recall;
        totals.baseline_recall_at_1_sum += baseline_recall_at_1;
        totals.v0_recall_at_1_sum += v0_recall_at_1;
        totals.worst_v0_recall_at_k =
            std::min(totals.worst_v0_recall_at_k, v0_recall);
        addMetrics(totals, metrics);
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        totals.lost_ground_truth_neighbors += lost_ground_truth_neighbors;
        addRatioMetrics(totals, ratio_metrics);
#endif

        query_out
            << query_id << ','
            << std::setprecision(17) << baseline_recall << ','
            << v0_recall << ','
            << baseline_recall_at_1 << ','
            << v0_recall_at_1 << ','
            << lost_ground_truth_neighbors << ','
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
            << metrics.false_prune << ','
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
            << ratio_metrics.ratio_bound_evaluated << ','
            << ratio_metrics.ratio_eligible << ','
            << ratio_metrics.ratio_bound_pruned << ','
            << ratio_metrics.ratio_current_lb_fallback << ','
            << ratio_metrics.ratio_current_lb_fallback_pruned << ','
            << ratio_metrics.ratio_exact_fallback << ','
            << ratio_metrics.ratio_invalid_fallback << ','
            << ratio_metrics.exact_distance_saved << ','
            << ratio_metrics.oracle_prunable << ','
            << ratio_metrics.ratio_interval_violation << ','
            << ratio_metrics.ratio_false_prune << ','
            << ratio_metrics.visited_nodes << ','
            << ratio_metrics.candidate_expansions << ','
            << ratio_metrics.estimator_time_ns << ','
            << ratio_metrics.exact_distance_time_ns << '\n';
#else
            << "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
#endif
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
#ifdef HNSWLIB_ENABLE_V0_RATIO_SHADOW
    if (ratio_shadow_collector) {
        ratio_shadow_collector->flush();
    }
#endif

    writeMetadata(options, dataset, index, count
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        , ratio_calibrator.get()
#endif
    );
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
        << " bound_evaluated="
#ifdef HNSWLIB_ENABLE_V0_RATIO_ESTIMATOR
        << ((options.mode == "ratio-shadow" || options.mode == "ratio-prune") ?
                totals.ratio_bound_evaluated : totals.bound_evaluated)
        << ((options.mode == "prune" || options.mode == "ratio-prune") ?
                " pruned=" : " would_prune=")
        << ((options.mode == "ratio-shadow" || options.mode == "ratio-prune") ?
                totals.ratio_bound_pruned : totals.bound_pruned)
        << " exact_distance_saved="
        << ((options.mode == "ratio-shadow" || options.mode == "ratio-prune") ?
                totals.ratio_exact_distance_saved : totals.exact_distance_saved)
        << " lower_bound_violation="
        << ((options.mode == "ratio-shadow" || options.mode == "ratio-prune") ?
                totals.ratio_interval_violation : totals.lower_bound_violation)
        << " false_prune="
        << ((options.mode == "ratio-shadow" || options.mode == "ratio-prune") ?
                totals.ratio_false_prune : totals.false_prune)
#else
        << totals.bound_evaluated
        << (options.mode == "prune" ? " pruned=" : " would_prune=")
        << totals.bound_pruned
        << " exact_distance_saved=" << totals.exact_distance_saved
        << " lower_bound_violation=" << totals.lower_bound_violation
        << " false_prune=" << totals.false_prune
#endif
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
