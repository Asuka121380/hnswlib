#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hnswlib/hnswlib.h"
#ifdef HNSWLIB_ENABLE_BASELINE_TRACE
#include "hnswlib/baseline_trace_writer.h"
#endif

namespace {

struct DatasetConfig {
    std::string dataset;
    std::string distance_kind;
    std::string base_path;
    std::string query_path;
    std::string ground_truth_path;
    std::string base_format;
    std::string query_format;
    std::string ground_truth_format;
    size_t dimension = 0;
    size_t n_base = 0;
    size_t n_query = 0;
    size_t ground_truth_k = 0;
};

struct Options {
    std::string dataset_config;
    std::string mode;
    std::string index_path;
    std::string output_dir;
    std::string run_id;
    size_t query_start = 0;
    size_t query_count = 0;
    size_t k = 10;
    size_t ef_search = 100;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t seed = 42;
    size_t threads = 1;
    size_t dco_sample_modulus = 1;
    size_t dco_sample_remainder = 0;
    size_t edge_samples = 1024;
    size_t warmup_queries = 100;
    size_t performance_repeats = 5;
    bool collect_geometry = true;
    bool collect_distance_timing = false;
};

struct EdgeSample {
    uint64_t key;
    uint64_t query_id;
    uint32_t current_node_id;
    uint32_t neighbor_id;
    double search_progress;
};

uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::string readText(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) throw std::runtime_error("Cannot read file: " + path);
    return buffer.str();
}

bool fileExists(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return static_cast<bool>(input);
}

std::string jsonString(const std::string& text, const std::string& key, bool required = true) {
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (std::regex_search(text, match, expression)) return match[1].str();
    if (required) throw std::runtime_error("Dataset manifest is missing string field: " + key);
    return std::string();
}

size_t jsonSize(const std::string& text, const std::string& key) {
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, expression)) {
        throw std::runtime_error("Dataset manifest is missing integer field: " + key);
    }
    return static_cast<size_t>(std::stoull(match[1].str()));
}

DatasetConfig loadDatasetConfig(const std::string& path) {
    const std::string text = readText(path);
    DatasetConfig config;
    config.dataset = jsonString(text, "dataset");
    config.distance_kind = jsonString(text, "distance_kind");
    config.base_path = jsonString(text, "base_path");
    config.query_path = jsonString(text, "query_path");
    config.ground_truth_path = jsonString(text, "ground_truth_path");
    config.base_format = jsonString(text, "base_format");
    config.query_format = jsonString(text, "query_format");
    config.ground_truth_format = jsonString(text, "ground_truth_format");
    config.dimension = jsonSize(text, "dimension");
    config.n_base = jsonSize(text, "n_base");
    config.n_query = jsonSize(text, "n_query");
    config.ground_truth_k = jsonSize(text, "ground_truth_k");
    if (config.distance_kind != "squared_l2_float32") {
        throw std::runtime_error("Only squared_l2_float32 is supported in the first real-data experiment");
    }
    if (config.base_format != "fvecs" && config.base_format != "bvecs") {
        throw std::runtime_error("base_format must be fvecs or bvecs");
    }
    if (config.query_format != "fvecs" && config.query_format != "bvecs") {
        throw std::runtime_error("query_format must be fvecs or bvecs");
    }
    if (config.ground_truth_format != "ivecs") {
        throw std::runtime_error("ground_truth_format must be ivecs");
    }
    if (config.dimension == 0 || config.n_base == 0 || config.n_query == 0 || config.ground_truth_k == 0) {
        throw std::runtime_error("Dataset manifest counts and dimensions must be positive");
    }
    return config;
}

size_t parseSize(const char* value, const std::string& name) {
    try {
        const std::string text(value);
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(text, &consumed);
        if (consumed != text.size()) throw std::runtime_error("trailing characters");
        return static_cast<size_t>(parsed);
    } catch (...) {
        throw std::runtime_error("Invalid value for " + name + ": " + value);
    }
}

bool parseBool(const char* value, const std::string& name) {
    const std::string text(value);
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    throw std::runtime_error("Invalid boolean for " + name + ": " + text);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const char* value = argv[++i];
        if (key == "--dataset-config") options.dataset_config = value;
        else if (key == "--mode") options.mode = value;
        else if (key == "--index-path") options.index_path = value;
        else if (key == "--output-dir") options.output_dir = value;
        else if (key == "--run-id") options.run_id = value;
        else if (key == "--query-start") options.query_start = parseSize(value, key);
        else if (key == "--query-count") options.query_count = parseSize(value, key);
        else if (key == "--k") options.k = parseSize(value, key);
        else if (key == "--ef-search") options.ef_search = parseSize(value, key);
        else if (key == "--M") options.M = parseSize(value, key);
        else if (key == "--ef-construction") options.ef_construction = parseSize(value, key);
        else if (key == "--seed") options.seed = parseSize(value, key);
        else if (key == "--threads") options.threads = parseSize(value, key);
        else if (key == "--dco-sample-modulus") options.dco_sample_modulus = parseSize(value, key);
        else if (key == "--dco-sample-remainder") options.dco_sample_remainder = parseSize(value, key);
        else if (key == "--edge-samples") options.edge_samples = parseSize(value, key);
        else if (key == "--warmup-queries") options.warmup_queries = parseSize(value, key);
        else if (key == "--performance-repeats") options.performance_repeats = parseSize(value, key);
        else if (key == "--collect-geometry") options.collect_geometry = parseBool(value, key);
        else if (key == "--collect-distance-timing") options.collect_distance_timing = parseBool(value, key);
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (options.dataset_config.empty() || options.mode.empty() || options.index_path.empty()) {
        throw std::runtime_error("--dataset-config, --mode, and --index-path are required");
    }
    if (options.mode != "build-index" && options.mode != "correctness" &&
        options.mode != "trace" && options.mode != "performance") {
        throw std::runtime_error("--mode must be build-index, correctness, trace, or performance");
    }
    if (options.output_dir.empty()) throw std::runtime_error("--output-dir is required");
    if (options.run_id.empty()) throw std::runtime_error("--run-id is required");
    if (options.k == 0 || options.threads == 0 || options.dco_sample_modulus == 0 || options.performance_repeats == 0) {
        throw std::runtime_error("k, threads, dco-sample-modulus, and performance-repeats must be positive");
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
    if (result != 0 && errno != EEXIST) throw std::runtime_error("Cannot create directory: " + path);
}

void makeDirectories(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    std::string current;
    if (!path.empty() && path[0] == '/') current = "/";
    std::istringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/')) {
        if (part.empty()) continue;
        if (!current.empty() && current[current.size() - 1] != '/') current += '/';
        current += part;
        makeOneDirectory(current);
    }
}

void readExact(std::ifstream& input, char* destination, size_t bytes, const std::string& path) {
    input.read(destination, static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("Truncated vector file: " + path);
    }
}

std::vector<float> loadVectors(
    const std::string& path,
    const std::string& format,
    size_t expected_dimension,
    size_t start,
    size_t count) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open vector file: " + path);
    const size_t component_bytes = format == "fvecs" ? sizeof(float) : sizeof(uint8_t);
    if (expected_dimension > (std::numeric_limits<size_t>::max() - sizeof(int32_t)) / component_bytes) {
        throw std::runtime_error("Vector record size overflow");
    }
    const size_t record_bytes = sizeof(int32_t) + expected_dimension * component_bytes;
    if (start > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) / record_bytes) {
        throw std::runtime_error("Vector seek offset overflow");
    }
    input.seekg(static_cast<std::streamoff>(start * record_bytes), std::ios::beg);
    if (!input) throw std::runtime_error("Cannot seek vector file: " + path);
    if (count > std::numeric_limits<size_t>::max() / expected_dimension) {
        throw std::runtime_error("Vector allocation size overflow");
    }
    std::vector<float> vectors(count * expected_dimension);
    std::vector<uint8_t> bytes(expected_dimension);
    for (size_t row = 0; row < count; ++row) {
        int32_t dimension = 0;
        readExact(input, reinterpret_cast<char*>(&dimension), sizeof(dimension), path);
        if (dimension <= 0 || static_cast<size_t>(dimension) != expected_dimension) {
            throw std::runtime_error("Inconsistent vector dimension in " + path + " at record " + std::to_string(start + row));
        }
        float* output = vectors.data() + row * expected_dimension;
        if (format == "fvecs") {
            readExact(input, reinterpret_cast<char*>(output), expected_dimension * sizeof(float), path);
        } else {
            readExact(input, reinterpret_cast<char*>(bytes.data()), expected_dimension, path);
            for (size_t column = 0; column < expected_dimension; ++column) output[column] = static_cast<float>(bytes[column]);
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
    if (!input) throw std::runtime_error("Cannot open ground-truth file: " + path);
    const size_t record_bytes = sizeof(int32_t) + expected_k * sizeof(int32_t);
    input.seekg(static_cast<std::streamoff>(start * record_bytes), std::ios::beg);
    if (!input) throw std::runtime_error("Cannot seek ground-truth file: " + path);
    std::vector<uint32_t> result(count * expected_k);
    for (size_t row = 0; row < count; ++row) {
        int32_t k = 0;
        readExact(input, reinterpret_cast<char*>(&k), sizeof(k), path);
        if (k <= 0 || static_cast<size_t>(k) != expected_k) {
            throw std::runtime_error("Inconsistent ground-truth K at record " + std::to_string(start + row));
        }
        for (size_t column = 0; column < expected_k; ++column) {
            int32_t label = -1;
            readExact(input, reinterpret_cast<char*>(&label), sizeof(label), path);
            if (label < 0) throw std::runtime_error("Negative ground-truth label");
            result[row * expected_k + column] = static_cast<uint32_t>(label);
        }
    }
    return result;
}

std::set<hnswlib::labeltype> labelsFromQueue(
    std::priority_queue<std::pair<float, hnswlib::labeltype> > queue) {
    std::set<hnswlib::labeltype> labels;
    while (!queue.empty()) {
        labels.insert(queue.top().second);
        queue.pop();
    }
    return labels;
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

double recallAtK(
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >& approximate,
    const uint32_t* ground_truth,
    size_t k) {
    const std::set<hnswlib::labeltype> approximate_labels = labelsFromQueue(approximate);
    size_t matches = 0;
    for (size_t i = 0; i < k; ++i) {
        if (approximate_labels.count(static_cast<hnswlib::labeltype>(ground_truth[i])) != 0) ++matches;
    }
    return static_cast<double>(matches) / static_cast<double>(k);
}

std::string indexManifestPath(const std::string& index_path) {
    return index_path + ".manifest.json";
}

void writeIndexManifest(const Options& options, const DatasetConfig& dataset, uint64_t build_ns) {
    std::ofstream out(indexManifestPath(options.index_path).c_str());
    if (!out) throw std::runtime_error("Cannot create index manifest");
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"dataset\": \"" << dataset.dataset << "\",\n"
        << "  \"distance_kind\": \"" << dataset.distance_kind << "\",\n"
        << "  \"dimension\": " << dataset.dimension << ",\n"
        << "  \"n_base\": " << dataset.n_base << ",\n"
        << "  \"M\": " << options.M << ",\n"
        << "  \"ef_construction\": " << options.ef_construction << ",\n"
        << "  \"seed\": " << options.seed << ",\n"
        << "  \"build_threads\": " << options.threads << ",\n"
        << "  \"build_latency_ns\": " << build_ns << "\n"
        << "}\n";
}

void validateIndexManifest(const Options& options, const DatasetConfig& dataset) {
    const std::string text = readText(indexManifestPath(options.index_path));
    if (jsonString(text, "dataset") != dataset.dataset ||
        jsonString(text, "distance_kind") != dataset.distance_kind ||
        jsonSize(text, "dimension") != dataset.dimension ||
        jsonSize(text, "n_base") != dataset.n_base ||
        jsonSize(text, "M") != options.M ||
        jsonSize(text, "ef_construction") != options.ef_construction ||
        jsonSize(text, "seed") != options.seed) {
        throw std::runtime_error("Index manifest does not match dataset or index parameters");
    }
}

void writeCommonMetadata(
    const Options& options,
    const DatasetConfig& dataset,
    size_t effective_query_count,
    const std::string& path,
    bool trace_enabled) {
    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("Cannot create metadata: " + path);
    const char* slurm_job = std::getenv("SLURM_JOB_ID");
    const char* slurm_array = std::getenv("SLURM_ARRAY_TASK_ID");
    out << "{\n"
        << "  \"schema_version\": 2,\n"
        << "  \"run_id\": \"" << options.run_id << "\",\n"
        << "  \"dataset\": \"" << dataset.dataset << "\",\n"
        << "  \"distance_kind\": \"" << dataset.distance_kind << "\",\n"
        << "  \"mode\": \"" << options.mode << "\",\n"
        << "  \"dimension\": " << dataset.dimension << ",\n"
        << "  \"n_base\": " << dataset.n_base << ",\n"
        << "  \"n_query_total\": " << dataset.n_query << ",\n"
        << "  \"query_start\": " << options.query_start << ",\n"
        << "  \"query_count\": " << effective_query_count << ",\n"
        << "  \"k\": " << options.k << ",\n"
        << "  \"ef_search\": " << options.ef_search << ",\n"
        << "  \"M\": " << options.M << ",\n"
        << "  \"ef_construction\": " << options.ef_construction << ",\n"
        << "  \"seed\": " << options.seed << ",\n"
        << "  \"threads\": " << options.threads << ",\n"
        << "  \"dco_sample_modulus\": " << options.dco_sample_modulus << ",\n"
        << "  \"dco_sample_remainder\": " << options.dco_sample_remainder << ",\n"
        << "  \"collect_geometry\": " << (options.collect_geometry ? "true" : "false") << ",\n"
        << "  \"collect_distance_timing\": " << (options.collect_distance_timing ? "true" : "false") << ",\n"
        << "  \"baseline_trace_enabled\": " << (trace_enabled ? "true" : "false") << ",\n"
        << "  \"slurm_job_id\": \"" << (slurm_job ? slurm_job : "") << "\",\n"
        << "  \"slurm_array_task_id\": \"" << (slurm_array ? slurm_array : "") << "\"\n"
        << "}\n";
}

void buildIndex(const Options& options, const DatasetConfig& dataset) {
    if (fileExists(options.index_path) || fileExists(indexManifestPath(options.index_path))) {
        throw std::runtime_error("Refusing to overwrite an existing index or index manifest");
    }
    const std::vector<float> base = loadVectors(
        dataset.base_path, dataset.base_format, dataset.dimension, 0, dataset.n_base);
    hnswlib::L2Space space(dataset.dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, dataset.n_base, options.M, options.ef_construction, options.seed);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(options.threads)
#endif
    for (long long i = 0; i < static_cast<long long>(dataset.n_base); ++i) {
        index.addPoint(base.data() + static_cast<size_t>(i) * dataset.dimension, static_cast<hnswlib::labeltype>(i));
    }
    const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    index.saveIndex(options.index_path);
    writeIndexManifest(options, dataset, elapsed);
    writeCommonMetadata(options, dataset, 0, options.output_dir + "/metadata.json", false);
    std::ofstream complete((options.output_dir + "/complete.json").c_str());
    complete << "{\"status\":\"complete\",\"mode\":\"build-index\"}\n";
    std::cout << "real_data_index_build_ok build_latency_ns=" << elapsed << std::endl;
}

#ifdef HNSWLIB_ENABLE_BASELINE_TRACE
void validateTrace(const hnswlib::BaselineTraceCollector& trace) {
    const hnswlib::BaselineQuerySummary& summary = trace.summary;
    if (summary.n_edge_scan != summary.n_duplicate + summary.n_unique_neighbor) {
        throw std::runtime_error("Trace invariant failed: edge scan accounting");
    }
    if (summary.n_unique_neighbor != summary.n_dist) {
        throw std::runtime_error("Trace invariant failed: distance accounting");
    }
    if (trace.config.dco_sample_modulus == 1 && trace.records.size() != summary.n_dist) {
        throw std::runtime_error("Trace invariant failed: full DCO row count");
    }
    for (size_t i = 0; i < trace.records.size(); ++i) {
        const hnswlib::BaselineDcoRecord& record = trace.records[i];
        if (record.threshold_valid_before) {
            const double expected = record.dist_qd - record.threshold_before;
            if (std::fabs(expected - record.absolute_margin) > 1e-9 * std::max(1.0, std::fabs(expected))) {
                throw std::runtime_error("Trace invariant failed: margin identity");
            }
        }
        if (record.geometry_valid) {
            const double reconstructed = record.dist_qc + record.edge_length_cd * record.edge_length_cd -
                2.0 * record.edge_dot_qcd;
            const double tolerance = 5e-4 * std::max(1.0, std::fabs(record.dist_qd));
            if (std::fabs(reconstructed - record.dist_qd) > tolerance) {
                throw std::runtime_error("Trace invariant failed: geometry identity");
            }
        }
    }
}

void considerEdgeSamples(
    const hnswlib::BaselineTraceCollector& trace,
    size_t limit,
    uint64_t seed,
    std::vector<EdgeSample>& samples) {
    if (limit == 0) return;
    for (size_t i = 0; i < trace.records.size(); ++i) {
        const hnswlib::BaselineDcoRecord& record = trace.records[i];
        if (!record.geometry_valid) continue;
        const uint64_t key = mix64(seed ^ mix64(record.query_id) ^ mix64(record.dco_index));
        EdgeSample sample{key, record.query_id, record.current_node_id, record.neighbor_id, record.search_progress_fraction};
        if (samples.size() < limit) {
            samples.push_back(sample);
            std::push_heap(samples.begin(), samples.end(), [](const EdgeSample& a, const EdgeSample& b) { return a.key < b.key; });
        } else if (key < samples.front().key) {
            std::pop_heap(samples.begin(), samples.end(), [](const EdgeSample& a, const EdgeSample& b) { return a.key < b.key; });
            samples.back() = sample;
            std::push_heap(samples.begin(), samples.end(), [](const EdgeSample& a, const EdgeSample& b) { return a.key < b.key; });
        }
    }
}

void writeEdgeSamples(
    const std::string& path,
    const std::vector<EdgeSample>& samples,
    const std::vector<float>& base,
    size_t dimension) {
    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("Cannot create edge sample file");
    out << "query_id,current_node_id,neighbor_id,search_progress_fraction,edge_length";
    for (size_t d = 0; d < dimension; ++d) out << ",dir_" << d;
    out << '\n';
    for (size_t i = 0; i < samples.size(); ++i) {
        const EdgeSample& sample = samples[i];
        const float* current = base.data() + static_cast<size_t>(sample.current_node_id) * dimension;
        const float* neighbor = base.data() + static_cast<size_t>(sample.neighbor_id) * dimension;
        double length_sq = 0.0;
        for (size_t d = 0; d < dimension; ++d) {
            const double delta = static_cast<double>(neighbor[d]) - current[d];
            length_sq += delta * delta;
        }
        const double length = std::sqrt(length_sq);
        if (length == 0.0) continue;
        out << sample.query_id << ',' << sample.current_node_id << ',' << sample.neighbor_id << ','
            << std::setprecision(17) << sample.search_progress << ',' << length;
        for (size_t d = 0; d < dimension; ++d) {
            out << ',' << (static_cast<double>(neighbor[d]) - current[d]) / length;
        }
        out << '\n';
    }
}

void runTrace(const Options& options, const DatasetConfig& dataset) {
    if (options.k > dataset.ground_truth_k) throw std::runtime_error("k exceeds ground-truth K");
    validateIndexManifest(options, dataset);
    const size_t count = options.query_count == 0 ? dataset.n_query - options.query_start : options.query_count;
    if (options.query_start > dataset.n_query || count > dataset.n_query - options.query_start) {
        throw std::runtime_error("Query range exceeds dataset query count");
    }
    if (count == 0) throw std::runtime_error("Query range is empty");
    const std::vector<float> base = loadVectors(dataset.base_path, dataset.base_format, dataset.dimension, 0, dataset.n_base);
    const std::vector<float> queries = loadVectors(
        dataset.query_path, dataset.query_format, dataset.dimension, options.query_start, count);
    const std::vector<uint32_t> truth = loadGroundTruth(
        dataset.ground_truth_path, dataset.ground_truth_k, options.query_start, count);
    for (size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] >= dataset.n_base) throw std::runtime_error("Ground-truth label exceeds base count");
    }
    hnswlib::L2Space space(dataset.dimension);
    hnswlib::HierarchicalNSW<float> index(&space, options.index_path);
    index.setEf(options.ef_search);
    std::ofstream query_out((options.output_dir + "/query_stats.csv").c_str());
    std::ofstream dco_out((options.output_dir + "/dco_trace.csv").c_str());
    if (!query_out || !dco_out) throw std::runtime_error("Cannot create trace output files");
    hnswlib::BaselineTraceCsvWriter::writeQueryHeader(query_out);
    hnswlib::BaselineTraceCsvWriter::writeDcoHeader(dco_out);
    std::vector<EdgeSample> edge_samples;
    double recall_sum = 0.0;
    uint64_t dco_sum = 0;
    for (size_t local = 0; local < count; ++local) {
        const uint64_t query_id = static_cast<uint64_t>(options.query_start + local);
        const float* query = queries.data() + local * dataset.dimension;
        std::priority_queue<std::pair<float, hnswlib::labeltype> > baseline;
        std::chrono::steady_clock::time_point baseline_start;
        std::chrono::steady_clock::time_point baseline_end;
        if (options.mode == "correctness") {
            baseline_start = std::chrono::steady_clock::now();
            baseline = index.searchKnn(query, options.k);
            baseline_end = std::chrono::steady_clock::now();
        }
        hnswlib::BaselineTraceConfig config;
        config.query_id = query_id;
        config.dimension = dataset.dimension;
        config.seed = options.seed;
        config.collect_geometry = options.collect_geometry;
        config.collect_distance_timing = options.collect_distance_timing;
        config.dco_sample_modulus = options.dco_sample_modulus;
        config.dco_sample_remainder = options.dco_sample_remainder;
        hnswlib::BaselineTraceCollector trace(config);
        const std::priority_queue<std::pair<float, hnswlib::labeltype> > traced =
            index.searchKnnWithTrace(query, options.k, trace);
        if (options.mode == "correctness" && !queuesExactlyEqual(baseline, traced)) {
            throw std::runtime_error("Tracing changed result for query " + std::to_string(query_id));
        }
        validateTrace(trace);
        trace.summary.recall_at_k = recallAtK(
            traced, truth.data() + local * dataset.ground_truth_k, options.k);
        if (options.mode == "correctness") {
            trace.summary.baseline_query_latency_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(baseline_end - baseline_start).count());
        }
        hnswlib::BaselineTraceCsvWriter::writeQuery(query_out, trace.summary);
        for (size_t i = 0; i < trace.records.size(); ++i) {
            hnswlib::BaselineTraceCsvWriter::writeDco(dco_out, trace.records[i]);
        }
        considerEdgeSamples(trace, options.edge_samples, options.seed, edge_samples);
        recall_sum += trace.summary.recall_at_k;
        dco_sum += trace.summary.n_dist;
    }
    query_out.flush();
    dco_out.flush();
    if (!query_out || !dco_out) throw std::runtime_error("Trace output write failed");
    writeEdgeSamples(options.output_dir + "/edge_directions.csv", edge_samples, base, dataset.dimension);
    writeCommonMetadata(options, dataset, count, options.output_dir + "/metadata.json", true);
    std::ofstream complete((options.output_dir + "/complete.json").c_str());
    complete << "{\"status\":\"complete\",\"query_start\":" << options.query_start
             << ",\"query_count\":" << count << "}\n";
    std::cout << "real_data_trace_ok mode=" << options.mode
              << " mean_recall=" << recall_sum / static_cast<double>(count)
              << " mean_n_dist=" << static_cast<double>(dco_sum) / static_cast<double>(count) << std::endl;
}
#endif

void runPerformance(const Options& options, const DatasetConfig& dataset) {
    if (options.k > dataset.ground_truth_k) throw std::runtime_error("k exceeds ground-truth K");
    validateIndexManifest(options, dataset);
    const size_t count = options.query_count == 0 ? dataset.n_query - options.query_start : options.query_count;
    if (options.query_start > dataset.n_query || count > dataset.n_query - options.query_start) {
        throw std::runtime_error("Query range exceeds dataset query count");
    }
    if (count == 0) throw std::runtime_error("Query range is empty");
    const std::vector<float> queries = loadVectors(
        dataset.query_path, dataset.query_format, dataset.dimension, options.query_start, count);
    const std::vector<uint32_t> truth = loadGroundTruth(
        dataset.ground_truth_path, dataset.ground_truth_k, options.query_start, count);
    hnswlib::L2Space space(dataset.dimension);
    hnswlib::HierarchicalNSW<float> index(&space, options.index_path);
    index.setEf(options.ef_search);
    const size_t warmups = std::min(options.warmup_queries, count);
    for (size_t i = 0; i < warmups; ++i) index.searchKnn(queries.data() + i * dataset.dimension, options.k);
    std::ofstream out((options.output_dir + "/performance.csv").c_str());
    if (!out) throw std::runtime_error("Cannot create performance.csv");
    out << "repeat,query_count,total_latency_ns,qps,mean_recall_at_k\n";
    for (size_t repeat = 0; repeat < options.performance_repeats; ++repeat) {
        double recall_sum = 0.0;
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        for (size_t local = 0; local < count; ++local) {
            const std::priority_queue<std::pair<float, hnswlib::labeltype> > result =
                index.searchKnn(queries.data() + local * dataset.dimension, options.k);
            recall_sum += recallAtK(result, truth.data() + local * dataset.ground_truth_k, options.k);
        }
        const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
        const double qps = static_cast<double>(count) * 1e9 / static_cast<double>(elapsed);
        out << repeat << ',' << count << ',' << elapsed << ',' << std::setprecision(17) << qps << ','
            << recall_sum / static_cast<double>(count) << '\n';
    }
    writeCommonMetadata(options, dataset, count, options.output_dir + "/metadata.json", false);
    std::ofstream complete((options.output_dir + "/complete.json").c_str());
    complete << "{\"status\":\"complete\",\"mode\":\"performance\"}\n";
    std::cout << "real_data_performance_ok" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const DatasetConfig dataset = loadDatasetConfig(options.dataset_config);
        if (fileExists(options.output_dir + "/complete.json")) {
            throw std::runtime_error("Refusing to overwrite a completed output directory");
        }
        makeDirectories(options.output_dir);
        if (options.mode == "build-index") {
            buildIndex(options, dataset);
        } else if (options.mode == "performance") {
            runPerformance(options, dataset);
        } else {
#ifdef HNSWLIB_ENABLE_BASELINE_TRACE
            runTrace(options, dataset);
#else
            throw std::runtime_error("correctness and trace modes require HNSWLIB_ENABLE_BASELINE_TRACE=ON");
#endif
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real_data_trace_runner failed: " << error.what() << std::endl;
        return 1;
    }
}
