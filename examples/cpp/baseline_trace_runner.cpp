#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "hnswlib/hnswlib.h"
#include "hnswlib/baseline_trace_writer.h"

namespace {

struct Options {
    std::string output_dir = "results/pipeline_validation/raw";
    size_t n_base = 2000;
    size_t n_query = 32;
    size_t dimension = 1024;
    size_t k = 10;
    size_t ef_search = 100;
    size_t M = 16;
    size_t ef_construction = 100;
    uint64_t seed = 42;
    size_t dco_sample_modulus = 1;
    size_t edge_samples = 256;
};

size_t parseSize(const char* value, const std::string& name) {
    try {
        return static_cast<size_t>(std::stoull(value));
    } catch (...) {
        throw std::runtime_error("Invalid value for " + name + ": " + value);
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const char* value = argv[++i];
        if (key == "--output-dir") options.output_dir = value;
        else if (key == "--n-base") options.n_base = parseSize(value, key);
        else if (key == "--n-query") options.n_query = parseSize(value, key);
        else if (key == "--dim") options.dimension = parseSize(value, key);
        else if (key == "--k") options.k = parseSize(value, key);
        else if (key == "--ef-search") options.ef_search = parseSize(value, key);
        else if (key == "--M") options.M = parseSize(value, key);
        else if (key == "--ef-construction") options.ef_construction = parseSize(value, key);
        else if (key == "--seed") options.seed = parseSize(value, key);
        else if (key == "--dco-sample-modulus") options.dco_sample_modulus = parseSize(value, key);
        else if (key == "--edge-samples") options.edge_samples = parseSize(value, key);
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (options.n_base == 0 || options.n_query == 0 || options.dimension == 0 || options.k == 0) {
        throw std::runtime_error("n-base, n-query, dim and k must be positive");
    }
    if (options.k > options.n_base) throw std::runtime_error("k must not exceed n-base");
    if (options.dco_sample_modulus == 0) throw std::runtime_error("dco-sample-modulus must be positive");
    return options;
}

void makeDirectory(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

template<typename Queue>
bool queuesExactlyEqual(Queue left, Queue right) {
    if (left.size() != right.size()) return false;
    while (!left.empty()) {
        if (left.top().first != right.top().first || left.top().second != right.top().second) return false;
        left.pop();
        right.pop();
    }
    return true;
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

std::priority_queue<std::pair<float, hnswlib::labeltype> > exactTopK(
    const float* query,
    const std::vector<float>& base,
    size_t n_base,
    size_t dimension,
    size_t k,
    hnswlib::DISTFUNC<float> distance,
    void* distance_param) {
    std::priority_queue<std::pair<float, hnswlib::labeltype> > result;
    for (size_t i = 0; i < n_base; ++i) {
        const float value = distance(query, base.data() + i * dimension, distance_param);
        if (result.size() < k || value < result.top().first) {
            result.push(std::make_pair(value, static_cast<hnswlib::labeltype>(i)));
            if (result.size() > k) result.pop();
        }
    }
    return result;
}

double recallAtK(
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >& approximate,
    const std::priority_queue<std::pair<float, hnswlib::labeltype> >& exact,
    size_t k) {
    const std::set<hnswlib::labeltype> approximate_labels = labelsFromQueue(approximate);
    const std::set<hnswlib::labeltype> exact_labels = labelsFromQueue(exact);
    size_t matches = 0;
    for (std::set<hnswlib::labeltype>::const_iterator it = approximate_labels.begin();
         it != approximate_labels.end(); ++it) {
        if (exact_labels.count(*it) != 0) ++matches;
    }
    return static_cast<double>(matches) / static_cast<double>(k);
}

void validateTrace(const hnswlib::BaselineTraceCollector& trace) {
    const hnswlib::BaselineQuerySummary& summary = trace.summary;
    if (summary.n_edge_scan != summary.n_duplicate + summary.n_unique_neighbor) {
        throw std::runtime_error("Trace invariant failed: edge_scan != duplicate + unique_neighbor");
    }
    if (summary.n_unique_neighbor != summary.n_dist) {
        throw std::runtime_error("Trace invariant failed: unique_neighbor != n_dist");
    }
    if (trace.config.dco_sample_modulus == 1 && trace.records.size() != summary.n_dist) {
        throw std::runtime_error("Trace invariant failed: full trace record count != n_dist");
    }
    for (size_t i = 0; i < trace.records.size(); ++i) {
        const hnswlib::BaselineDcoRecord& record = trace.records[i];
        if (record.threshold_valid_before) {
            const double expected_margin = record.dist_qd - record.threshold_before;
            if (std::fabs(expected_margin - record.absolute_margin) > 1e-9 * std::max(1.0, std::fabs(expected_margin))) {
                throw std::runtime_error("Trace invariant failed: absolute margin mismatch");
            }
        }
        if (record.geometry_valid) {
            const double reconstructed = record.dist_qc +
                record.edge_length_cd * record.edge_length_cd - 2.0 * record.edge_dot_qcd;
            const double tolerance = 5e-4 * std::max(1.0, std::fabs(record.dist_qd));
            if (std::fabs(reconstructed - record.dist_qd) > tolerance) {
                throw std::runtime_error("Trace invariant failed: edge geometry identity mismatch");
            }
        }
    }
}

void writeEdgeDirections(
    const Options& options,
    const hnswlib::HierarchicalNSW<float>& index,
    const std::vector<float>& base) {
    std::ofstream out((options.output_dir + "/edge_directions.csv").c_str());
    if (!out) throw std::runtime_error("Cannot create edge_directions.csv");
    out << "current_node_id,neighbor_id,edge_length";
    for (size_t d = 0; d < options.dimension; ++d) out << ",dir_" << d;
    out << '\n';

    size_t written = 0;
    for (hnswlib::tableint current = 0; current < options.n_base && written < options.edge_samples; ++current) {
        int* links = reinterpret_cast<int*>(index.get_linklist0(current));
        const size_t count = index.getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(links));
        for (size_t j = 1; j <= count && written < options.edge_samples; ++j) {
            const hnswlib::tableint neighbor = static_cast<hnswlib::tableint>(links[j]);
            double length_sq = 0.0;
            for (size_t d = 0; d < options.dimension; ++d) {
                const double delta = static_cast<double>(base[neighbor * options.dimension + d]) -
                    static_cast<double>(base[current * options.dimension + d]);
                length_sq += delta * delta;
            }
            const double length = std::sqrt(length_sq);
            if (length == 0.0) continue;
            out << current << ',' << neighbor << ',' << std::setprecision(17) << length;
            for (size_t d = 0; d < options.dimension; ++d) {
                const double delta = static_cast<double>(base[neighbor * options.dimension + d]) -
                    static_cast<double>(base[current * options.dimension + d]);
                out << ',' << delta / length;
            }
            out << '\n';
            ++written;
        }
    }
    if (written == 0) throw std::runtime_error("No edge directions were sampled");
}

}  // namespace

int main(int argc, char** argv) {
#ifndef HNSWLIB_ENABLE_BASELINE_TRACE
    std::cerr << "This runner requires HNSWLIB_ENABLE_BASELINE_TRACE" << std::endl;
    return 2;
#else
    try {
        const Options options = parseOptions(argc, argv);
        makeDirectory(options.output_dir);

        std::mt19937 rng(static_cast<uint32_t>(options.seed));
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::normal_distribution<float> noise(0.0f, 0.02f);
        std::vector<float> base(options.n_base * options.dimension);
        std::vector<float> queries(options.n_query * options.dimension);
        for (size_t i = 0; i < base.size(); ++i) base[i] = normal(rng);
        for (size_t q = 0; q < options.n_query; ++q) {
            const size_t source = (q * 37) % options.n_base;
            for (size_t d = 0; d < options.dimension; ++d) {
                queries[q * options.dimension + d] = base[source * options.dimension + d] + noise(rng);
            }
        }

        hnswlib::L2Space space(options.dimension);
        hnswlib::HierarchicalNSW<float> index(
            &space,
            options.n_base,
            options.M,
            options.ef_construction,
            static_cast<size_t>(options.seed));
        const std::chrono::steady_clock::time_point build_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < options.n_base; ++i) {
            index.addPoint(base.data() + i * options.dimension, static_cast<hnswlib::labeltype>(i));
        }
        const std::chrono::steady_clock::time_point build_end = std::chrono::steady_clock::now();
        const uint64_t build_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(build_end - build_start).count());
        index.setEf(options.ef_search);

        std::ofstream query_out((options.output_dir + "/query_stats.csv").c_str());
        std::ofstream dco_out((options.output_dir + "/dco_trace.csv").c_str());
        if (!query_out || !dco_out) throw std::runtime_error("Cannot create trace output files");
        hnswlib::BaselineTraceCsvWriter::writeQueryHeader(query_out);
        hnswlib::BaselineTraceCsvWriter::writeDcoHeader(dco_out);

        double recall_sum = 0.0;
        uint64_t distance_sum = 0;
        for (size_t q = 0; q < options.n_query; ++q) {
            const float* query = queries.data() + q * options.dimension;
            const std::chrono::steady_clock::time_point baseline_start = std::chrono::steady_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype> > baseline = index.searchKnn(query, options.k);
            const std::chrono::steady_clock::time_point baseline_end = std::chrono::steady_clock::now();

            hnswlib::BaselineTraceConfig config;
            config.query_id = q;
            config.requested_k = options.k;
            config.ef_search = options.ef_search;
            config.dimension = options.dimension;
            config.seed = options.seed;
            config.collect_geometry = true;
            config.collect_distance_timing = true;
            config.dco_sample_modulus = options.dco_sample_modulus;
            hnswlib::BaselineTraceCollector trace(config);
            std::priority_queue<std::pair<float, hnswlib::labeltype> > traced =
                index.searchKnnWithTrace(query, options.k, trace);

            if (!queuesExactlyEqual(baseline, traced)) {
                throw std::runtime_error("Tracing changed the search result for query " + std::to_string(q));
            }
            validateTrace(trace);

            const std::priority_queue<std::pair<float, hnswlib::labeltype> > exact = exactTopK(
                query,
                base,
                options.n_base,
                options.dimension,
                options.k,
                space.get_dist_func(),
                space.get_dist_func_param());
            trace.summary.recall_at_k = recallAtK(traced, exact, options.k);
            trace.summary.baseline_query_latency_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(baseline_end - baseline_start).count());
            recall_sum += trace.summary.recall_at_k;
            distance_sum += trace.summary.n_dist;
            hnswlib::BaselineTraceCsvWriter::writeQuery(query_out, trace.summary);
            for (size_t i = 0; i < trace.records.size(); ++i) {
                hnswlib::BaselineTraceCsvWriter::writeDco(dco_out, trace.records[i]);
            }
        }

        writeEdgeDirections(options, index, base);

        std::ofstream metadata((options.output_dir + "/metadata.json").c_str());
        if (!metadata) throw std::runtime_error("Cannot create metadata.json");
        metadata << "{\n"
                 << "  \"schema_version\": 1,\n"
                 << "  \"dataset\": \"synthetic_high_dim_validation\",\n"
                 << "  \"distance_kind\": \"squared_l2_float32\",\n"
                 << "  \"n_base\": " << options.n_base << ",\n"
                 << "  \"n_query\": " << options.n_query << ",\n"
                 << "  \"dimension\": " << options.dimension << ",\n"
                 << "  \"k\": " << options.k << ",\n"
                 << "  \"ef_search\": " << options.ef_search << ",\n"
                 << "  \"M\": " << options.M << ",\n"
                 << "  \"ef_construction\": " << options.ef_construction << ",\n"
                 << "  \"seed\": " << options.seed << ",\n"
                 << "  \"dco_sample_modulus\": " << options.dco_sample_modulus << ",\n"
                 << "  \"edge_samples\": " << options.edge_samples << ",\n"
                 << "  \"build_latency_ns\": " << build_ns << ",\n"
                 << "  \"baseline_trace_enabled\": true\n"
                 << "}\n";

        std::cout << "pipeline_runner_ok\n"
                  << "output_dir=" << options.output_dir << '\n'
                  << "n_base=" << options.n_base << " n_query=" << options.n_query
                  << " dimension=" << options.dimension << '\n'
                  << "mean_recall=" << recall_sum / static_cast<double>(options.n_query) << '\n'
                  << "mean_n_dist=" << static_cast<double>(distance_sum) / static_cast<double>(options.n_query)
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "baseline_trace_runner failed: " << error.what() << std::endl;
        return 1;
    }
#endif
}
