#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {

struct Options {
    std::string sidecar_path;
    std::string output_path;
    size_t iterations = 100000U;
    size_t warmup_iterations = 10000U;
    size_t working_set = 64U;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help") {
            std::cout << "v0_fast_kernel_microbenchmark --sidecar-path FILE "
                      << "--output FILE [--iterations N] "
                      << "[--warmup-iterations N] [--working-set N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value");
        const std::string value(argv[++i]);
        if (key == "--sidecar-path") options.sidecar_path = value;
        else if (key == "--output") options.output_path = value;
        else if (key == "--iterations")
            options.iterations = static_cast<size_t>(std::stoull(value));
        else if (key == "--warmup-iterations")
            options.warmup_iterations =
                static_cast<size_t>(std::stoull(value));
        else if (key == "--working-set")
            options.working_set = static_cast<size_t>(std::stoull(value));
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (options.sidecar_path.empty() || options.output_path.empty() ||
        options.iterations == 0U || options.warmup_iterations == 0U ||
        options.working_set == 0U) {
        throw std::invalid_argument("incomplete microbenchmark contract");
    }
    return options;
}

template<typename Callable>
double measureNsPerOperation(
    size_t iterations,
    size_t warmup_iterations,
    Callable callable) {
    for (size_t i = 0U; i < warmup_iterations; ++i) callable(i);
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    for (size_t i = 0U; i < iterations; ++i) callable(i);
    const std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());
    return static_cast<double>(elapsed) / static_cast<double>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const hnswlib::V0OwnedSidecar owned =
            hnswlib::loadV0Sidecar(options.sidecar_path);
        const hnswlib::V0SidecarView sidecar = owned.view();
        const hnswlib::V0SidecarHeader& header = sidecar.header();
        if (header.directed_edge_count == 0U) {
            throw std::runtime_error("sidecar has no edge records");
        }
        std::vector<float> native_codebook(
            static_cast<size_t>(header.codebook.size / sizeof(float)));
        for (size_t i = 0U; i < native_codebook.size(); ++i)
            native_codebook[i] = sidecar.codebookCentroid(i);
        const size_t working_set = std::min<size_t>(
            options.working_set,
            static_cast<size_t>(header.directed_edge_count));
        std::vector<hnswlib::V0EdgeRecordView> edges;
        edges.reserve(working_set);
        for (size_t i = 0U; i < working_set; ++i) {
            edges.push_back(sidecar.edgeRecordUnchecked(i));
        }
        std::vector<float> queries(working_set * header.dimension);
        std::vector<float> candidates(working_set * header.dimension);
        for (size_t item = 0U; item < working_set; ++item) {
            for (size_t d = 0U; d < header.dimension; ++d) {
                const size_t offset = item * header.dimension + d;
                queries[offset] = static_cast<float>(
                    static_cast<int>((d + item * 7U) % 31U) - 15) *
                    0.03125f;
                candidates[offset] = static_cast<float>(
                    static_cast<int>((d + item * 11U) % 17U) - 8) *
                    0.0625f;
            }
        }
        const float* query = queries.data();
        volatile double sink = 0.0;

        const size_t lut_iterations =
            std::max<size_t>(10U, options.iterations / 100U);
        const size_t lut_warmup_iterations =
            std::max<size_t>(10U, options.warmup_iterations / 100U);
        const double strict_lut_ns = measureNsPerOperation(
            lut_iterations, lut_warmup_iterations, [&](size_t i) {
                const float* current_query =
                    queries.data() + (i % working_set) * header.dimension;
                const hnswlib::V0QueryLut lut(current_query, sidecar);
                sink += static_cast<double>(lut.tableBytes());
            });
        const double fast_lut_ns = measureNsPerOperation(
            lut_iterations, lut_warmup_iterations, [&](size_t i) {
                const float* current_query =
                    queries.data() + (i % working_set) * header.dimension;
                const hnswlib::V0ApproxQueryLut lut(
                    current_query, header, native_codebook.data());
                sink += static_cast<double>(lut.tableBytes());
            });

        const hnswlib::V0QueryLut strict_lut(query, sidecar);
        const hnswlib::V0ApproxQueryLut fast_lut(
            query, header, native_codebook.data());
        const double strict_lookup_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                sink += strict_lut.innerProductUpper(edges[i % working_set]);
            });
        const double fast_lookup_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                sink += fast_lut.innerProductUnchecked(edges[i % working_set]);
            });

        const hnswlib::EdgeQuantV0QueryContext strict_context(
            query, sidecar);
        const hnswlib::EdgeQuantV0ApproxQueryContext fast_context(
            query, header, native_codebook.data());
        const double strict_estimator_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                sink += strict_context.evaluateRaw(
                    edges[i % working_set], 1000.0 + (i % 17U))
                    .approximate_squared_distance;
            });
        const double fast_estimator_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                sink += fast_context.evaluateRawFast(
                    edges[i % working_set], 1000.0 + (i % 17U))
                    .approximate_squared_distance;
            });

        hnswlib::L2Space exact_l2_space(header.dimension);
        const hnswlib::DISTFUNC<float> exact_l2 =
            exact_l2_space.get_dist_func();
        void* exact_l2_param = exact_l2_space.get_dist_func_param();
        const double exact_l2_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                const float* current_candidate = candidates.data() +
                    (i % working_set) * header.dimension;
                sink += exact_l2(query, current_candidate, exact_l2_param);
            });
        const double checked_record_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                const size_t edge_index = i %
                    static_cast<size_t>(header.directed_edge_count);
                sink += sidecar.edgeRecord(edge_index).edgeLength();
            });
        const double direct_record_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                const size_t edge_index = i %
                    static_cast<size_t>(header.directed_edge_count);
                sink += sidecar.edgeRecordUnchecked(edge_index)
                    .edgeLengthNativeUnchecked();
            });

        hnswlib::VisitedList state(1000000);
        const double state_reset_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t) {
                state.reset();
                sink += state.curV;
            });
        const double state_mark_ns = measureNsPerOperation(
            options.iterations, options.warmup_iterations, [&](size_t i) {
                state.approx_pruned_mass[i % state.numelements] = state.curV;
                sink += state.approx_pruned_mass[i % state.numelements];
            });

        std::ofstream output(options.output_path.c_str());
        if (!output) throw std::runtime_error("cannot create output JSON");
        output << std::setprecision(17)
            << "{\n"
            << "  \"schema_version\": 2,\n"
            << "  \"kernel\": \"raw_fast_v1\",\n"
            << "  \"dimension\": " << header.dimension << ",\n"
            << "  \"pq_m\": " << header.pq_m << ",\n"
            << "  \"pq_ksub\": " << header.pq_ksub << ",\n"
            << "  \"iterations\": " << options.iterations << ",\n"
            << "  \"warmup_iterations\": "
            << options.warmup_iterations << ",\n"
            << "  \"working_set\": " << working_set << ",\n"
            << "  \"strict_lut_build_ns\": " << strict_lut_ns << ",\n"
            << "  \"fast_lut_build_ns\": " << fast_lut_ns << ",\n"
            << "  \"strict_lookup_ns\": " << strict_lookup_ns << ",\n"
            << "  \"fast_lookup_ns\": " << fast_lookup_ns << ",\n"
            << "  \"strict_estimator_ns\": " << strict_estimator_ns << ",\n"
            << "  \"fast_estimator_ns\": " << fast_estimator_ns << ",\n"
            << "  \"exact_l2_ns\": " << exact_l2_ns << ",\n"
            << "  \"checked_record_ns\": " << checked_record_ns << ",\n"
            << "  \"direct_record_ns\": " << direct_record_ns << ",\n"
            << "  \"state_reset_ns\": " << state_reset_ns << ",\n"
            << "  \"state_mark_ns\": " << state_mark_ns << ",\n"
            << "  \"state_bytes_per_query\": "
            << (sizeof(hnswlib::vl_type) * state.numelements) << ",\n"
            << "  \"optimizer_sink\": " << sink << "\n"
            << "}\n";
        return output ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
}
