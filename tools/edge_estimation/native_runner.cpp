#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <filesystem>
#include <string>
#include <vector>

#include "event_format.h"
#include "evaluator.h"
#include "backends/pq_packed.h"
#include "backends/pq_packed_artifact.h"

namespace {

class IdentityReplayKernel {
 public:
    IdentityReplayKernel() : query_calls_(0), source_calls_(0) {}
    void prepareQuery(uint64_t) { ++query_calls_; }
    void prepareSource(const uq::EventRecord&) { ++source_calls_; }
    double score(const uq::EventRecord& event) {
        return event.d_current + event.threshold_before * 0.0;
    }
    uint64_t queryCalls() const { return query_calls_; }
    uint64_t sourceCalls() const { return source_calls_; }
 private:
    uint64_t query_calls_;
    uint64_t source_calls_;
};

void usage() {
    std::cerr << "usage:\n"
              << "  uq_native_runner capabilities\n"
              << "  uq_native_runner validate EVENTS [LABELS RANGES]\n"
              << "  uq_native_runner validate-artifact ARTIFACT EVENTS QUERIES\n"
              << "  uq_native_runner quality-pq EVENTS LABELS ARTIFACT QUERIES ALPHA\n"
              << "  uq_native_runner bench-pq EVENTS ARTIFACT QUERIES [REPEATS]\n"
              << "  uq_native_runner bench EVENTS [REPEATS]\n";
}

int run(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string command(argv[1]);
    if (command == "capabilities") {
        bool avx2 = false, avx512f = false;
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        __builtin_cpu_init();
        avx2 = __builtin_cpu_supports("avx2");
        avx512f = __builtin_cpu_supports("avx512f");
#endif
        std::cout
            << "{\"schema_version\":1,\"event_schema\":\"UQEV0001\","
            << "\"numeric_profile\":\"reference\","
            << "\"cpu\":{\"avx2\":" << (avx2 ? "true" : "false")
            << ",\"avx512f\":" << (avx512f ? "true" : "false") << "},"
#ifdef HNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE
            << "\"capture_compiled\":true,"
#else
            << "\"capture_compiled\":false,"
#endif
            << "\"backends\":["
            << "{\"name\":\"pq_packed\",\"available\":true,"
            << "\"supports_scalar\":true,\"layouts\":[\"packed4\",\"packed_nbits\"]},"
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
            << "{\"name\":\"pq_legacy\",\"available\":true,\"requires_artifact\":true},"
#else
            << "{\"name\":\"pq_legacy\",\"available\":false,\"reason\":\"not-compiled\"},"
#endif
#ifdef HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR
            << "{\"name\":\"pq_qjl_legacy\",\"available\":true,\"requires_artifact\":true},"
#else
            << "{\"name\":\"pq_qjl_legacy\",\"available\":false,\"reason\":\"not-compiled\"},"
#endif
            << "{\"name\":\"opq\",\"available\":false,\"reason\":\"faiss-adapter-not-compiled\"},"
            << "{\"name\":\"prq\",\"available\":false,\"reason\":\"faiss-adapter-not-compiled\"},"
            << "{\"name\":\"jq\",\"available\":false,\"reason\":\"jq-source-not-imported\"},"
            << "{\"name\":\"rabitq\",\"available\":false,\"reason\":\"external-dependency-not-compiled\"},"
            << "{\"name\":\"saq\",\"available\":false,\"reason\":\"external-dependency-or-isa-unavailable\"}]"
            << "}\n";
        return 0;
    }
    if (command == "validate") {
        if (argc != 3 && argc != 5) { usage(); return 2; }
        uq::Header event_header;
        const std::vector<uq::EventRecord> events = uq::readEvents(argv[2], &event_header);
        size_t label_count = 0U;
        size_t range_count = 0U;
        if (argc == 5) {
            uq::Header label_header, range_header;
            const std::vector<uq::LabelRecord> labels = uq::readLabels(argv[3], &label_header);
            const std::vector<uq::QueryRangeRecord> ranges = uq::readQueryRanges(argv[4], &range_header);
            if (event_header.identity != label_header.identity ||
                event_header.identity != range_header.identity ||
                event_header.dimension != label_header.dimension ||
                event_header.dimension != range_header.dimension)
                throw std::runtime_error("dataset identity mismatch");
            uq::validateDataset(events, labels, ranges);
            label_count = labels.size(); range_count = ranges.size();
        }
        std::cout << "{\"valid\":true,\"event_count\":" << events.size()
                  << ",\"label_count\":" << label_count
                  << ",\"query_count\":" << range_count << "}\n";
        return 0;
    }
    if (command == "validate-artifact") {
        if (argc != 5) { usage(); return 2; }
        uq::Header header; const std::vector<uq::EventRecord> events = uq::readEvents(argv[3], &header);
        uq::PackedPqArtifactKernel kernel(argv[2], argv[4], header);
        if (!events.empty()) {
            for (const uq::EventRecord& event : events)
                if (event.kind == uq::EventKind::QueryBegin) { kernel.prepareQuery(event.query_id); break; }
        }
        std::cout << "{\"valid\":true,\"backend\":\"pq_packed\",\"backend_bytes\":"
                  << kernel.backendBytes() << "}\n";
        return 0;
    }
    if (command == "quality-pq") {
        if (argc != 7) { usage(); return 2; }
        uq::Header event_header, label_header;
        const std::vector<uq::EventRecord> events = uq::readEvents(argv[2], &event_header);
        const std::vector<uq::LabelRecord> labels = uq::readLabels(argv[3], &label_header);
        if (event_header.identity != label_header.identity) throw std::runtime_error("label identity mismatch");
        uq::PackedPqArtifactKernel kernel(argv[4], argv[5], event_header);
        const double alpha = std::stod(argv[6]);
        const uq::QualityCounters value = uq::evaluateQuality(events, labels, alpha, kernel);
        std::cout << "{\"schema_version\":1,\"backend\":\"pq_packed\",\"alpha\":" << alpha
                  << ",\"event_count_u\":" << value.event_count_u
                  << ",\"decision_count_s\":" << value.decision_count_s
                  << ",\"valid_estimate_count\":" << value.valid_estimate_count
                  << ",\"fallback_count\":" << value.fallback_count
                  << ",\"tp\":" << value.tp << ",\"fp\":" << value.fp
                  << ",\"fn\":" << value.fn << ",\"tn\":" << value.tn << "}\n";
        return 0;
    }
    if (command == "bench-pq") {
        if (argc != 5 && argc != 6) { usage(); return 2; }
        const size_t repeats = argc == 6 ? static_cast<size_t>(std::stoull(argv[5])) : 1U;
        uq::Header header; const std::vector<uq::EventRecord> events = uq::readEvents(argv[2], &header);
        uq::PackedPqArtifactKernel warmup(argv[3], argv[4], header);
        (void)uq::replayOrderedEstimator(events, warmup, 1U);
        std::vector<uq::ReplayCounters> runs;
        uint64_t backend_bytes = 0U, scratch_bytes = 0U;
        for (size_t i = 0; i < repeats; ++i) {
            uq::PackedPqArtifactKernel kernel(argv[3], argv[4], header);
            runs.push_back(uq::replayOrderedEstimator(events, kernel, 1U));
            backend_bytes = kernel.backendBytes(); scratch_bytes = kernel.scratchBytes();
        }
        std::cout << "{\"schema_version\":1,\"backend\":\"pq_packed\",\"mode\":\"ordered_estimator\","
                  << "\"quality_valid\":true,\"raw_elapsed_ns\":[";
        for (size_t i = 0; i < runs.size(); ++i) { if (i) std::cout << ','; std::cout << runs[i].elapsed_ns; }
        const uq::ReplayCounters& value = runs.at(0);
        std::cout << "],\"query_count\":" << value.query_count
                  << ",\"source_setup_calls\":" << value.source_count
                  << ",\"eligible_events\":" << value.eligible_count
                  << ",\"fallback_count\":" << value.fallback_count
                  << ",\"checksum\":" << value.checksum
                  << ",\"memory_report\":{\"backend_bytes\":" << backend_bytes
                  << ",\"scratch_bytes\":" << scratch_bytes << ",\"peak_rss_bytes\":null}}\n";
        return 0;
    }
    if (command == "bench") {
        if (argc != 3 && argc != 4) { usage(); return 2; }
        const size_t repeats = argc == 4 ?
            static_cast<size_t>(std::strtoull(argv[3], NULL, 10)) : 1U;
        const std::vector<uq::EventRecord> events = uq::readEvents(argv[2]);
        if (repeats == 0U) throw std::invalid_argument("repeats must be positive");
        IdentityReplayKernel warmup_kernel;
        (void)uq::replayOrderedEstimator(events, warmup_kernel, 1U);
        std::vector<uq::ReplayCounters> runs;
        runs.reserve(repeats);
        for (size_t repeat = 0; repeat < repeats; ++repeat) {
            IdentityReplayKernel kernel;
            runs.push_back(uq::replayOrderedEstimator(events, kernel, 1U));
        }
        const uq::ReplayCounters& counters = runs[0];
        std::cout << "{\"schema_version\":1,\"mode\":\"ordered_estimator\","
                  << "\"quality_valid\":false,\"repeats\":" << repeats
                  << ",\"warmup_passes\":1,\"raw_elapsed_ns\":[";
        for (size_t i = 0; i < runs.size(); ++i) {
            if (i != 0U) std::cout << ',';
            std::cout << runs[i].elapsed_ns;
        }
        std::cout << "]"
                  << ",\"query_count\":" << counters.query_count
                  << ",\"source_setup_calls\":" << counters.source_count
                  << ",\"eligible_events\":" << counters.eligible_count
                  << ",\"fallback_count\":" << counters.fallback_count
                  << ",\"checksum\":" << counters.checksum
                  << ",\"memory_report\":{\"backend_bytes\":0,"
                  << "\"mapping_bytes\":0,\"scratch_bytes\":0,\"peak_rss_bytes\":null}"
                  << "}\n";
        return 0;
    }
    usage();
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "uq_native_runner: " << error.what() << std::endl;
        return 1;
    }
}
