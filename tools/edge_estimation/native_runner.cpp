#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "backend_factory.h"
#include "backend_registry.h"
#include "event_format.h"
#include "evaluator.h"
#include "backends/pq_packed.h"
#include "query_store.h"

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

void printErrorPercentiles(const uq::ErrorPercentiles& value) {
    std::cout << "{\"count\":" << value.count
              << ",\"p50\":" << value.p50
              << ",\"p90\":" << value.p90
              << ",\"p95\":" << value.p95
              << ",\"p99\":" << value.p99
              << ",\"max\":" << value.maximum << '}';
}

void printQualityCounters(const uq::QualityCounters& value) {
    std::cout << "{\"event_count_u\":" << value.event_count_u
              << ",\"decision_count_s\":" << value.decision_count_s
              << ",\"valid_estimate_count\":" << value.valid_estimate_count
              << ",\"fallback_count\":" << value.fallback_count
              << ",\"tp\":" << value.tp << ",\"fp\":" << value.fp
              << ",\"fn\":" << value.fn << ",\"tn\":" << value.tn << '}';
}

void usage() {
    std::cerr << "usage:\n"
              << "  uq_native_runner capabilities\n"
              << "  uq_native_runner validate EVENTS [LABELS RANGES]\n"
              << "  uq_native_runner validate-artifact ARTIFACT EVENTS QUERIES\n"
              << "  uq_native_runner quality EVENTS LABELS ARTIFACT QUERIES ALPHA\n"
              << "  uq_native_runner bench-artifact EVENTS ARTIFACT QUERIES [REPEATS]\n"
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
            << "\"backends\":[";
        const std::vector<uq::BackendCapability> capabilities = uq::allBackendCapabilities();
        for (size_t i = 0; i < capabilities.size(); ++i) {
            const uq::BackendCapability& value = capabilities[i];
            if (i) std::cout << ',';
            std::cout << "{\"name\":\"" << value.name << "\""
                      << ",\"compiled\":" << (value.compiled ? "true" : "false")
                      << ",\"native_available\":" << (value.native_available ? "true" : "false")
                      << ",\"artifact_supported\":" << (value.artifact_supported ? "true" : "false")
                      << ",\"runtime_isa_supported\":" << (value.runtime_isa_supported ? "true" : "false")
                      << ",\"formal_validation_passed\":" << (value.formal_validation_passed ? "true" : "false")
                      << ",\"supports_scalar\":" << (value.supports_scalar ? "true" : "false");
            if (value.reason && *value.reason)
                std::cout << ",\"reason\":\"" << value.reason << "\"";
            std::cout << '}';
        }
        std::cout << "]}\n";
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
        std::shared_ptr<const uq::QueryStore> queries =
            std::make_shared<uq::QueryStore>(argv[4], header.dimension);
        return uq::withArtifactKernel(argv[2], queries, header,
            [&](auto& kernel, const uq::ArtifactDescriptor& descriptor) {
                for (const uq::EventRecord& event : events)
                    if (event.kind == uq::EventKind::QueryBegin) {
                        kernel.prepareQuery(event.query_id); break;
                    }
                std::cout << "{\"valid\":true,\"backend\":\"" << descriptor.backend_name
                          << "\",\"format\":\"" << descriptor.format
                          << "\",\"backend_bytes\":" << kernel.backendBytes() << "}\n";
                return 0;
            });
    }
    if (command == "quality" || command == "quality-pq") {
        if (argc != 7) { usage(); return 2; }
        uq::Header event_header, label_header;
        const std::vector<uq::EventRecord> events = uq::readEvents(argv[2], &event_header);
        const std::vector<uq::LabelRecord> labels = uq::readLabels(argv[3], &label_header);
        if (event_header.identity != label_header.identity) throw std::runtime_error("label identity mismatch");
        std::shared_ptr<const uq::QueryStore> queries =
            std::make_shared<uq::QueryStore>(argv[5], event_header.dimension);
        const double alpha = std::stod(argv[6]);
        return uq::withArtifactKernel(argv[4], queries, event_header,
            [&](auto& kernel, const uq::ArtifactDescriptor& descriptor) {
                const uq::QualityReport report =
                    uq::evaluateQualityDetailed(events, labels, alpha, kernel);
                const uq::QualityCounters& value = report.counters;
                std::cout << "{\"schema_version\":1,\"backend\":\"" << descriptor.backend_name
                          << "\",\"alpha\":" << alpha
                          << ",\"event_count_u\":" << value.event_count_u
                          << ",\"decision_count_s\":" << value.decision_count_s
                          << ",\"valid_estimate_count\":" << value.valid_estimate_count
                          << ",\"fallback_count\":" << value.fallback_count
                          << ",\"tp\":" << value.tp << ",\"fp\":" << value.fp
                          << ",\"fn\":" << value.fn << ",\"tn\":" << value.tn
                          << ",\"error_percentiles\":{\"overestimate\":";
                printErrorPercentiles(report.overestimate_error);
                std::cout << ",\"underestimate\":";
                printErrorPercentiles(report.underestimate_error);
                std::cout << "},\"per_query\":[";
                for (size_t i = 0; i < report.per_query.size(); ++i) {
                    if (i) std::cout << ',';
                    const uq::QueryQualityReport& query = report.per_query[i];
                    std::cout << "{\"query_id\":" << query.query_id << ",\"counts\":";
                    printQualityCounters(query.counters);
                    std::cout << ",\"overestimate\":";
                    printErrorPercentiles(query.overestimate_error);
                    std::cout << ",\"underestimate\":";
                    printErrorPercentiles(query.underestimate_error);
                    std::cout << '}';
                }
                std::cout << "]}\n";
                return 0;
            });
    }
    if (command == "bench-artifact" || command == "bench-pq") {
        if (argc != 5 && argc != 6) { usage(); return 2; }
        const size_t repeats = argc == 6 ? static_cast<size_t>(std::stoull(argv[5])) : 1U;
        if (repeats == 0U) throw std::invalid_argument("repeats must be positive");
        uq::Header header; const std::vector<uq::EventRecord> events = uq::readEvents(argv[2], &header);
        std::shared_ptr<const uq::QueryStore> queries =
            std::make_shared<uq::QueryStore>(argv[4], header.dimension);
        return uq::withArtifactKernel(argv[3], queries, header,
            [&](auto& kernel, const uq::ArtifactDescriptor& descriptor) {
                (void)uq::replayOrderedEstimator(events, kernel, 1U);
                std::vector<uq::ReplayCounters> runs;
                for (size_t i = 0; i < repeats; ++i)
                    runs.push_back(uq::replayOrderedEstimator(events, kernel, 1U));
                std::cout << "{\"schema_version\":1,\"backend\":\"" << descriptor.backend_name
                          << "\",\"mode\":\"ordered_estimator\","
                          << "\"quality_valid\":false,\"formal_validation_passed\":false,"
                          << "\"raw_elapsed_ns\":[";
                for (size_t i = 0; i < runs.size(); ++i) {
                    if (i) std::cout << ','; std::cout << runs[i].elapsed_ns;
                }
                const uq::ReplayCounters& value = runs.at(0);
                std::cout << "],\"query_count\":" << value.query_count
                          << ",\"source_setup_calls\":" << value.source_count
                          << ",\"eligible_events\":" << value.eligible_count
                          << ",\"fallback_count\":" << value.fallback_count
                          << ",\"checksum\":" << value.checksum
                          << ",\"memory_report\":{\"backend_bytes\":" << kernel.backendBytes()
                          << ",\"query_store_bytes\":" << queries->bytes()
                          << ",\"scratch_bytes\":" << kernel.scratchBytes()
                          << ",\"peak_rss_bytes\":null}}\n";
                return 0;
            });
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
