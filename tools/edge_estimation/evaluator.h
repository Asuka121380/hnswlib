#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "event_format.h"

namespace uq {

struct ReplayCounters {
    uint64_t query_count;
    uint64_t source_count;
    uint64_t eligible_count;
    uint64_t fallback_count;
    uint64_t elapsed_ns;
    uint64_t checksum;

    ReplayCounters()
        : query_count(0), source_count(0), eligible_count(0),
          fallback_count(0), elapsed_ns(0), checksum(0) {}
};

struct QualityCounters {
    uint64_t event_count_u;
    uint64_t decision_count_s;
    uint64_t valid_estimate_count;
    uint64_t fallback_count;
    uint64_t tp;
    uint64_t fp;
    uint64_t fn;
    uint64_t tn;

    QualityCounters()
        : event_count_u(0), decision_count_s(0), valid_estimate_count(0),
          fallback_count(0), tp(0), fp(0), fn(0), tn(0) {}
};

// Kernel contract: prepareQuery(query_id), prepareSource(event), score(event).
// score returns a finite double or NaN to request exact fallback.
template<class Kernel>
ReplayCounters replayOrderedEstimator(
    const std::vector<EventRecord>& events,
    Kernel& kernel,
    size_t repeats) {
    if (repeats == 0U) throw std::invalid_argument("repeats must be positive");
    ReplayCounters result;
    EventRecord pending_source{};
    bool have_source = false;
    bool source_prepared = false;
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        for (size_t i = 0; i < events.size(); ++i) {
            const EventRecord& event = events[i];
            if (event.kind == EventKind::QueryBegin) {
                kernel.prepareQuery(event.query_id);
                ++result.query_count;
                have_source = false;
                source_prepared = false;
            } else if (event.kind == EventKind::SourceBegin) {
                pending_source = event;
                have_source = true;
                source_prepared = false;
            } else if (event.kind == EventKind::Candidate &&
                       (event.flags & ScoreSlotEligible) != 0U) {
                if (!have_source)
                    throw std::runtime_error("eligible candidate has no source context");
                if (!source_prepared) {
                    kernel.prepareSource(pending_source);
                    source_prepared = true;
                    ++result.source_count;
                }
                const double score = kernel.score(event);
                ++result.eligible_count;
                if (!std::isfinite(score)) {
                    ++result.fallback_count;
                    result.checksum ^= event.event_id + 0x9e3779b97f4a7c15ULL;
                } else {
                    uint64_t bits = 0U;
                    std::memcpy(&bits, &score, sizeof(bits));
                    result.checksum = (result.checksum << 7U) ^
                        (result.checksum >> 3U) ^ bits ^ event.event_id;
                }
            }
        }
    }
    const std::chrono::steady_clock::time_point finish =
        std::chrono::steady_clock::now();
    result.elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    return result;
}

// Native quality pass using the same kernel lifecycle and score method as the
// ordered timing pass. Labels are consumed here only and are never accepted by
// replayOrderedEstimator.
template<class Kernel>
QualityCounters evaluateQuality(
    const std::vector<EventRecord>& events,
    const std::vector<LabelRecord>& labels,
    double alpha,
    Kernel& kernel) {
    if (!std::isfinite(alpha) || alpha < 0.0)
        throw std::invalid_argument("alpha must be finite and non-negative");
    QualityCounters result;
    EventRecord pending_source{};
    bool have_source = false;
    bool source_prepared = false;
    size_t label_index = 0U;
    for (size_t i = 0; i < events.size(); ++i) {
        const EventRecord& event = events[i];
        if (event.kind == EventKind::QueryBegin) {
            kernel.prepareQuery(event.query_id);
            have_source = false;
            source_prepared = false;
        } else if (event.kind == EventKind::SourceBegin) {
            pending_source = event;
            have_source = true;
            source_prepared = false;
        }
        const bool has_label = event.kind == EventKind::Candidate ||
            event.kind == EventKind::ExactOnly;
        double exact = 0.0;
        if (has_label) {
            if (label_index >= labels.size() || labels[label_index].event_id != i)
                throw std::runtime_error("quality labels do not match events");
            exact = labels[label_index++].exact_squared_distance;
        }
        if (event.kind != EventKind::Candidate) continue;
        ++result.event_count_u;
        const bool in_decision_set = (event.flags & ThresholdValid) != 0U &&
            (event.flags & ScoreSlotEligible) != 0U &&
            std::isfinite(event.threshold_before) &&
            event.threshold_before >= 0.0 && std::isfinite(exact);
        if (!in_decision_set) continue;
        ++result.decision_count_s;
        if (!have_source)
            throw std::runtime_error("quality candidate has no source context");
        if (!source_prepared) {
            kernel.prepareSource(pending_source);
            source_prepared = true;
        }
        const double score = kernel.score(event);
        const bool valid = std::isfinite(score);
        if (valid) ++result.valid_estimate_count;
        else ++result.fallback_count;
        const bool prune = valid && score > alpha * event.threshold_before;
        const bool far = exact > event.threshold_before;
        if (prune && far) ++result.tp;
        else if (prune) ++result.fp;
        else if (far) ++result.fn;
        else ++result.tn;
    }
    if (label_index != labels.size())
        throw std::runtime_error("quality labels contain extra records");
    return result;
}

}  // namespace uq
