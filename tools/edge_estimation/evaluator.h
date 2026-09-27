#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
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

struct BatchParityReport {
    uint64_t compared_count;
    uint64_t nonfinite_mismatch_count;
    uint64_t tolerance_failure_count;
    double max_absolute_error;
    double max_relative_error;

    BatchParityReport()
        : compared_count(0), nonfinite_mismatch_count(0),
          tolerance_failure_count(0), max_absolute_error(0.0),
          max_relative_error(0.0) {}
    bool valid() const {
        return compared_count > 0U && nonfinite_mismatch_count == 0U &&
            tolerance_failure_count == 0U;
    }
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

struct ErrorPercentiles {
    uint64_t count;
    double p50;
    double p90;
    double p95;
    double p99;
    double maximum;

    ErrorPercentiles()
        : count(0), p50(0.0), p90(0.0), p95(0.0), p99(0.0), maximum(0.0) {}
};

struct QueryQualityReport {
    uint64_t query_id;
    QualityCounters counters;
    ErrorPercentiles overestimate_error;
    ErrorPercentiles underestimate_error;
};

struct QualityReport {
    QualityCounters counters;
    ErrorPercentiles overestimate_error;
    ErrorPercentiles underestimate_error;
    std::vector<QueryQualityReport> per_query;
};

inline double percentileFromSorted(const std::vector<double>& values, double fraction) {
    if (values.empty()) return 0.0;
    const double position = fraction * static_cast<double>(values.size() - 1U);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

inline ErrorPercentiles summarizeErrors(std::vector<double> values) {
    ErrorPercentiles result;
    if (values.empty()) return result;
    std::sort(values.begin(), values.end());
    result.count = static_cast<uint64_t>(values.size());
    result.p50 = percentileFromSorted(values, 0.50);
    result.p90 = percentileFromSorted(values, 0.90);
    result.p95 = percentileFromSorted(values, 0.95);
    result.p99 = percentileFromSorted(values, 0.99);
    result.maximum = values.back();
    return result;
}

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

inline std::vector<uint64_t> orderedUniqueQueryIds(
    const std::vector<EventRecord>& events) {
    std::vector<uint64_t> result;
    std::map<uint64_t, bool> seen;
    for (const EventRecord& event : events) {
        if (event.kind == EventKind::QueryBegin &&
            seen.emplace(event.query_id, true).second)
            result.push_back(event.query_id);
    }
    return result;
}

namespace detail {

template<class Kernel>
auto prepareQueryBatch(Kernel& kernel,
                       const std::vector<uint64_t>& query_ids,
                       size_t batch_size,
                       int) -> decltype(kernel.prepareQueryBatch(query_ids, batch_size), void()) {
    kernel.prepareQueryBatch(query_ids, batch_size);
}

template<class Kernel>
void prepareQueryBatch(Kernel&,
                       const std::vector<uint64_t>&,
                       size_t,
                       long) {
    throw std::runtime_error("backend does not support batch query preparation");
}

template<class Kernel>
auto batchPreparationEngine(const Kernel& kernel, int)
    -> decltype(kernel.batchPreparationEngine(), std::string()) {
    return kernel.batchPreparationEngine();
}

template<class Kernel>
std::string batchPreparationEngine(const Kernel&, long) {
    return "unsupported";
}

}  // namespace detail

template<class Kernel>
std::string batchPreparationEngine(const Kernel& kernel) {
    return detail::batchPreparationEngine(kernel, 0);
}

// Batch preprocessing remains inside the timed region. Query ids are collected
// once from the immutable event stream, then each repeat rebuilds every rotated
// query and LUT before replaying the original ordered events.
template<class Kernel>
ReplayCounters replayBatchPreparedEstimator(
    const std::vector<EventRecord>& events,
    Kernel& kernel,
    size_t repeats,
    size_t batch_size) {
    if (repeats == 0U) throw std::invalid_argument("repeats must be positive");
    if (batch_size == 0U) throw std::invalid_argument("query batch size must be positive");
    const std::vector<uint64_t> query_ids = orderedUniqueQueryIds(events);
    if (query_ids.empty()) throw std::invalid_argument("event stream contains no queries");
    ReplayCounters result;
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        detail::prepareQueryBatch(kernel, query_ids, batch_size, 0);
        EventRecord pending_source{};
        bool have_source = false;
        bool source_prepared = false;
        for (const EventRecord& event : events) {
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

template<class Kernel>
BatchParityReport compareScalarAndBatchEstimator(
    const std::vector<EventRecord>& events,
    Kernel& kernel,
    size_t batch_size,
    double absolute_tolerance,
    double relative_tolerance) {
    if (batch_size == 0U || absolute_tolerance < 0.0 || relative_tolerance < 0.0 ||
        !std::isfinite(absolute_tolerance) || !std::isfinite(relative_tolerance))
        throw std::invalid_argument("invalid batch parity parameters");
    std::vector<double> scalar_scores;
    EventRecord pending_source{};
    bool have_source = false;
    bool source_prepared = false;
    for (const EventRecord& event : events) {
        if (event.kind == EventKind::QueryBegin) {
            kernel.prepareQuery(event.query_id);
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
            }
            scalar_scores.push_back(kernel.score(event));
        }
    }

    const std::vector<uint64_t> query_ids = orderedUniqueQueryIds(events);
    if (query_ids.empty()) throw std::invalid_argument("event stream contains no queries");
    detail::prepareQueryBatch(kernel, query_ids, batch_size, 0);
    BatchParityReport report;
    have_source = false;
    source_prepared = false;
    size_t score_index = 0U;
    for (const EventRecord& event : events) {
        if (event.kind == EventKind::QueryBegin) {
            kernel.prepareQuery(event.query_id);
            have_source = false;
            source_prepared = false;
        } else if (event.kind == EventKind::SourceBegin) {
            pending_source = event;
            have_source = true;
            source_prepared = false;
        } else if (event.kind == EventKind::Candidate &&
                   (event.flags & ScoreSlotEligible) != 0U) {
            if (!have_source || score_index >= scalar_scores.size())
                throw std::runtime_error("batch parity event traversal mismatch");
            if (!source_prepared) {
                kernel.prepareSource(pending_source);
                source_prepared = true;
            }
            const double scalar = scalar_scores[score_index++];
            const double batch = kernel.score(event);
            const bool scalar_finite = std::isfinite(scalar);
            const bool batch_finite = std::isfinite(batch);
            if (scalar_finite != batch_finite) {
                ++report.nonfinite_mismatch_count;
                continue;
            }
            if (!scalar_finite) continue;
            ++report.compared_count;
            const double absolute_error = std::fabs(scalar - batch);
            const double scale = std::max(std::fabs(scalar), std::fabs(batch));
            const double relative_error = scale == 0.0 ? 0.0 : absolute_error / scale;
            report.max_absolute_error = std::max(report.max_absolute_error, absolute_error);
            report.max_relative_error = std::max(report.max_relative_error, relative_error);
            if (absolute_error > absolute_tolerance + relative_tolerance * scale)
                ++report.tolerance_failure_count;
        }
    }
    if (score_index != scalar_scores.size())
        throw std::runtime_error("batch parity score count mismatch");
    return report;
}

// Native quality pass using the same kernel lifecycle and score method as the
// ordered timing pass. Labels are consumed here only and are never accepted by
// replayOrderedEstimator.
template<class Kernel>
QualityReport evaluateQualityDetailed(
    const std::vector<EventRecord>& events,
    const std::vector<LabelRecord>& labels,
    double alpha,
    Kernel& kernel) {
    if (!std::isfinite(alpha) || alpha < 0.0)
        throw std::invalid_argument("alpha must be finite and non-negative");
    QualityReport report;
    QualityCounters& result = report.counters;
    struct WorkingQuery {
        QualityCounters counters;
        std::vector<double> over;
        std::vector<double> under;
    };
    std::map<uint64_t, WorkingQuery> per_query;
    std::vector<double> overestimate_error;
    std::vector<double> underestimate_error;
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
        WorkingQuery& query = per_query[event.query_id];
        ++query.counters.event_count_u;
        const bool in_decision_set = (event.flags & ThresholdValid) != 0U &&
            (event.flags & ScoreSlotEligible) != 0U &&
            std::isfinite(event.threshold_before) &&
            event.threshold_before >= 0.0 && std::isfinite(exact);
        if (!in_decision_set) continue;
        ++result.decision_count_s;
        ++query.counters.decision_count_s;
        if (!have_source)
            throw std::runtime_error("quality candidate has no source context");
        if (!source_prepared) {
            kernel.prepareSource(pending_source);
            source_prepared = true;
        }
        const double score = kernel.score(event);
        const bool valid = std::isfinite(score);
        if (valid) {
            ++result.valid_estimate_count;
            ++query.counters.valid_estimate_count;
            const double signed_error = score - exact;
            const double over = std::max(0.0, signed_error);
            const double under = std::max(0.0, -signed_error);
            overestimate_error.push_back(over);
            underestimate_error.push_back(under);
            query.over.push_back(over);
            query.under.push_back(under);
        } else {
            ++result.fallback_count;
            ++query.counters.fallback_count;
        }
        const bool prune = valid && score > alpha * event.threshold_before;
        const bool far = exact > event.threshold_before;
        if (prune && far) { ++result.tp; ++query.counters.tp; }
        else if (prune) { ++result.fp; ++query.counters.fp; }
        else if (far) { ++result.fn; ++query.counters.fn; }
        else { ++result.tn; ++query.counters.tn; }
    }
    if (label_index != labels.size())
        throw std::runtime_error("quality labels contain extra records");
    report.overestimate_error = summarizeErrors(overestimate_error);
    report.underestimate_error = summarizeErrors(underestimate_error);
    for (typename std::map<uint64_t, WorkingQuery>::iterator it = per_query.begin();
         it != per_query.end(); ++it) {
        QueryQualityReport query_report;
        query_report.query_id = it->first;
        query_report.counters = it->second.counters;
        query_report.overestimate_error = summarizeErrors(it->second.over);
        query_report.underestimate_error = summarizeErrors(it->second.under);
        report.per_query.push_back(query_report);
    }
    return report;
}

template<class Kernel>
QualityCounters evaluateQuality(
    const std::vector<EventRecord>& events,
    const std::vector<LabelRecord>& labels,
    double alpha,
    Kernel& kernel) {
    return evaluateQualityDetailed(events, labels, alpha, kernel).counters;
}

}  // namespace uq
