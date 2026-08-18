#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "edge_quant_v0_metrics.h"

namespace hnswlib {

#if defined(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION) && \
    defined(HNSWLIB_ENABLE_V0_APPROX_SHADOW)

static const uint32_t V0_RETRY_SHADOW_SCHEMA_VERSION = 1U;

struct V0RetryShadowQuerySummary {
    uint64_t query_id = 0;
    double beta = 1.0;
    uint64_t eligible_first_visits = 0;
    uint64_t first_pruned = 0;
    uint64_t first_false_pruned = 0;
    uint64_t pruned_revisited = 0;
    uint64_t false_pruned_revisited = 0;
    uint64_t duplicate_encounters_after_prune = 0;
    uint64_t expanded_nodes = 0;
};

struct V0RetryShadowCandidateRecord {
    uint64_t query_id = 0;
    double beta = 1.0;
    uint64_t candidate_id = 0;
    uint64_t first_parent_id = 0;
    uint64_t second_parent_id = 0;
    uint64_t first_expansion_index = 0;
    uint64_t second_expansion_index = 0;
    uint64_t duplicate_encounters = 0;
    uint64_t first_parent_degree = 0;
    uint64_t candidate_degree = 0;
    double first_threshold = 0.0;
    double first_raw_estimate = 0.0;
    double first_exact_distance = 0.0;
    bool first_false_prune = false;
    bool revisited = false;
};

class V0RetryShadowSink {
 public:
    virtual ~V0RetryShadowSink() {}
    virtual void appendQuerySummary(
        const V0RetryShadowQuerySummary& summary) = 0;
    virtual void appendCandidate(
        const V0RetryShadowCandidateRecord& record) = 0;
};

class V0RetryShadowTracker : public V0ShadowValidationCollector {
 public:
    V0RetryShadowTracker(
        const std::vector<double>& betas,
        V0RetryShadowSink* sink)
        : betas_(betas), sink_(sink) {
        if (betas_.empty() || sink_ == nullptr) {
            throw std::invalid_argument(
                "retry shadow requires betas and a sink");
        }
        std::sort(betas_.begin(), betas_.end());
        betas_.erase(
            std::unique(betas_.begin(), betas_.end()),
            betas_.end());
        for (size_t i = 0; i < betas_.size(); ++i) {
            if (!std::isfinite(betas_[i]) || betas_[i] < 1.0) {
                throw std::invalid_argument(
                    "retry shadow beta must be finite and >= 1");
            }
        }
        summaries_.resize(betas_.size());
    }

    void beginQuery(uint64_t query_id) {
        if (query_active_) {
            throw std::logic_error(
                "retry shadow query already active");
        }
        query_active_ = true;
        query_id_ = query_id;
        expanded_nodes_ = 0U;
        candidates_.clear();
        summaries_.assign(
            betas_.size(), V0RetryShadowQuerySummary());
        for (size_t i = 0; i < betas_.size(); ++i) {
            summaries_[i].query_id = query_id;
            summaries_[i].beta = betas_[i];
        }
    }

    void onExpansion(const V0ShadowExpansionRecord& record) {
        requireActiveQuery(record.query_id);
        expanded_nodes_ = std::max(
            expanded_nodes_, record.expansion_index + 1U);
    }

    void append(const V0ShadowRecord& record) {
        requireActiveQuery(record.query_id);
        if (!record.lower_bound_valid) {
            return;
        }
        CandidateState state;
        state.first = record;
        state.first_pruned.assign(betas_.size(), 0U);
        state.first_false_pruned.assign(betas_.size(), 0U);
        bool pruned_for_any_beta = false;
        for (size_t i = 0; i < betas_.size(); ++i) {
            V0RetryShadowQuerySummary& summary = summaries_[i];
            ++summary.eligible_first_visits;
            const bool first_pruned =
                record.approximate_squared_distance >
                betas_[i] * record.threshold;
            const bool first_false_pruned =
                first_pruned &&
                record.shadow_exact_squared_distance <=
                    record.threshold;
            state.first_pruned[i] = first_pruned ? 1U : 0U;
            state.first_false_pruned[i] =
                first_false_pruned ? 1U : 0U;
            summary.first_pruned += first_pruned ? 1U : 0U;
            summary.first_false_pruned +=
                first_false_pruned ? 1U : 0U;
            pruned_for_any_beta =
                pruned_for_any_beta || first_pruned;
        }
        if (pruned_for_any_beta) {
            const std::pair<
                CandidateMap::iterator, bool> inserted =
                    candidates_.insert(
                        std::make_pair(record.candidate_id, state));
            if (!inserted.second) {
                throw std::logic_error(
                    "retry shadow observed duplicate first visit");
            }
        }
    }

    void onDuplicate(const V0ShadowDuplicateRecord& record) {
        requireActiveQuery(record.query_id);
        CandidateMap::iterator found =
            candidates_.find(record.candidate_id);
        if (found == candidates_.end()) {
            return;
        }
        CandidateState& state = found->second;
        if (record.current_node_id ==
            state.first.current_node_id) {
            return;
        }
        ++state.duplicate_encounters;
        if (!state.revisited) {
            state.revisited = true;
            state.second_parent_id = record.current_node_id;
            state.second_expansion_index =
                record.expansion_index;
        }
    }

    void endQuery(uint64_t query_id) {
        requireActiveQuery(query_id);
        for (CandidateMap::const_iterator it =
                 candidates_.begin();
             it != candidates_.end();
             ++it) {
            const CandidateState& state = it->second;
            for (size_t i = 0; i < betas_.size(); ++i) {
                if (!state.first_pruned[i]) {
                    continue;
                }
                V0RetryShadowQuerySummary& summary =
                    summaries_[i];
                if (state.revisited) {
                    ++summary.pruned_revisited;
                    if (state.first_false_pruned[i]) {
                        ++summary.false_pruned_revisited;
                    }
                }
                summary.duplicate_encounters_after_prune +=
                    state.duplicate_encounters;

                V0RetryShadowCandidateRecord record;
                record.query_id = query_id;
                record.beta = betas_[i];
                record.candidate_id = state.first.candidate_id;
                record.first_parent_id =
                    state.first.current_node_id;
                record.second_parent_id =
                    state.second_parent_id;
                record.first_expansion_index =
                    state.first.expansion_index;
                record.second_expansion_index =
                    state.second_expansion_index;
                record.duplicate_encounters =
                    state.duplicate_encounters;
                record.first_parent_degree =
                    state.first.current_node_degree;
                record.candidate_degree =
                    state.first.candidate_degree;
                record.first_threshold = state.first.threshold;
                record.first_raw_estimate =
                    state.first.approximate_squared_distance;
                record.first_exact_distance =
                    state.first.shadow_exact_squared_distance;
                record.first_false_prune =
                    state.first_false_pruned[i] != 0U;
                record.revisited = state.revisited;
                sink_->appendCandidate(record);
            }
        }
        for (size_t i = 0; i < summaries_.size(); ++i) {
            summaries_[i].expanded_nodes = expanded_nodes_;
            sink_->appendQuerySummary(summaries_[i]);
        }
        candidates_.clear();
        query_active_ = false;
    }

    const std::vector<double>& betas() const {
        return betas_;
    }

 private:
    struct CandidateState {
        V0ShadowRecord first;
        std::vector<uint8_t> first_pruned;
        std::vector<uint8_t> first_false_pruned;
        uint64_t duplicate_encounters = 0;
        uint64_t second_parent_id = 0;
        uint64_t second_expansion_index = 0;
        bool revisited = false;
    };

    typedef std::unordered_map<uint64_t, CandidateState>
        CandidateMap;

    void requireActiveQuery(uint64_t query_id) const {
        if (!query_active_ || query_id != query_id_) {
            throw std::logic_error(
                "retry shadow callback outside active query");
        }
    }

    std::vector<double> betas_;
    V0RetryShadowSink* sink_;
    CandidateMap candidates_;
    std::vector<V0RetryShadowQuerySummary> summaries_;
    uint64_t query_id_ = 0;
    uint64_t expanded_nodes_ = 0;
    bool query_active_ = false;
};

#endif

}  // namespace hnswlib
