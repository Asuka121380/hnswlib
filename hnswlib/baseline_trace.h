#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hnswlib {

struct BaselineTraceConfig {
    uint64_t query_id = 0;
    size_t requested_k = 0;
    size_t ef_search = 0;
    size_t dimension = 0;
    uint64_t seed = 0;
    bool collect_geometry = true;
    bool collect_distance_timing = true;
    size_t dco_sample_modulus = 1;
    size_t dco_sample_remainder = 0;
    double epsilon = 1e-12;
    double threshold_stability_delta = 0.01;
};

struct BaselineQuerySummary {
    uint64_t query_id = 0;
    size_t requested_k = 0;
    size_t ef_search = 0;
    size_t ef_effective = 0;
    size_t dimension = 0;
    uint64_t seed = 0;

    uint64_t n_entry_distance = 0;
    uint64_t n_upper_edge_scan = 0;
    uint64_t n_upper_dist = 0;
    uint64_t n_base_entry_distance = 0;
    uint64_t n_edge_scan = 0;
    uint64_t n_duplicate = 0;
    uint64_t n_unique_neighbor = 0;
    uint64_t n_dist = 0;
    uint64_t n_expanded = 0;
    uint64_t n_inserted_candidate = 0;
    uint64_t n_inserted_result = 0;
    uint64_t n_threshold_changed = 0;
    uint64_t n_expanded_later = 0;
    uint64_t n_final_topk = 0;
    uint64_t n_strict_state_neutral = 0;
    uint64_t trace_records_written = 0;

    uint64_t baseline_query_latency_ns = 0;
    uint64_t trace_query_latency_ns = 0;
    uint64_t exact_distance_time_ns = 0;
    double recall_at_k = std::numeric_limits<double>::quiet_NaN();

    uint64_t nExactCallsTotal() const {
        return n_entry_distance + n_upper_dist + n_base_entry_distance + n_dist;
    }
};

struct BaselineDcoRecord {
    uint64_t query_id = 0;
    int graph_layer = 0;
    uint64_t expansion_index = 0;
    uint64_t dco_index = 0;
    uint32_t current_node_id = 0;
    uint64_t current_node_label = 0;
    uint32_t neighbor_id = 0;
    uint64_t neighbor_label = 0;

    double dist_qc = std::numeric_limits<double>::quiet_NaN();
    double dist_qd = std::numeric_limits<double>::quiet_NaN();
    double threshold_before = std::numeric_limits<double>::quiet_NaN();
    double threshold_after = std::numeric_limits<double>::quiet_NaN();
    bool threshold_valid_before = false;
    bool threshold_valid_after = false;

    size_t candidate_queue_size_before = 0;
    size_t candidate_queue_size_after = 0;
    size_t result_queue_size_before = 0;
    size_t result_queue_size_after = 0;

    bool inserted_candidate_queue = false;
    bool inserted_result_queue = false;
    bool threshold_changed = false;
    bool expanded_later = false;
    bool in_final_topk = false;

    bool geometry_valid = false;
    double edge_length_cd = std::numeric_limits<double>::quiet_NaN();
    double edge_dot_qcd = std::numeric_limits<double>::quiet_NaN();
    double edge_cosine_qcd = std::numeric_limits<double>::quiet_NaN();
    double triangle_lower_bound = std::numeric_limits<double>::quiet_NaN();

    double absolute_margin = std::numeric_limits<double>::quiet_NaN();
    double relative_margin = std::numeric_limits<double>::quiet_NaN();
    bool is_negative_at_evaluation = false;
    bool is_state_neutral = false;
    double search_progress_fraction = 0.0;
    bool threshold_stability_indicator = false;
};

class BaselineTraceCollector {
 public:
    explicit BaselineTraceCollector(const BaselineTraceConfig& config)
        : config(config) {
        beginQuery();
    }

    void beginQuery() {
        summary = BaselineQuerySummary();
        summary.query_id = config.query_id;
        summary.requested_k = config.requested_k;
        summary.ef_search = config.ef_search;
        summary.dimension = config.dimension;
        summary.seed = config.seed;
        records.clear();
        record_by_neighbor_.clear();
        evaluated_ids_.clear();
        expanded_counted_.clear();
        final_counted_.clear();
        threshold_history_.clear();
        finalized_ = false;
    }

    bool shouldSampleDco(uint64_t dco_index) const {
        const size_t modulus = config.dco_sample_modulus == 0 ? 1 : config.dco_sample_modulus;
        uint64_t value = config.seed ^ (config.query_id + 0x9e3779b97f4a7c15ULL);
        value ^= dco_index + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        value ^= value >> 31;
        return (value % modulus) == (config.dco_sample_remainder % modulus);
    }

    void registerEvaluation(uint32_t neighbor_id) {
        evaluated_ids_.insert(neighbor_id);
    }

    void recordThreshold(bool valid, double value) {
        ThresholdPoint point;
        point.valid = valid;
        point.value = value;
        threshold_history_.push_back(point);
    }

    void appendRecord(const BaselineDcoRecord& record) {
        const size_t index = records.size();
        records.push_back(record);
        record_by_neighbor_[record.neighbor_id] = index;
    }

    void markExpanded(uint32_t internal_id) {
        if (evaluated_ids_.count(internal_id) != 0 && expanded_counted_.insert(internal_id).second) {
            ++summary.n_expanded_later;
        }
        std::unordered_map<uint32_t, size_t>::const_iterator found = record_by_neighbor_.find(internal_id);
        if (found != record_by_neighbor_.end()) {
            records[found->second].expanded_later = true;
        }
    }

    void markFinalTopK(uint32_t internal_id) {
        if (evaluated_ids_.count(internal_id) != 0 && final_counted_.insert(internal_id).second) {
            ++summary.n_final_topk;
        }
        std::unordered_map<uint32_t, size_t>::const_iterator found = record_by_neighbor_.find(internal_id);
        if (found != record_by_neighbor_.end()) {
            records[found->second].in_final_topk = true;
        }
    }

    void addExactDistanceTime(uint64_t nanoseconds) {
        summary.exact_distance_time_ns += nanoseconds;
    }

    void computeFloatL2Geometry(
        const void* query_data,
        const void* current_data,
        const void* neighbor_data,
        BaselineDcoRecord& record) const {
        if (!config.collect_geometry || config.dimension == 0) return;

        const float* q = static_cast<const float*>(query_data);
        const float* c = static_cast<const float*>(current_data);
        const float* d = static_cast<const float*>(neighbor_data);
        double norm_qc_sq = 0.0;
        double norm_cd_sq = 0.0;
        double dot = 0.0;
        for (size_t i = 0; i < config.dimension; ++i) {
            const double qc = static_cast<double>(q[i]) - static_cast<double>(c[i]);
            const double cd = static_cast<double>(d[i]) - static_cast<double>(c[i]);
            norm_qc_sq += qc * qc;
            norm_cd_sq += cd * cd;
            dot += qc * cd;
        }

        const double norm_qc = std::sqrt(norm_qc_sq);
        const double norm_cd = std::sqrt(norm_cd_sq);
        record.geometry_valid = true;
        record.edge_length_cd = norm_cd;
        record.edge_dot_qcd = dot;
        record.triangle_lower_bound = std::fabs(norm_qc - norm_cd);
        const double denominator = norm_qc * norm_cd;
        if (denominator > config.epsilon) {
            record.edge_cosine_qcd = dot / denominator;
        }
    }

    void finalize() {
        if (finalized_) return;
        finalized_ = true;
        summary.trace_records_written = records.size();

        uint64_t stable_start = summary.n_dist;
        double final_threshold = std::numeric_limits<double>::quiet_NaN();
        for (size_t i = threshold_history_.size(); i > 0; --i) {
            if (threshold_history_[i - 1].valid) {
                final_threshold = threshold_history_[i - 1].value;
                stable_start = static_cast<uint64_t>(i - 1);
                break;
            }
        }
        if (std::isfinite(final_threshold)) {
            for (size_t i = threshold_history_.size(); i > 0; --i) {
                const ThresholdPoint& point = threshold_history_[i - 1];
                if (!point.valid) break;
                const double relative_change = std::fabs(point.value - final_threshold) /
                    std::max(std::fabs(final_threshold), config.epsilon);
                if (relative_change > config.threshold_stability_delta) break;
                stable_start = static_cast<uint64_t>(i - 1);
            }
        }

        const double denominator = summary.n_dist > 1 ? static_cast<double>(summary.n_dist - 1) : 1.0;
        for (size_t i = 0; i < records.size(); ++i) {
            BaselineDcoRecord& record = records[i];
            record.search_progress_fraction = static_cast<double>(record.dco_index) / denominator;
            record.threshold_stability_indicator = record.dco_index >= stable_start;
            record.is_state_neutral = !record.inserted_candidate_queue &&
                !record.inserted_result_queue &&
                !record.threshold_changed &&
                !record.expanded_later &&
                !record.in_final_topk;
        }
    }

    BaselineTraceConfig config;
    BaselineQuerySummary summary;
    std::vector<BaselineDcoRecord> records;

 private:
    struct ThresholdPoint {
        bool valid = false;
        double value = std::numeric_limits<double>::quiet_NaN();
    };

    std::unordered_map<uint32_t, size_t> record_by_neighbor_;
    std::unordered_set<uint32_t> evaluated_ids_;
    std::unordered_set<uint32_t> expanded_counted_;
    std::unordered_set<uint32_t> final_counted_;
    std::vector<ThresholdPoint> threshold_history_;
    bool finalized_ = false;
};

}  // namespace hnswlib
