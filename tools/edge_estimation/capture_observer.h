#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "event_format.h"
#include "hnswlib/edge_estimation/graph_catalog.h"
#include "hnswlib/edge_estimation/observer.h"

namespace uq {

class DatasetCaptureObserver : public hnswlib::edge_estimation::CaptureObserver {
 public:
    DatasetCaptureObserver(
        const hnswlib::edge_estimation::EdgeCatalog& catalog,
        const std::vector<uint64_t>& query_ids)
        : catalog_(catalog), query_ids_(query_ids), next_query_(0U),
          active_(false), pending_candidate_(false), current_query_(0U),
          query_begin_event_(0U), pending_event_id_(0U) {}

    void onQueryBegin() override {
        if (active_ || next_query_ >= query_ids_.size())
            throw std::logic_error("capture query lifecycle mismatch");
        active_ = true;
        pending_candidate_ = false;
        current_query_ = query_ids_[next_query_++];
        query_begin_event_ = events_.size();
        EventRecord record = blank(EventKind::QueryBegin, -1);
        append(record);
    }

    void onExactOnly(
        int32_t graph_layer,
        uint32_t target_id,
        double exact_squared_distance) override {
        requireActive();
        EventRecord record = blank(EventKind::ExactOnly, graph_layer);
        record.target_id = target_id;
        append(record);
        appendLabel(record.event_id, exact_squared_distance);
    }

    void onSourceBegin(
        uint64_t expansion_id,
        uint32_t source_id,
        uint32_t source_degree,
        double d_current) override {
        requireActive();
        if (pending_candidate_) throw std::logic_error("candidate label missing");
        EventRecord record = blank(EventKind::SourceBegin, -1);
        record.expansion_id = expansion_id;
        record.source_id = source_id;
        record.source_degree = source_degree;
        record.d_current = d_current;
        append(record);
    }

    void onCandidateBefore(
        uint64_t expansion_id,
        uint64_t,
        uint32_t source_id,
        uint32_t target_id,
        uint32_t neighbor_slot,
        uint32_t source_degree,
        double d_current,
        double threshold_before,
        bool threshold_valid,
        bool score_slot_eligible) override {
        requireActive();
        if (pending_candidate_) throw std::logic_error("candidate label missing");
        const uint64_t edge_id = catalog_.edgeId(source_id, neighbor_slot);
        if (catalog_.target(edge_id) != target_id)
            throw std::runtime_error("capture target does not match edge catalog");
        EventRecord record = blank(EventKind::Candidate, 0);
        record.flags = static_cast<uint8_t>(FirstVisit |
            (threshold_valid ? ThresholdValid : 0U) |
            (score_slot_eligible ? ScoreSlotEligible : 0U));
        record.expansion_id = expansion_id;
        record.edge_id = edge_id;
        record.source_id = source_id;
        record.target_id = target_id;
        record.neighbor_slot = neighbor_slot;
        record.source_degree = source_degree;
        record.d_current = d_current;
        record.threshold_before = threshold_valid ? threshold_before : 0.0;
        append(record);
        pending_event_id_ = record.event_id;
        pending_candidate_ = true;
    }

    void onCandidateExact(uint64_t, double exact_squared_distance) override {
        requireActive();
        if (!pending_candidate_) throw std::logic_error("unexpected candidate label");
        appendLabel(pending_event_id_, exact_squared_distance);
        pending_candidate_ = false;
    }

    void onQueryEnd() override {
        requireActive();
        if (pending_candidate_) throw std::logic_error("candidate label missing");
        EventRecord record = blank(EventKind::QueryEnd, -1);
        append(record);
        ranges_.push_back(QueryRangeRecord{
            current_query_, query_begin_event_,
            static_cast<uint64_t>(events_.size()) - query_begin_event_});
        active_ = false;
    }

    const std::vector<EventRecord>& events() const { return events_; }
    const std::vector<LabelRecord>& labels() const { return labels_; }
    const std::vector<QueryRangeRecord>& ranges() const { return ranges_; }

    void write(
        const std::string& events_path,
        const std::string& labels_path,
        const std::string& ranges_path,
        uint32_t dimension) const {
        if (active_ || next_query_ != query_ids_.size())
            throw std::logic_error("capture did not consume all configured queries");
        const std::array<uint8_t, 32>& identity = catalog_.identityDigest();
        writeEvents(events_path, dimension, identity, events_);
        writeLabels(labels_path, dimension, identity, labels_);
        writeQueryRanges(ranges_path, dimension, identity, ranges_);
    }

 private:
    EventRecord blank(EventKind kind, int32_t graph_layer) const {
        EventRecord record{};
        record.kind = kind;
        record.graph_layer = graph_layer;
        record.event_id = events_.size();
        record.query_id = current_query_;
        record.expansion_id = std::numeric_limits<uint64_t>::max();
        record.edge_id = std::numeric_limits<uint64_t>::max();
        record.source_id = std::numeric_limits<uint32_t>::max();
        record.target_id = std::numeric_limits<uint32_t>::max();
        record.neighbor_slot = std::numeric_limits<uint32_t>::max();
        return record;
    }

    void append(const EventRecord& record) { events_.push_back(record); }
    void appendLabel(uint64_t event_id, double value) {
        if (!std::isfinite(value)) throw std::runtime_error("non-finite exact label");
        labels_.push_back(LabelRecord{event_id, value});
    }
    void requireActive() const {
        if (!active_) throw std::logic_error("capture event outside query");
    }

    const hnswlib::edge_estimation::EdgeCatalog& catalog_;
    std::vector<uint64_t> query_ids_;
    size_t next_query_;
    bool active_;
    bool pending_candidate_;
    uint64_t current_query_;
    uint64_t query_begin_event_;
    uint64_t pending_event_id_;
    std::vector<EventRecord> events_;
    std::vector<LabelRecord> labels_;
    std::vector<QueryRangeRecord> ranges_;
};

}  // namespace uq
