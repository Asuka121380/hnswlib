#pragma once

#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
#error "pq_legacy.h requires HNSWLIB_ENABLE_EDGE_QUANT_V0"
#endif

#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_estimation/types.h"
#include "hnswlib/edge_quant_v0_io.h"
#include "../event_format.h"
#include "../query_store.h"
#include "pq_packed_artifact.h"

namespace uq {

inline hnswlib::edge_estimation::EstimateStatus legacyStatus(
    hnswlib::V0BoundStatus status) {
    using hnswlib::edge_estimation::EstimateStatus;
    switch (status) {
        case hnswlib::V0BoundStatus::Valid: return EstimateStatus::Valid;
        case hnswlib::V0BoundStatus::ZeroLength: return EstimateStatus::ZeroLength;
        case hnswlib::V0BoundStatus::ExactOnly: return EstimateStatus::LegacyExactOnly;
        case hnswlib::V0BoundStatus::NumericFailure: return EstimateStatus::NonFinite;
        default: return EstimateStatus::UnsupportedRecord;
    }
}

// Thin score-level adapter: expression order and float LUT accumulation remain
// entirely inside the production raw_fast_v1 implementation.
class PqLegacyScoreKernel {
 public:
    explicit PqLegacyScoreKernel(
        const hnswlib::EdgeQuantV0ApproxQueryContext& query)
        : query_(query) {}

    hnswlib::edge_estimation::EdgeScore score(
        const hnswlib::V0EdgeRecordView& record,
        double d_current) const {
        const hnswlib::V0RawEstimateResult raw =
            query_.evaluateRawFast(record, d_current);
        return hnswlib::edge_estimation::EdgeScore(
            raw.approximate_squared_distance, legacyStatus(raw.status));
    }

 private:
    const hnswlib::EdgeQuantV0ApproxQueryContext& query_;
};

// Complete unified-evaluator wrapper for an existing validated V0 sidecar.
// The stable catalog edge order is the V0 source/slot order; every score call
// verifies that identity before exposing the record to raw_fast_v1.
class PqLegacyArtifactKernel {
 public:
    PqLegacyArtifactKernel(const std::filesystem::path& root,
                           std::shared_ptr<const QueryStore> queries,
                           const Header& event_header,
                           const char* expected_format = "uq-pq-legacy/1",
                           const char* expected_backend = "pq_legacy")
        : queries_(std::move(queries)), config_(readNativeConfig(root / "native.cfg")),
          sidecar_(hnswlib::loadV0Sidecar(checkedArtifactPath(
              root, config_.at("sidecar")).string())),
          view_(sidecar_.view()), current_query_(~uint64_t(0)) {
        require("format", expected_format); require("backend", expected_backend);
        require("coverage", "full_graph");
        const std::filesystem::path sidecar_path =
            checkedArtifactPath(root, config_.at("sidecar"));
        if (artifactSha256(sidecar_path) != config_.at("sidecar_sha256"))
            throw std::runtime_error("legacy sidecar artifact hash mismatch");
        if (!queries_ || queries_->dimension() != view_.header().dimension ||
            event_header.dimension != view_.header().dimension ||
            parseHexDigest(config_.at("catalog_identity")) != event_header.identity ||
            std::stoull(config_.at("edge_count")) != view_.header().directed_edge_count)
            throw std::runtime_error("legacy sidecar/dataset identity or shape mismatch");
        const size_t count = static_cast<size_t>(view_.header().codebook.size / 4U);
        native_codebook_.resize(count);
        for (size_t i = 0; i < count; ++i)
            native_codebook_[i] = view_.codebookCentroid(i);
    }

    void prepareQuery(uint64_t query_id) {
        query_context_.reset(new hnswlib::EdgeQuantV0ApproxQueryContext(
            queries_->query(query_id), view_.header(), native_codebook_.data()));
        current_query_ = query_id;
    }
    void prepareSource(const EventRecord&) {}
    hnswlib::V0RawEstimateResult rawScore(const EventRecord& event) {
        hnswlib::V0RawEstimateResult fallback;
        if (!query_context_ || current_query_ != event.query_id ||
            event.edge_id >= view_.header().directed_edge_count ||
            event.source_id >= view_.header().node_count)
            return fallback;
        const uint64_t first = view_.nodeOffsetNativeUnchecked(event.source_id);
        const uint64_t last = view_.nodeOffsetNativeUnchecked(
            static_cast<size_t>(event.source_id) + 1U);
        if (last < first || event.neighbor_slot >= last - first ||
            first + event.neighbor_slot != event.edge_id)
            return fallback;
        return query_context_->evaluateRawFast(
            view_.edgeRecordUnchecked(static_cast<size_t>(event.edge_id)), event.d_current);
    }
    double score(const EventRecord& event) {
        const hnswlib::V0RawEstimateResult result = rawScore(event);
        return result.valid() ? result.approximate_squared_distance
                              : std::numeric_limits<double>::quiet_NaN();
    }
    uint64_t backendBytes() const {
        return static_cast<uint64_t>(view_.fileSize()) + native_codebook_.size() * 4U;
    }
    uint64_t scratchBytes() const {
        return query_context_ ? query_context_->lutBytes() : 0U;
    }
    const hnswlib::V0SidecarView& sidecarView() const { return view_; }

 private:
    void require(const char* key, const char* value) const {
        if (config_.at(key) != value)
            throw std::runtime_error("unsupported legacy PQ artifact contract");
    }
    std::shared_ptr<const QueryStore> queries_;
    std::map<std::string, std::string> config_;
    hnswlib::V0OwnedSidecar sidecar_;
    hnswlib::V0SidecarView view_;
    std::vector<float> native_codebook_;
    std::unique_ptr<hnswlib::EdgeQuantV0ApproxQueryContext> query_context_;
    uint64_t current_query_;
};

}  // namespace uq
