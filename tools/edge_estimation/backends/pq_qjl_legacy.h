#pragma once

#if !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) || !defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
#error "pq_qjl_legacy.h requires V0 PQ and residual estimator support"
#endif

#include <cmath>

#include "pq_legacy.h"
#include "hnswlib/edge_quant_v0_residual.h"
#include "hnswlib/edge_quant_v0_residual_io.h"

namespace uq {

// The companion reader supplies packed signs, float32 scale, and float32
// offset. This adapter deliberately invokes the existing correction method
// instead of translating the correction through a dot-product abstraction.
class PqQjlLegacyScoreKernel {
 public:
    PqQjlLegacyScoreKernel(
        const hnswlib::EdgeQuantV0ApproxQueryContext& pq_query,
        const hnswlib::V0ResidualQueryContext& residual_query)
        : pq_query_(pq_query), residual_query_(residual_query) {}

    hnswlib::edge_estimation::EdgeScore score(
        const hnswlib::V0EdgeRecordView& record,
        double d_current,
        const uint8_t* packed_signs,
        float scale,
        float offset) const {
        const hnswlib::V0RawEstimateResult raw =
            pq_query_.evaluateRawFast(record, d_current);
        const hnswlib::edge_estimation::EstimateStatus status =
            legacyStatus(raw.status);
        if (status != hnswlib::edge_estimation::EstimateStatus::Valid)
            return hnswlib::edge_estimation::EdgeScore(0.0, status);
        const double corrected = residual_query_.correct(
            raw.approximate_squared_distance, packed_signs, scale, offset);
        if (!std::isfinite(corrected))
            return hnswlib::edge_estimation::EdgeScore(
                0.0, hnswlib::edge_estimation::EstimateStatus::NonFinite);
        return hnswlib::edge_estimation::EdgeScore(
            corrected, hnswlib::edge_estimation::EstimateStatus::Valid);
    }

 private:
    const hnswlib::EdgeQuantV0ApproxQueryContext& pq_query_;
    const hnswlib::V0ResidualQueryContext& residual_query_;
};

class PqQjlLegacyArtifactKernel {
 public:
    PqQjlLegacyArtifactKernel(const std::filesystem::path& root,
                              std::shared_ptr<const QueryStore> queries,
                              const Header& event_header)
        : queries_(queries), config_(readNativeConfig(root / "native.cfg")),
          pq_(root, std::move(queries), event_header,
              "uq-pq-qjl-legacy/1", "pq_qjl_legacy"),
          current_query_(~uint64_t(0)) {
        const std::filesystem::path sidecar_path =
            checkedArtifactPath(root, config_.at("sidecar"));
        const std::filesystem::path residual_path =
            checkedArtifactPath(root, config_.at("residual"));
        if (artifactSha256(residual_path) != config_.at("residual_sha256"))
            throw std::runtime_error("legacy residual artifact hash mismatch");
        hnswlib::V0ResidualIdentity expected;
        expected.index_sha = pq_.sidecarView().header().base_index_sha256;
        expected.sidecar_sha = hnswlib::v0ResidualFileSha(sidecar_path.string());
        expected.adjacency_sha = pq_.sidecarView().header().adjacency_sha256;
        residual_.reset(new hnswlib::V0ResidualCompanion(
            residual_path.string(), expected, event_header.dimension,
            pq_.sidecarView().header().directed_edge_count));
    }

    void prepareQuery(uint64_t query_id) {
        pq_.prepareQuery(query_id);
        residual_query_.reset(new hnswlib::V0ResidualQueryContext(
            queries_->query(query_id), residual_->matrix(), residual_->dimension(),
            residual_->bits()));
        current_query_ = query_id;
    }
    void prepareSource(const EventRecord& event) { pq_.prepareSource(event); }
    double score(const EventRecord& event) {
        if (!residual_query_ || current_query_ != event.query_id ||
            event.edge_id >= residual_->count())
            return std::numeric_limits<double>::quiet_NaN();
        const hnswlib::V0RawEstimateResult raw = pq_.rawScore(event);
        const uint8_t* record = residual_->record(event.edge_id);
        if (!raw.valid() || !residual_->valid(record))
            return std::numeric_limits<double>::quiet_NaN();
        const double corrected = residual_query_->correct(
            raw.approximate_squared_distance, record,
            residual_->scale(record), residual_->offset(record));
        return std::isfinite(corrected) ? corrected
                                        : std::numeric_limits<double>::quiet_NaN();
    }
    uint64_t backendBytes() const {
        return pq_.backendBytes() + residual_->storageBytes();
    }
    uint64_t scratchBytes() const {
        return pq_.scratchBytes() + static_cast<uint64_t>(residual_->bits()) * 20U;
    }

 private:
    std::shared_ptr<const QueryStore> queries_;
    std::map<std::string, std::string> config_;
    PqLegacyArtifactKernel pq_;
    std::unique_ptr<hnswlib::V0ResidualCompanion> residual_;
    std::unique_ptr<hnswlib::V0ResidualQueryContext> residual_query_;
    uint64_t current_query_;
};

}  // namespace uq
