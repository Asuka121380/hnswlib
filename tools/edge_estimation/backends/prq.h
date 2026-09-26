#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pq_packed_artifact.h"
#include "hnswlib/edge_estimation/score_bridge.h"
#include "../event_format.h"
#include "../query_store.h"

namespace uq {

class PrqArtifactKernel {
 public:
    PrqArtifactKernel(const std::filesystem::path& root,
                      std::shared_ptr<const QueryStore> queries,
                      const Header& event_header)
        : queries_(std::move(queries)), config_(readNativeConfig(root / "native.cfg")),
          dimension_(number("dimension")), nsplits_(number("nsplits")),
          stages_(number("stages_per_split")), m_(number("m")),
          nbits_(number("nbits")), dsub_(number("dsub")),
          edge_count_(number64("edge_count")), record_size_(number("record_size")),
          model_(m_, nbits_), current_query_(~uint64_t(0)) {
        require("format", "uq-prq/1"); require("backend", "prq");
        require("coverage", "full_graph"); require("norm_bits", "0");
        require("codebook_layout", "split_stage_centroid_dimension");
        if (!queries_ || queries_->dimension() != dimension_ ||
            dimension_ != event_header.dimension || nsplits_ * dsub_ != dimension_ ||
            nsplits_ * stages_ != m_ ||
            parseHexDigest(config_.at("catalog_identity")) != event_header.identity ||
            record_size_ != 16U + model_.packedCodeBytes())
            throw std::runtime_error("PRQ artifact/dataset identity or shape mismatch");
        const std::filesystem::path codebook_path =
            checkedArtifactPath(root, config_.at("codebook"));
        const std::filesystem::path records_path =
            checkedArtifactPath(root, config_.at("records"));
        if (artifactSha256(codebook_path) != config_.at("codebook_sha256") ||
            artifactSha256(records_path) != config_.at("records_sha256"))
            throw std::runtime_error("artifact file hash mismatch");
        codebook_ = readArtifactFloats(codebook_path);
        if (codebook_.size() != static_cast<size_t>(m_) * model_.centroidCount() * dsub_)
            throw std::runtime_error("PRQ codebook size mismatch");
        records_ = detail::readFile(records_path.string());
        if (records_.size() != edge_count_ * record_size_)
            throw std::runtime_error("PRQ records size mismatch");
    }

    void prepareQuery(uint64_t query_id) {
        const float* query = queries_->query(query_id);
        lut_.assign(static_cast<size_t>(m_) * model_.centroidCount(), 0.0f);
        for (uint32_t split = 0; split < nsplits_; ++split)
            for (uint32_t stage = 0; stage < stages_; ++stage) {
                const uint32_t book = split * stages_ + stage;
                for (uint32_t code = 0; code < model_.centroidCount(); ++code) {
                    float sum = 0.0f;
                    const size_t center =
                        (static_cast<size_t>(book) * model_.centroidCount() + code) * dsub_;
                    for (uint32_t d = 0; d < dsub_; ++d)
                        sum += query[split * dsub_ + d] * codebook_[center + d];
                    lut_[static_cast<size_t>(book) * model_.centroidCount() + code] = sum;
                }
            }
        current_query_ = query_id;
    }
    void prepareSource(const EventRecord&) {}
    double score(const EventRecord& event) {
        if (current_query_ != event.query_id || event.edge_id >= edge_count_)
            return std::numeric_limits<double>::quiet_NaN();
        const uint8_t* record = records_.data() + event.edge_id * record_size_;
        const double length = detail::readF64(record);
        const double anchor = detail::readF64(record + 8U);
        if (!(length > 0.0) || !std::isfinite(length) || !std::isfinite(anchor))
            return std::numeric_limits<double>::quiet_NaN();
        const hnswlib::edge_estimation::DotEstimate qdot =
            model_.estimate(lut_, record + 16U, model_.packedCodeBytes());
        if (!qdot.valid()) return std::numeric_limits<double>::quiet_NaN();
        const hnswlib::edge_estimation::EdgeScore score =
            hnswlib::edge_estimation::bridgeDotToSquaredDistance(
                event.d_current, length,
                hnswlib::edge_estimation::DotEstimate(qdot.value - anchor, qdot.status));
        return score.valid() ? score.squared_distance
                             : std::numeric_limits<double>::quiet_NaN();
    }
    uint64_t backendBytes() const { return codebook_.size() * 4U + records_.size(); }
    uint64_t scratchBytes() const { return lut_.capacity() * 4U; }

 private:
    uint32_t number(const char* key) const {
        return static_cast<uint32_t>(std::stoul(config_.at(key)));
    }
    uint64_t number64(const char* key) const { return std::stoull(config_.at(key)); }
    void require(const char* key, const char* value) const {
        if (config_.at(key) != value) throw std::runtime_error("unsupported PRQ artifact contract");
    }
    std::shared_ptr<const QueryStore> queries_;
    std::map<std::string, std::string> config_;
    uint32_t dimension_, nsplits_, stages_, m_, nbits_, dsub_;
    uint64_t edge_count_;
    uint32_t record_size_;
    PackedPqModel model_;
    uint64_t current_query_;
    std::vector<float> codebook_, lut_;
    std::vector<uint8_t> records_;
};

}  // namespace uq
