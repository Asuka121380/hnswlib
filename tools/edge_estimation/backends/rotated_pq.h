#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_rotation.h"
#include "pq_packed_artifact.h"
#include "optional_backend.h"
#include "hnswlib/edge_estimation/score_bridge.h"
#include "../event_format.h"
#include "../query_store.h"
#include "../query_selection.h"

namespace uq {

class RotatedPqArtifactKernel {
 public:
    RotatedPqArtifactKernel(const std::filesystem::path& root,
                            std::shared_ptr<const QueryStore> queries,
                            const Header& event_header)
        : queries_(std::move(queries)), config_(readNativeConfig(root / "native.cfg")),
          dimension_(number("dimension")), m_(number("m")), nbits_(number("nbits")),
          dsub_(number("dsub")), edge_count_(number64("edge_count")),
          record_size_(number("record_size")), model_(m_, nbits_), current_query_(~uint64_t(0)),
          current_lut_(nullptr), batch_prepared_(false) {
        require("format", "uq-rotated-pq/1");
        require("coverage", "full_graph");
        require("rotation_layout", "row_major_r_times_column");
        require("rotation_bias", "none");
        if (config_.at("backend") != "opq" && config_.at("backend") != "jq")
            throw std::runtime_error("unsupported rotated-PQ backend");
        if (!queries_ || queries_->dimension() != dimension_ ||
            dimension_ != event_header.dimension || m_ * dsub_ != dimension_ ||
            parseHexDigest(config_.at("catalog_identity")) != event_header.identity ||
            record_size_ != 16U + model_.packedCodeBytes())
            throw std::runtime_error("artifact/dataset identity or shape mismatch");
        const std::filesystem::path codebook_path =
            checkedArtifactPath(root, config_.at("codebook"));
        const std::filesystem::path rotation_path =
            checkedArtifactPath(root, config_.at("rotation"));
        const std::filesystem::path records_path =
            checkedArtifactPath(root, config_.at("records"));
        if (artifactSha256(codebook_path) != config_.at("codebook_sha256") ||
            artifactSha256(rotation_path) != config_.at("rotation_sha256") ||
            artifactSha256(records_path) != config_.at("records_sha256"))
            throw std::runtime_error("artifact file hash mismatch");
        codebook_ = readArtifactFloats(codebook_path);
        rotation_ = readArtifactFloats(rotation_path);
        if (codebook_.size() != static_cast<size_t>(m_) * model_.centroidCount() * dsub_ ||
            rotation_.size() != static_cast<size_t>(dimension_) * dimension_)
            throw std::runtime_error("rotated-PQ model size mismatch");
        records_ = detail::readFile(records_path.string());
        if (records_.size() != edge_count_ * record_size_)
            throw std::runtime_error("rotated-PQ records size mismatch");
    }

    void prepareQuery(uint64_t query_id) {
        if (batch_prepared_) {
            const size_t slot = batch_slots_.at(query_id);
            current_lut_ = batch_luts_.data() + slot * lutSize();
            current_query_ = query_id;
            return;
        }
        const float* query = queries_->query(query_id);
        rotated_query_.assign(dimension_, 0.0f);
        for (uint32_t row = 0; row < dimension_; ++row) {
            float sum = 0.0f;
            const size_t offset = static_cast<size_t>(row) * dimension_;
            for (uint32_t column = 0; column < dimension_; ++column)
                sum += rotation_[offset + column] * query[column];
            rotated_query_[row] = sum;
        }
        lut_.assign(static_cast<size_t>(m_) * model_.centroidCount(), 0.0f);
        for (uint32_t sub = 0; sub < m_; ++sub)
            for (uint32_t code = 0; code < model_.centroidCount(); ++code) {
                float sum = 0.0f;
                const size_t center =
                    (static_cast<size_t>(sub) * model_.centroidCount() + code) * dsub_;
                for (uint32_t d = 0; d < dsub_; ++d)
                    sum += rotated_query_[sub * dsub_ + d] * codebook_[center + d];
                lut_[static_cast<size_t>(sub) * model_.centroidCount() + code] = sum;
            }
        current_lut_ = lut_.data();
        current_query_ = query_id;
    }

    void prepareQueryBatch(const std::vector<uint64_t>& query_ids, size_t batch_size) {
        if (query_ids.empty()) throw std::invalid_argument("query batch must not be empty");
        if (batch_size == 0U) throw std::invalid_argument("query batch size must be positive");
        batch_prepared_ = false;
        current_query_ = ~uint64_t(0);
        current_lut_ = nullptr;
        batch_slots_.reset(query_ids, static_cast<size_t>(queries_->count()));
        batch_luts_.assign(query_ids.size() * lutSize(), 0.0f);
        const size_t effective_batch = std::min(batch_size, query_ids.size());
        batch_queries_.resize(effective_batch * dimension_);
        batch_rotated_queries_.resize(effective_batch * dimension_);
        for (size_t begin = 0; begin < query_ids.size(); begin += effective_batch) {
            const size_t count = std::min(effective_batch, query_ids.size() - begin);
            for (size_t row = 0; row < count; ++row) {
                const float* source = queries_->query(query_ids[begin + row]);
                std::copy(source, source + dimension_,
                          batch_queries_.begin() + row * dimension_);
            }
            detail::rotateQueryBatch(batch_queries_.data(), count, dimension_,
                                     rotation_.data(), batch_rotated_queries_.data());
            detail::buildPqInnerProductTablesBatch(
                batch_rotated_queries_.data(), count, dimension_, m_,
                model_.centroidCount(), dsub_, codebook_.data(),
                batch_luts_.data() + begin * lutSize());
        }
        prepared_query_count_ = query_ids.size();
        prepared_batch_size_ = effective_batch;
        batch_prepared_ = true;
    }

    void clearPreparedQueryBatch() {
        batch_prepared_ = false;
        current_query_ = ~uint64_t(0);
        current_lut_ = nullptr;
    }

    const char* batchPreparationEngine() const { return detail::batchRotationEngine(); }
    size_t preparedQueryCount() const { return prepared_query_count_; }
    size_t preparedBatchSize() const { return prepared_batch_size_; }
    const char* method() const { return "opq"; }
    void verifyIndex(const std::filesystem::path& path) const {
        const auto found = config_.find("index_sha256");
        if (found != config_.end() && artifactSha256(path) != found->second)
            throw std::runtime_error("OPQ index identity mismatch");
    }

    void prepareSource(const EventRecord&) {}

    bool zeroLengthEdge(uint64_t id) const {
        return id < edge_count_ && detail::readF64(records_.data() + id * record_size_) == 0.0;
    }
    double score(const EventRecord& event) {
        if (current_query_ != event.query_id || event.edge_id >= edge_count_)
            return std::numeric_limits<double>::quiet_NaN();
        const uint8_t* record = records_.data() + event.edge_id * record_size_;
        const double length = detail::readF64(record);
        const double anchor = detail::readF64(record + 8U);
        if (!(length > 0.0) || !std::isfinite(length) || !std::isfinite(anchor))
            return std::numeric_limits<double>::quiet_NaN();
        const hnswlib::edge_estimation::DotEstimate qdot =
            model_.estimate(current_lut_, lutSize(), record + 16U,
                            model_.packedCodeBytes());
        if (!qdot.valid()) return std::numeric_limits<double>::quiet_NaN();
        const hnswlib::edge_estimation::EdgeScore score =
            hnswlib::edge_estimation::bridgeDotToSquaredDistance(
                event.d_current, length,
                hnswlib::edge_estimation::DotEstimate(qdot.value - anchor, qdot.status));
        return score.valid() ? score.squared_distance
                             : std::numeric_limits<double>::quiet_NaN();
    }

    uint64_t backendBytes() const {
        return static_cast<uint64_t>(codebook_.size() + rotation_.size()) * 4U + records_.size();
    }
    uint64_t scratchBytes() const {
        return static_cast<uint64_t>(
            lut_.capacity() + rotated_query_.capacity() + batch_luts_.capacity() +
            batch_queries_.capacity() + batch_rotated_queries_.capacity()) * 4U +
            batch_slots_.bytes();
    }

 private:
    uint32_t number(const char* key) const {
        return static_cast<uint32_t>(std::stoul(config_.at(key)));
    }
    uint64_t number64(const char* key) const { return std::stoull(config_.at(key)); }
    void require(const char* key, const char* value) const {
        if (config_.at(key) != value) throw std::runtime_error("unsupported artifact contract");
    }
    size_t lutSize() const {
        return static_cast<size_t>(m_) * model_.centroidCount();
    }
    static size_t noBatchSlot() { return std::numeric_limits<size_t>::max(); }

    std::shared_ptr<const QueryStore> queries_;
    std::map<std::string, std::string> config_;
    uint32_t dimension_, m_, nbits_, dsub_;
    uint64_t edge_count_;
    uint32_t record_size_;
    PackedPqModel model_;
    uint64_t current_query_;
    const float* current_lut_;
    bool batch_prepared_;
    size_t prepared_query_count_ = 0U;
    size_t prepared_batch_size_ = 0U;
    std::vector<float> codebook_, rotation_, rotated_query_, lut_;
    std::vector<float> batch_queries_, batch_rotated_queries_, batch_luts_;
    QuerySlots batch_slots_;
    std::vector<uint8_t> records_;
};

}  // namespace uq
