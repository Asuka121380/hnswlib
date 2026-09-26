#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pq_packed.h"
#include "hnswlib/edge_estimation/score_bridge.h"
#include "hnswlib/edge_quant_v0_checksum.h"
#include "../event_format.h"

namespace uq {

inline std::map<std::string, std::string> readNativeConfig(const std::filesystem::path& path) {
    std::ifstream input(path); std::map<std::string, std::string> values; std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const size_t equal = line.find('=');
        if (equal == std::string::npos || equal == 0U || equal + 1U >= line.size())
            throw std::runtime_error("invalid native artifact config");
        if (!values.emplace(line.substr(0, equal), line.substr(equal + 1U)).second)
            throw std::runtime_error("duplicate native artifact key");
    }
    if (!input.eof()) throw std::runtime_error("failed to read native artifact config");
    return values;
}

inline std::array<uint8_t, 32> parseHexDigest(const std::string& text) {
    if (text.size() != 64U) throw std::runtime_error("invalid digest length");
    std::array<uint8_t, 32> result{};
    for (size_t i = 0; i < 32U; ++i) {
        const std::string byte = text.substr(i * 2U, 2U);
        size_t used = 0U; const unsigned long value = std::stoul(byte, &used, 16);
        if (used != 2U) throw std::runtime_error("invalid digest hex");
        result[i] = static_cast<uint8_t>(value);
    }
    return result;
}

inline std::filesystem::path checkedArtifactPath(
    const std::filesystem::path& root, const std::string& relative) {
    const std::filesystem::path value(relative);
    if (relative.empty() || value.is_absolute() || value.has_root_name())
        throw std::runtime_error("artifact path must be relative");
    for (const std::filesystem::path& part : value)
        if (part == "..") throw std::runtime_error("artifact path escapes root");
    return root / value;
}

inline std::string artifactSha256(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash artifact file");
    hnswlib::EdgeQuantV0Sha256 sha;
    std::array<char, 1U << 16U> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) sha.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("artifact file hash read failed");
    return hnswlib::edgeQuantV0Sha256Hex(sha.final());
}

class PackedPqArtifactKernel {
 public:
    PackedPqArtifactKernel(const std::filesystem::path& root,
                           const std::filesystem::path& queries,
                           const Header& event_header)
        : root_(root), queries_(queries), config_(readNativeConfig(root / "native.cfg")),
          dimension_(number("dimension")), m_(number("m")), nbits_(number("nbits")),
          dsub_(number("dsub")), edge_count_(number64("edge_count")),
          record_size_(number("record_size")), model_(m_, nbits_), current_query_(~uint64_t(0)) {
        require("format", "uq-pq-packed/1"); require("coverage", "full_graph");
        if (dimension_ != event_header.dimension || m_ * dsub_ != dimension_ ||
            parseHexDigest(config_.at("catalog_identity")) != event_header.identity ||
            record_size_ != 16U + model_.packedCodeBytes())
            throw std::runtime_error("artifact/dataset identity or shape mismatch");
        const std::filesystem::path codebook_path =
            checkedArtifactPath(root, config_.at("codebook"));
        const std::filesystem::path records_path =
            checkedArtifactPath(root, config_.at("records"));
        if (artifactSha256(codebook_path) != config_.at("codebook_sha256") ||
            artifactSha256(records_path) != config_.at("records_sha256"))
            throw std::runtime_error("artifact file hash mismatch");
        codebook_ = readFloats(codebook_path);
        if (codebook_.size() != static_cast<size_t>(m_) * model_.centroidCount() * dsub_)
            throw std::runtime_error("packed PQ codebook size mismatch");
        records_ = detail::readFile(records_path.string());
        if (records_.size() != edge_count_ * record_size_)
            throw std::runtime_error("packed PQ records size mismatch");
        const uint64_t stride = 4U + static_cast<uint64_t>(dimension_) * 4U;
        if (std::filesystem::file_size(queries_) % stride)
            throw std::runtime_error("query fvecs file size mismatch");
    }

    void prepareQuery(uint64_t query_id) {
        const uint64_t stride = 4U + static_cast<uint64_t>(dimension_) * 4U;
        if (query_id >= std::filesystem::file_size(queries_) / stride)
            throw std::runtime_error("query id out of range");
        std::ifstream input(queries_, std::ios::binary);
        input.seekg(static_cast<std::streamoff>(query_id * stride));
        uint32_t stored = 0U; input.read(reinterpret_cast<char*>(&stored), 4);
        std::vector<float> query(dimension_); input.read(reinterpret_cast<char*>(query.data()), dimension_ * 4U);
        if (!input || stored != dimension_) throw std::runtime_error("query fvec read failed");
        lut_.assign(static_cast<size_t>(m_) * model_.centroidCount(), 0.0f);
        for (uint32_t sub = 0; sub < m_; ++sub)
            for (uint32_t code = 0; code < model_.centroidCount(); ++code) {
                float sum = 0.0f;
                const size_t center = (static_cast<size_t>(sub) * model_.centroidCount() + code) * dsub_;
                for (uint32_t d = 0; d < dsub_; ++d)
                    sum += query[sub * dsub_ + d] * codebook_[center + d];
                lut_[static_cast<size_t>(sub) * model_.centroidCount() + code] = sum;
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
    uint32_t number(const char* key) const { return static_cast<uint32_t>(std::stoul(config_.at(key))); }
    uint64_t number64(const char* key) const { return std::stoull(config_.at(key)); }
    void require(const char* key, const char* value) const {
        if (config_.at(key) != value) throw std::runtime_error("unsupported artifact contract");
    }
    static std::vector<float> readFloats(const std::filesystem::path& path) {
        const std::vector<uint8_t> bytes = detail::readFile(path.string());
        if (bytes.size() % 4U) throw std::runtime_error("float file size mismatch");
        std::vector<float> values(bytes.size() / 4U);
        for (size_t i = 0; i < values.size(); ++i) {
            const uint32_t raw = detail::readU32(bytes.data() + i * 4U);
            std::memcpy(&values[i], &raw, 4U);
            if (!std::isfinite(values[i])) throw std::runtime_error("non-finite codebook value");
        }
        return values;
    }
    std::filesystem::path root_, queries_;
    std::map<std::string, std::string> config_;
    uint32_t dimension_, m_, nbits_, dsub_; uint64_t edge_count_; uint32_t record_size_;
    PackedPqModel model_; uint64_t current_query_;
    std::vector<float> codebook_, lut_; std::vector<uint8_t> records_;
};

}  // namespace uq
