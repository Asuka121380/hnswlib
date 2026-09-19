#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "edge_quant_v0_io.h"
#include "edge_quant_v0_residual.h"

namespace hnswlib {

static const uint32_t V0_RESIDUAL_VERSION = 1U;
static const size_t V0_RESIDUAL_HEADER_SIZE = 208U;

inline V0Sha256Digest v0ResidualFileSha(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open residual input: " + path);
    EdgeQuantV0Sha256 sha;
    std::vector<char> block(1U << 20);
    while (input) {
        input.read(block.data(), static_cast<std::streamsize>(block.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) sha.update(block.data(), static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("cannot read residual input: " + path);
    return sha.final();
}

inline uint32_t v0ResidualStride(uint32_t bits) {
    if (!bits || bits % 8U || bits > 256U)
        throw std::invalid_argument("residual bits must be a multiple of 8 in [8,256]");
    return bits / 8U + 9U;  // signs, float32 scale/offset, validity byte
}

struct V0ResidualIdentity {
    V0Sha256Digest index_sha;
    V0Sha256Digest sidecar_sha;
    V0Sha256Digest adjacency_sha;
};

class V0ResidualCompanion {
 public:
    V0ResidualCompanion(const std::string& path,
                        const V0ResidualIdentity& expected,
                        uint32_t expected_dimension, uint64_t expected_records) {
        std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open residual companion");
        const std::streamoff size = input.tellg();
        if (size < static_cast<std::streamoff>(V0_RESIDUAL_HEADER_SIZE))
            throw std::runtime_error("truncated residual companion");
        bytes_.resize(static_cast<size_t>(size));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(bytes_.data()), size);
        if (!input) throw std::runtime_error("truncated residual companion");
        const uint8_t* h = bytes_.data();
        if (std::memcmp(h, "V0RES001", 8U) != 0 ||
            edge_quant_v0_detail::readUint32LittleEndian(h + 8U) != V0_RESIDUAL_VERSION ||
            edge_quant_v0_detail::readUint32LittleEndian(h + 12U) != V0_RESIDUAL_HEADER_SIZE ||
            h[200U] != 1U)
            throw std::runtime_error("invalid or incomplete residual companion header");
        dimension_ = edge_quant_v0_detail::readUint32LittleEndian(h + 16U);
        bits_ = edge_quant_v0_detail::readUint32LittleEndian(h + 20U);
        seed_ = edge_quant_v0_detail::readUint32LittleEndian(h + 24U);
        const uint32_t stride = edge_quant_v0_detail::readUint32LittleEndian(h + 28U);
        count_ = edge_quant_v0_detail::readUint64LittleEndian(h + 32U);
        if (dimension_ != expected_dimension || count_ != expected_records ||
            stride != v0ResidualStride(bits_) ||
            std::memcmp(h + 40U, expected.index_sha.data(), 32U) ||
            std::memcmp(h + 72U, expected.sidecar_sha.data(), 32U) ||
            std::memcmp(h + 104U, expected.adjacency_sha.data(), 32U))
            throw std::runtime_error("residual companion identity or layout mismatch");
        const uint64_t matrix_bytes = static_cast<uint64_t>(dimension_) * bits_ * 4U;
        const uint64_t payload_bytes = count_ * stride;
        if (count_ && payload_bytes / count_ != stride)
            throw std::runtime_error("residual payload size overflow");
        if (matrix_bytes > bytes_.size() - V0_RESIDUAL_HEADER_SIZE ||
            payload_bytes != bytes_.size() - V0_RESIDUAL_HEADER_SIZE - matrix_bytes)
            throw std::runtime_error("residual companion byte count mismatch");
        const uint8_t* matrix_bytes_ptr = h + V0_RESIDUAL_HEADER_SIZE;
        const uint8_t* payload = matrix_bytes_ptr + matrix_bytes;
        EdgeQuantV0Sha256 sha;
        sha.update(matrix_bytes_ptr, static_cast<size_t>(matrix_bytes));
        if (sha.final() != digestAt(h + 136U))
            throw std::runtime_error("residual matrix SHA mismatch");
        EdgeQuantV0Sha256 payload_sha;
        payload_sha.update(payload, static_cast<size_t>(payload_bytes));
        if (payload_sha.final() != digestAt(h + 168U))
            throw std::runtime_error("residual payload SHA mismatch");
        matrix_.resize(static_cast<size_t>(dimension_) * bits_);
        for (size_t i = 0; i < matrix_.size(); ++i) {
            matrix_[i] = edge_quant_v0_detail::readFloat32LittleEndian(
                matrix_bytes_ptr + i * 4U);
            if (!std::isfinite(matrix_[i]))
                throw std::runtime_error("nonfinite residual matrix");
        }
        payload_ = payload;
        stride_ = stride;
    }

    uint32_t dimension() const { return dimension_; }
    uint32_t bits() const { return bits_; }
    uint32_t seed() const { return seed_; }
    uint64_t count() const { return count_; }
    size_t storageBytes() const { return bytes_.size() + matrix_.size() * 4U; }
    const float* matrix() const { return matrix_.data(); }
    const uint8_t* record(uint64_t ordinal) const {
        if (ordinal >= count_) throw std::out_of_range("residual edge ordinal");
        return payload_ + ordinal * stride_;
    }
    bool valid(const uint8_t* record) const { return record[stride_ - 1U] == 1U; }
    float scale(const uint8_t* record) const {
        return edge_quant_v0_detail::readFloat32LittleEndian(record + bits_ / 8U);
    }
    float offset(const uint8_t* record) const {
        return edge_quant_v0_detail::readFloat32LittleEndian(record + bits_ / 8U + 4U);
    }

 private:
    static V0Sha256Digest digestAt(const uint8_t* ptr) {
        V0Sha256Digest digest;
        std::copy(ptr, ptr + 32U, digest.begin());
        return digest;
    }
    uint32_t dimension_ = 0U, bits_ = 0U, seed_ = 0U, stride_ = 0U;
    uint64_t count_ = 0U;
    std::vector<uint8_t> bytes_;
    std::vector<float> matrix_;
    const uint8_t* payload_ = nullptr;
};

class V0ResidualCompanionWriter {
 public:
    V0ResidualCompanionWriter(const std::string& path,
                              const V0ResidualIdentity& identity,
                              const std::vector<float>& matrix,
                              uint32_t dimension, uint32_t bits,
                              uint32_t seed, uint64_t count,
                              bool resume = false)
        : path_(path), temp_(path + ".partial"),
          checkpoint_(path + ".checkpoint"), bits_(bits), count_(count),
          stride_(v0ResidualStride(bits)) {
        if (!dimension || matrix.size() !=
            static_cast<size_t>(dimension) * bits)
            throw std::invalid_argument("invalid residual writer inputs");
        header_.resize(V0_RESIDUAL_HEADER_SIZE, 0U);
        std::memcpy(header_.data(), "V0RES001", 8U);
        put32(8U, V0_RESIDUAL_VERSION);
        put32(12U, static_cast<uint32_t>(V0_RESIDUAL_HEADER_SIZE));
        put32(16U, dimension); put32(20U, bits);
        put32(24U, seed); put32(28U, stride_); put64(32U, count);
        std::memcpy(header_.data() + 40U, identity.index_sha.data(), 32U);
        std::memcpy(header_.data() + 72U, identity.sidecar_sha.data(), 32U);
        std::memcpy(header_.data() + 104U, identity.adjacency_sha.data(), 32U);
        std::vector<uint8_t> matrix_bytes;
        matrix_bytes.reserve(matrix.size() * 4U);
        for (size_t i = 0; i < matrix.size(); ++i) {
            if (!std::isfinite(matrix[i]))
                throw std::invalid_argument("nonfinite residual matrix");
            edge_quant_v0_detail::appendFloat32LittleEndian(matrix_bytes, matrix[i]);
        }
        matrix_sha_.update(matrix_bytes.data(), matrix_bytes.size());
        if (resume) {
            resumePartial(matrix_bytes);
            return;
        }
        output_.open(temp_.c_str(), std::ios::binary | std::ios::trunc);
        if (!output_) throw std::runtime_error("cannot create residual partial file");
        output_.write(reinterpret_cast<const char*>(header_.data()), header_.size());
        output_.write(reinterpret_cast<const char*>(matrix_bytes.data()),
                      static_cast<std::streamsize>(matrix_bytes.size()));
        if (!output_) throw std::runtime_error("cannot write residual matrix");
        checkpoint();
    }

    uint64_t written() const { return written_; }

    void checkpoint() {
        output_.flush();
        if (!output_) throw std::runtime_error("cannot flush residual partial file");
        EdgeQuantV0Sha256 snapshot = payload_sha_;
        const V0Sha256Digest digest = snapshot.final();
        std::ofstream state(checkpoint_.c_str(), std::ios::binary | std::ios::trunc);
        state.write("V0RCP001", 8U);
        uint8_t count_bytes[8];
        for (size_t i = 0; i < 8U; ++i)
            count_bytes[i] = static_cast<uint8_t>(written_ >> (i * 8U));
        state.write(reinterpret_cast<const char*>(count_bytes), 8U);
        state.write(reinterpret_cast<const char*>(digest.data()), 32U);
        state.close();
        if (!state) throw std::runtime_error("cannot write residual checkpoint");
    }

    void append(const std::vector<uint8_t>& signs, float scale,
                float offset, bool valid) {
        if (written_ >= count_ || signs.size() != bits_ / 8U ||
            (valid && (!std::isfinite(scale) || !std::isfinite(offset))))
            throw std::invalid_argument("invalid residual record");
        std::vector<uint8_t> record(signs);
        edge_quant_v0_detail::appendFloat32LittleEndian(record, valid ? scale : 0.0f);
        edge_quant_v0_detail::appendFloat32LittleEndian(record, valid ? offset : 0.0f);
        record.push_back(valid ? 1U : 0U);
        output_.write(reinterpret_cast<const char*>(record.data()), record.size());
        if (!output_) throw std::runtime_error("cannot write residual record");
        payload_sha_.update(record.data(), record.size());
        ++written_;
    }

    void finish() {
        if (written_ != count_) throw std::runtime_error("incomplete residual payload");
        const V0Sha256Digest matrix_digest = matrix_sha_.final();
        const V0Sha256Digest payload_digest = payload_sha_.final();
        std::memcpy(header_.data() + 136U, matrix_digest.data(), 32U);
        std::memcpy(header_.data() + 168U, payload_digest.data(), 32U);
        header_[200U] = 1U;
        output_.close();
        std::fstream final(temp_.c_str(), std::ios::binary | std::ios::in | std::ios::out);
        final.write(reinterpret_cast<const char*>(header_.data()), header_.size());
        final.flush();
        if (!final) throw std::runtime_error("cannot finalize residual companion");
        final.close();
        if (std::rename(temp_.c_str(), path_.c_str()) != 0)
            throw std::runtime_error("cannot rename residual companion");
        std::remove(checkpoint_.c_str());
    }

 private:
    void resumePartial(const std::vector<uint8_t>& matrix_bytes) {
        std::ifstream state(checkpoint_.c_str(), std::ios::binary | std::ios::ate);
        if (!state || state.tellg() != 48)
            throw std::runtime_error("missing or malformed residual checkpoint");
        state.seekg(0);
        uint8_t checkpoint[48];
        state.read(reinterpret_cast<char*>(checkpoint), sizeof(checkpoint));
        if (!state || std::memcmp(checkpoint, "V0RCP001", 8U))
            throw std::runtime_error("invalid residual checkpoint");
        const uint64_t checkpoint_count =
            edge_quant_v0_detail::readUint64LittleEndian(checkpoint + 8U);
        if (checkpoint_count > count_ ||
            checkpoint_count > (std::numeric_limits<uint64_t>::max() -
                V0_RESIDUAL_HEADER_SIZE - matrix_bytes.size()) / stride_)
            throw std::runtime_error("residual checkpoint count mismatch");
        std::ifstream partial(temp_.c_str(), std::ios::binary | std::ios::ate);
        const uint64_t expected_size = V0_RESIDUAL_HEADER_SIZE +
            matrix_bytes.size() + checkpoint_count * stride_;
        if (!partial || static_cast<uint64_t>(partial.tellg()) != expected_size)
            throw std::runtime_error("residual partial size differs from checkpoint");
        partial.seekg(0);
        std::vector<uint8_t> existing_header(V0_RESIDUAL_HEADER_SIZE);
        partial.read(reinterpret_cast<char*>(existing_header.data()),
                     static_cast<std::streamsize>(existing_header.size()));
        if (!partial || std::memcmp(existing_header.data(), header_.data(),
                                   header_.size()))
            throw std::runtime_error("residual partial input identity mismatch");
        std::vector<uint8_t> existing_matrix(matrix_bytes.size());
        partial.read(reinterpret_cast<char*>(existing_matrix.data()),
                     static_cast<std::streamsize>(existing_matrix.size()));
        if (!partial || existing_matrix != matrix_bytes)
            throw std::runtime_error("residual partial matrix mismatch");
        std::vector<char> block(1U << 20);
        uint64_t remaining = checkpoint_count * stride_;
        while (remaining) {
            const size_t size = static_cast<size_t>(std::min<uint64_t>(
                remaining, block.size()));
            partial.read(block.data(), static_cast<std::streamsize>(size));
            if (!partial) throw std::runtime_error("truncated residual partial payload");
            payload_sha_.update(block.data(), size);
            remaining -= size;
        }
        EdgeQuantV0Sha256 snapshot = payload_sha_;
        const V0Sha256Digest digest = snapshot.final();
        if (std::memcmp(digest.data(), checkpoint + 16U, 32U))
            throw std::runtime_error("residual checkpoint payload SHA mismatch");
        written_ = checkpoint_count;
        output_.open(temp_.c_str(), std::ios::binary | std::ios::app);
        if (!output_) throw std::runtime_error("cannot append residual partial file");
    }
    void put32(size_t pos, uint32_t value) {
        for (size_t i = 0; i < 4U; ++i)
            header_[pos + i] = static_cast<uint8_t>(value >> (i * 8U));
    }
    void put64(size_t pos, uint64_t value) {
        for (size_t i = 0; i < 8U; ++i)
            header_[pos + i] = static_cast<uint8_t>(value >> (i * 8U));
    }
    std::string path_, temp_, checkpoint_;
    uint32_t bits_;
    uint64_t count_, written_ = 0U;
    uint32_t stride_;
    std::ofstream output_;
    std::vector<uint8_t> header_;
    EdgeQuantV0Sha256 matrix_sha_, payload_sha_;
};

}  // namespace hnswlib
