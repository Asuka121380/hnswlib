#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace hnswlib {

// Small dependency-free SHA-256 implementation for reproducible V0
// fingerprints. This is an integrity/checksum primitive, not an
// authentication boundary.
class EdgeQuantV0Sha256 {
 public:
    EdgeQuantV0Sha256()
        : block_size_(0),
          total_bits_(0) {
        state_[0] = 0x6a09e667U;
        state_[1] = 0xbb67ae85U;
        state_[2] = 0x3c6ef372U;
        state_[3] = 0xa54ff53aU;
        state_[4] = 0x510e527fU;
        state_[5] = 0x9b05688cU;
        state_[6] = 0x1f83d9abU;
        state_[7] = 0x5be0cd19U;
    }

    void update(const void* input, size_t length) {
        const uint8_t* bytes = static_cast<const uint8_t*>(input);
        for (size_t i = 0; i < length; ++i) {
            block_[block_size_++] = bytes[i];
            if (block_size_ == sizeof(block_)) {
                transform();
                total_bits_ += 512U;
                block_size_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        size_t i = block_size_;
        block_[i++] = 0x80U;

        if (i > 56U) {
            while (i < 64U) {
                block_[i++] = 0U;
            }
            transform();
            i = 0;
        }
        while (i < 56U) {
            block_[i++] = 0U;
        }

        total_bits_ += static_cast<uint64_t>(block_size_) * 8U;
        for (size_t byte = 0; byte < 8U; ++byte) {
            block_[63U - byte] =
                static_cast<uint8_t>((total_bits_ >> (byte * 8U)) & 0xffU);
        }
        transform();

        std::array<uint8_t, 32> digest;
        for (size_t word = 0; word < 8U; ++word) {
            digest[word * 4U] =
                static_cast<uint8_t>((state_[word] >> 24U) & 0xffU);
            digest[word * 4U + 1U] =
                static_cast<uint8_t>((state_[word] >> 16U) & 0xffU);
            digest[word * 4U + 2U] =
                static_cast<uint8_t>((state_[word] >> 8U) & 0xffU);
            digest[word * 4U + 3U] =
                static_cast<uint8_t>(state_[word] & 0xffU);
        }
        return digest;
    }

 private:
    static uint32_t rotateRight(uint32_t value, uint32_t shift) {
        return (value >> shift) | (value << (32U - shift));
    }

    void transform() {
        static const uint32_t constants[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
        };

        uint32_t schedule[64];
        for (size_t i = 0; i < 16U; ++i) {
            const size_t offset = i * 4U;
            schedule[i] =
                (static_cast<uint32_t>(block_[offset]) << 24U) |
                (static_cast<uint32_t>(block_[offset + 1U]) << 16U) |
                (static_cast<uint32_t>(block_[offset + 2U]) << 8U) |
                static_cast<uint32_t>(block_[offset + 3U]);
        }
        for (size_t i = 16U; i < 64U; ++i) {
            const uint32_t s0 =
                rotateRight(schedule[i - 15U], 7U) ^
                rotateRight(schedule[i - 15U], 18U) ^
                (schedule[i - 15U] >> 3U);
            const uint32_t s1 =
                rotateRight(schedule[i - 2U], 17U) ^
                rotateRight(schedule[i - 2U], 19U) ^
                (schedule[i - 2U] >> 10U);
            schedule[i] =
                schedule[i - 16U] + s0 + schedule[i - 7U] + s1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (size_t i = 0; i < 64U; ++i) {
            const uint32_t sigma1 =
                rotateRight(e, 6U) ^ rotateRight(e, 11U) ^
                rotateRight(e, 25U);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 =
                h + sigma1 + choose + constants[i] + schedule[i];
            const uint32_t sigma0 =
                rotateRight(a, 2U) ^ rotateRight(a, 13U) ^
                rotateRight(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sigma0 + majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    uint8_t block_[64];
    size_t block_size_;
    uint64_t total_bits_;
    uint32_t state_[8];
};

inline std::string edgeQuantV0Sha256Hex(
    const std::array<uint8_t, 32>& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (size_t i = 0; i < digest.size(); ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
}

inline void edgeQuantV0Sha256UpdateUint32LittleEndian(
    EdgeQuantV0Sha256& sha,
    uint32_t value) {
    uint8_t bytes[4];
    for (size_t i = 0; i < 4U; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xffU);
    }
    sha.update(bytes, sizeof(bytes));
}

inline void edgeQuantV0Sha256UpdateUint64LittleEndian(
    EdgeQuantV0Sha256& sha,
    uint64_t value) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8U; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xffU);
    }
    sha.update(bytes, sizeof(bytes));
}

}  // namespace hnswlib
