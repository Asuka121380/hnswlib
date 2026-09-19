#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/edge_quant_v0_residual_io.h"

int main() {
    const float query[] = {1, 2, 3, 4};
    const float matrix[] = {1, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 1};
    hnswlib::V0ResidualQueryContext context(query, matrix, 4U, 4U);
    const uint8_t signs[] = {0x05U};
    assert(std::fabs(context.signedDot(signs) - (-2.0)) < 1e-6);
    assert(std::fabs(context.correct(10.0, signs, 0.5f, 1.0f) - 12.0) < 1e-6);
    hnswlib::V0ResidualPruningConfig config;
    assert(config.valid());
    config.threshold_mode = true; config.theta = 0.8;
    assert(config.valid());
    hnswlib::V0ResidualIdentity id;
    id.index_sha.fill(1U); id.sidecar_sha.fill(2U);
    id.adjacency_sha.fill(3U);
    const std::string path = "v0_residual_test.v0res";
    std::remove(path.c_str());
    std::remove((path + ".partial").c_str());
    std::vector<float> full_matrix(8U * 4U, 0.0f);
    {
        hnswlib::V0ResidualCompanionWriter writer(
            path, id, full_matrix, 4U, 8U, 42U, 2U);
        writer.append(std::vector<uint8_t>(1U, 0x55U), 0.5f, -1.0f, true);
        writer.append(std::vector<uint8_t>(1U, 0U), 0.0f, 0.0f, false);
        writer.finish();
    }
    {
        hnswlib::V0ResidualCompanion companion(path, id, 4U, 2U);
        assert(companion.bits() == 8U && companion.seed() == 42U);
        assert(companion.valid(companion.record(0U)));
        assert(!companion.valid(companion.record(1U)));
        assert(std::fabs(companion.offset(companion.record(0U)) + 1.0f) < 1e-6);
    }
    const std::string resumed_path = "v0_residual_test_resumed.v0res";
    std::remove(resumed_path.c_str());
    std::remove((resumed_path + ".partial").c_str());
    std::remove((resumed_path + ".checkpoint").c_str());
    {
        hnswlib::V0ResidualCompanionWriter writer(
            resumed_path, id, full_matrix, 4U, 8U, 42U, 2U);
        writer.append(std::vector<uint8_t>(1U, 0x55U), 0.5f, -1.0f, true);
        writer.checkpoint();
    }
    {
        hnswlib::V0ResidualCompanionWriter writer(
            resumed_path, id, full_matrix, 4U, 8U, 42U, 2U, true);
        assert(writer.written() == 1U);
        writer.append(std::vector<uint8_t>(1U, 0U), 0.0f, 0.0f, false);
        writer.checkpoint();
        writer.finish();
    }
    {
        hnswlib::V0ResidualCompanion resumed(resumed_path, id, 4U, 2U);
        assert(resumed.valid(resumed.record(0U)) &&
               !resumed.valid(resumed.record(1U)));
    }
    std::remove(resumed_path.c_str());
    id.index_sha[0] = 0U;
    bool rejected = false;
    try { hnswlib::V0ResidualCompanion bad(path, id, 4U, 2U); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    id.index_sha[0] = 1U;
    auto rejects_bytes = [&](size_t position, uint8_t replacement) {
        std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
        std::vector<char> bytes(static_cast<size_t>(input.tellg()));
        input.seekg(0);
        input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        assert(input && position < bytes.size());
        bytes[position] = static_cast<char>(replacement);
        const std::string changed = "v0_residual_test_corrupt.v0res";
        std::ofstream output(changed.c_str(), std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        output.close();
        bool failed = false;
        try { hnswlib::V0ResidualCompanion bad(changed, id, 4U, 2U); }
        catch (const std::runtime_error&) { failed = true; }
        std::remove(changed.c_str());
        assert(failed);
    };
    rejects_bytes(20U, 16U);   // wrong bit length / stride
    rejects_bytes(32U, 3U);    // wrong record count
    rejects_bytes(200U, 0U);   // incomplete marker
    rejects_bytes(208U, 1U);   // matrix digest
    rejects_bytes(208U + 8U * 4U * 4U, 0U);  // payload digest
    {
        std::ifstream input(path.c_str(), std::ios::binary);
        std::vector<char> prefix(100U);
        input.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        const std::string truncated = "v0_residual_test_truncated.v0res";
        std::ofstream output(truncated.c_str(), std::ios::binary);
        output.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        output.close();
        bool failed = false;
        try { hnswlib::V0ResidualCompanion bad(truncated, id, 4U, 2U); }
        catch (const std::runtime_error&) { failed = true; }
        std::remove(truncated.c_str());
        assert(failed);
    }
    std::remove(path.c_str());
    return 0;
}
