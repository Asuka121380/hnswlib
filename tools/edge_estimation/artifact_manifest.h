#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace uq {

enum class ArtifactCoverage : uint8_t { Subset = 0, FullGraph = 1 };

struct ArtifactFile {
    std::string relative_path;
    uint64_t size;
    std::array<uint8_t, 32> sha256;
};

struct ArtifactManifest {
    uint32_t schema_version;
    std::string format_family;
    uint32_t format_version;
    std::string representation;
    std::string codec;
    std::string correction;
    std::string numeric_profile;
    uint32_t dimension;
    std::array<uint8_t, 32> edge_catalog_identity;
    ArtifactCoverage coverage;
    std::vector<ArtifactFile> files;

    void validate(bool require_full_graph) const {
        if (schema_version != 1U || format_family.empty() ||
            format_version == 0U || representation.empty() || codec.empty() ||
            correction.empty() || numeric_profile.empty() || dimension == 0U)
            throw std::runtime_error("invalid artifact manifest fields");
        if (require_full_graph && coverage != ArtifactCoverage::FullGraph)
            throw std::runtime_error("formal timing requires a full-graph artifact");
        for (size_t i = 0; i < files.size(); ++i) {
            const std::string& path = files[i].relative_path;
            if (path.empty() || path[0] == '/' || path[0] == '\\' ||
                path.find(':') != std::string::npos ||
                path == ".." || path.find("../") != std::string::npos ||
                path.find("..\\") != std::string::npos)
                throw std::runtime_error("artifact file path escapes artifact root");
        }
    }
};

}  // namespace uq
