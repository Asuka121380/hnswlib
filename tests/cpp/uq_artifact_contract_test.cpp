#include <iostream>
#include <stdexcept>

#include "tools/edge_estimation/artifact_manifest.h"
#include "tools/edge_estimation/backend_registry.h"

int main() {
    uq::ArtifactManifest manifest{};
    manifest.schema_version = 1U;
    manifest.format_family = "uq-pq-packed";
    manifest.format_version = 1U;
    manifest.representation = "direct_unit_edge";
    manifest.codec = "pq_packed";
    manifest.correction = "none";
    manifest.numeric_profile = "float_lut_v1";
    manifest.dimension = 8U;
    manifest.coverage = uq::ArtifactCoverage::FullGraph;
    uq::ArtifactFile file{}; file.relative_path = "records/codes.bin";
    manifest.files.push_back(file);
    manifest.validate(true);
    manifest.files[0].relative_path = "../escape.bin";
    bool rejected = false;
    try {
        manifest.validate(true);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("artifact path traversal was accepted");
    if (!uq::resolveBackend("pq_packed").compiled ||
        !uq::resolveBackend("opq").compiled ||
        !uq::resolveBackend("prq").compiled ||
        !uq::resolveBackend("jq").compiled ||
        !uq::resolveBackend("rabitq").compiled ||
        uq::resolveBackend("saq").compiled)
        throw std::runtime_error("backend registry mismatch");
    std::cout << "uq_artifact_contract_test_ok" << std::endl;
    return 0;
}
