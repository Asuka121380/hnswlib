#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace uq {

enum class BackendKind {
    PqPacked,
    PqLegacy,
    PqQjlLegacy,
    Opq,
    Prq,
    Jq,
    RaBitQ,
    Saq
};

struct BackendCapability {
    BackendKind kind;
    const char* name;
    bool compiled;
    bool supports_scalar;
    bool supports_source_batch;
    bool native_available;
    bool artifact_supported;
    bool runtime_isa_supported;
    bool formal_validation_passed;
    const char* reason;
};

inline BackendCapability resolveBackend(const std::string& name) {
    if (name == "pq_packed")
        return BackendCapability{BackendKind::PqPacked, "pq_packed", true, true, false,
                                 true, true, true, false, "formal validation is artifact-specific"};
    if (name == "pq_legacy") {
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
        return BackendCapability{BackendKind::PqLegacy, "pq_legacy", true, true, false,
                                 true, true, true, false, "formal parity requires the frozen legacy sidecar"};
#else
        return BackendCapability{BackendKind::PqLegacy, "pq_legacy", false, false, false,
                                 false, false, true, false, "not compiled"};
#endif
    }
    if (name == "pq_qjl_legacy") {
#if defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) && defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
        return BackendCapability{BackendKind::PqQjlLegacy, "pq_qjl_legacy", true, true, false,
                                 true, true, true, false, "formal parity requires the frozen sidecar and QJL companion"};
#else
        return BackendCapability{BackendKind::PqQjlLegacy, "pq_qjl_legacy", false, false, false,
                                 false, false, true, false, "not compiled"};
#endif
    }
    if (name == "opq") return BackendCapability{BackendKind::Opq, "opq", true, true, false,
        true, true, true, false, "formal validation is artifact-specific"};
    if (name == "prq") return BackendCapability{BackendKind::Prq, "prq", true, true, false,
        true, true, true, false, "formal validation is artifact-specific"};
    if (name == "jq") return BackendCapability{BackendKind::Jq, "jq", true, true, false,
        true, true, true, false, "behavioral port; author-binary oracle pending"};
    if (name == "rabitq") return BackendCapability{BackendKind::RaBitQ, "rabitq", true, true, false,
        true, true, true, false, "Faiss 1-bit qb=0 zero-centroid provider"};
    if (name == "saq") return BackendCapability{BackendKind::Saq, "saq", false, false, false,
        false, false, false, false, "optional SAQ dependency/required ISA is unavailable"};
    throw std::invalid_argument("unknown backend: " + name);
}

inline std::vector<BackendCapability> allBackendCapabilities() {
    const char* names[] = {"pq_packed", "pq_legacy", "pq_qjl_legacy", "opq",
                           "prq", "jq", "rabitq", "saq"};
    std::vector<BackendCapability> result;
    for (const char* name : names) result.push_back(resolveBackend(name));
    return result;
}

}  // namespace uq
