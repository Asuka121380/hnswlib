#pragma once

#include <stdexcept>
#include <string>

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
    bool compiled;
    bool supports_scalar;
    bool supports_source_batch;
    const char* reason;
};

inline BackendCapability resolveBackend(const std::string& name) {
    if (name == "pq_packed")
        return BackendCapability{BackendKind::PqPacked, true, true, false, ""};
    if (name == "pq_legacy") {
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
        return BackendCapability{BackendKind::PqLegacy, true, true, false, ""};
#else
        return BackendCapability{BackendKind::PqLegacy, false, false, false, "not compiled"};
#endif
    }
    if (name == "pq_qjl_legacy") {
#ifdef HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR
        return BackendCapability{BackendKind::PqQjlLegacy, true, true, false, ""};
#else
        return BackendCapability{BackendKind::PqQjlLegacy, false, false, false, "not compiled"};
#endif
    }
    if (name == "opq") return BackendCapability{BackendKind::Opq, false, false, false, "faiss adapter is not compiled"};
    if (name == "prq") return BackendCapability{BackendKind::Prq, false, false, false, "faiss adapter is not compiled"};
    if (name == "jq") return BackendCapability{BackendKind::Jq, false, false, false, "JQ source/artifact adapter has not been imported"};
    if (name == "rabitq") return BackendCapability{BackendKind::RaBitQ, false, false, false, "optional RaBitQ dependency is not compiled"};
    if (name == "saq") return BackendCapability{BackendKind::Saq, false, false, false, "optional SAQ dependency/required ISA is unavailable"};
    throw std::invalid_argument("unknown backend: " + name);
}

}  // namespace uq
