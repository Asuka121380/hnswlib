#pragma once

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "backends/pq_packed_artifact.h"
#include "backends/prq.h"
#include "backends/rabitq_adapter.h"
#include "backends/rotated_pq.h"
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
#include "backends/pq_legacy.h"
#endif
#if defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) && defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
#include "backends/pq_qjl_legacy.h"
#endif
#include "event_format.h"
#include "query_store.h"

namespace uq {

enum class ArtifactBackendKind { PqPacked, RotatedPq, Prq, RaBitQ,
                                 PqLegacy, PqQjlLegacy };

struct ArtifactDescriptor {
    ArtifactBackendKind kind;
    std::string backend_name;
    std::string format;
};

inline ArtifactDescriptor inspectArtifact(const std::filesystem::path& root) {
    const std::map<std::string, std::string> config = readNativeConfig(root / "native.cfg");
    const auto format = config.find("format");
    if (format == config.end()) throw std::runtime_error("artifact format is missing");
    if (format->second == "uq-pq-packed/1")
        return ArtifactDescriptor{ArtifactBackendKind::PqPacked, "pq_packed", format->second};
    if (format->second == "uq-rotated-pq/1") {
        const auto backend = config.find("backend");
        if (backend == config.end() ||
            (backend->second != "opq" && backend->second != "jq"))
            throw std::runtime_error("unsupported rotated-PQ backend identity");
        return ArtifactDescriptor{ArtifactBackendKind::RotatedPq, backend->second, format->second};
    }
    if (format->second == "uq-prq/1")
        return ArtifactDescriptor{ArtifactBackendKind::Prq, "prq", format->second};
    if (format->second == "uq-rabitq/1")
        return ArtifactDescriptor{ArtifactBackendKind::RaBitQ, "rabitq", format->second};
    if (format->second == "uq-pq-legacy/1") {
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
        return ArtifactDescriptor{ArtifactBackendKind::PqLegacy, "pq_legacy", format->second};
#else
        throw std::runtime_error("legacy PQ artifact support was not compiled");
#endif
    }
    if (format->second == "uq-pq-qjl-legacy/1") {
#if defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) && defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
        return ArtifactDescriptor{ArtifactBackendKind::PqQjlLegacy,
                                  "pq_qjl_legacy", format->second};
#else
        throw std::runtime_error("legacy PQ+QJL artifact support was not compiled");
#endif
    }
    throw std::runtime_error("unsupported artifact format: " + format->second);
}

template <typename Callback>
decltype(auto) withArtifactKernel(const std::filesystem::path& root,
                                  std::shared_ptr<const QueryStore> queries,
                                  const Header& event_header,
                                  Callback&& callback) {
    const ArtifactDescriptor descriptor = inspectArtifact(root);
    switch (descriptor.kind) {
        case ArtifactBackendKind::PqPacked: {
            PackedPqArtifactKernel kernel(root, std::move(queries), event_header);
            return std::forward<Callback>(callback)(kernel, descriptor);
        }
        case ArtifactBackendKind::RotatedPq: {
            RotatedPqArtifactKernel kernel(root, std::move(queries), event_header);
            return std::forward<Callback>(callback)(kernel, descriptor);
        }
        case ArtifactBackendKind::Prq: {
            PrqArtifactKernel kernel(root, std::move(queries), event_header);
            return std::forward<Callback>(callback)(kernel, descriptor);
        }
        case ArtifactBackendKind::RaBitQ: {
            RaBitQArtifactKernel kernel(root, std::move(queries), event_header);
            return std::forward<Callback>(callback)(kernel, descriptor);
        }
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
        case ArtifactBackendKind::PqLegacy: {
            PqLegacyArtifactKernel kernel(root, std::move(queries), event_header);
            return std::forward<Callback>(callback)(kernel, descriptor);
        }
#endif
#if defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) && defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
        case ArtifactBackendKind::PqQjlLegacy: {
            PqQjlLegacyArtifactKernel kernel(root, std::move(queries), event_header);
            return std::forward<Callback>(callback)(kernel, descriptor);
        }
#endif
#ifndef HNSWLIB_ENABLE_EDGE_QUANT_V0
        case ArtifactBackendKind::PqLegacy: break;
#endif
#if !defined(HNSWLIB_ENABLE_EDGE_QUANT_V0) || !defined(HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR)
        case ArtifactBackendKind::PqQjlLegacy: break;
#endif
    }
    throw std::runtime_error("artifact backend dispatch failed");
}

}  // namespace uq
