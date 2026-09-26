#include <stdexcept>
#include <string>

#include "tools/edge_estimation/backend_registry.h"
#include "tools/edge_estimation/backends/prq.h"
#include "tools/edge_estimation/backends/rabitq_adapter.h"
#include "tools/edge_estimation/backends/rotated_pq.h"
#include "tools/edge_estimation/backends/saq_adapter.h"

int main() {
    const uq::BackendCapability opq = uq::resolveBackend("opq");
    if (!opq.compiled || !opq.native_available || !opq.artifact_supported ||
        opq.formal_validation_passed)
        throw std::runtime_error("OPQ runtime capability is inconsistent");
    const uq::BackendCapability prq = uq::resolveBackend("prq");
    if (!prq.compiled || !prq.native_available || !prq.artifact_supported)
        throw std::runtime_error("PRQ runtime capability is inconsistent");
    const uq::BackendCapability jq = uq::resolveBackend("jq");
    if (!jq.compiled || !jq.native_available || !jq.artifact_supported)
        throw std::runtime_error("JQ runtime capability is inconsistent");
    const uq::BackendCapability rabitq = uq::resolveBackend("rabitq");
    if (!rabitq.compiled || !rabitq.native_available || !rabitq.artifact_supported)
        throw std::runtime_error("RaBitQ runtime capability is inconsistent");
    const char* unavailable[] = {"saq"};
    for (const char* name : unavailable) {
        const uq::BackendCapability capability = uq::resolveBackend(name);
        if (capability.compiled || !capability.reason || !*capability.reason)
            throw std::runtime_error(std::string("ambiguous capability: ") + name);
    }
    const uq::UnavailableBackend adapters[] = {
        uq::saqBackend()};
    for (const uq::UnavailableBackend& adapter : adapters) {
        if (adapter.available() || !adapter.reason() || !*adapter.reason())
            throw std::runtime_error("optional adapter did not report unavailable");
        bool threw = false;
        try { adapter.load(); } catch (const std::runtime_error&) { threw = true; }
        if (!threw) throw std::runtime_error("unavailable adapter load did not fail");
    }
    return 0;
}
