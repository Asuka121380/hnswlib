#include <stdexcept>
#include <string>

#include "tools/edge_estimation/backend_registry.h"
#include "tools/edge_estimation/backends/prq.h"
#include "tools/edge_estimation/backends/rabitq_adapter.h"
#include "tools/edge_estimation/backends/rotated_pq.h"
#include "tools/edge_estimation/backends/saq_adapter.h"

int main() {
    const char* unavailable[] = {"opq", "prq", "jq", "rabitq", "saq"};
    for (const char* name : unavailable) {
        const uq::BackendCapability capability = uq::resolveBackend(name);
        if (capability.compiled || !capability.reason || !*capability.reason)
            throw std::runtime_error(std::string("ambiguous capability: ") + name);
    }
    const uq::UnavailableBackend adapters[] = {
        uq::opqBackend(), uq::prqBackend(), uq::jqBackend(),
        uq::rabitqBackend(), uq::saqBackend()};
    for (const uq::UnavailableBackend& adapter : adapters) {
        if (adapter.available() || !adapter.reason() || !*adapter.reason())
            throw std::runtime_error("optional adapter did not report unavailable");
        bool threw = false;
        try { adapter.load(); } catch (const std::runtime_error&) { threw = true; }
        if (!threw) throw std::runtime_error("unavailable adapter load did not fail");
    }
    return 0;
}
