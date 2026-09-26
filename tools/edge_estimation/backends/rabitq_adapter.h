#pragma once

#include "optional_backend.h"

namespace uq {
inline UnavailableBackend rabitqBackend() {
    return UnavailableBackend("rabitq", "optional RaBitQ dependency is not compiled");
}
}  // namespace uq
