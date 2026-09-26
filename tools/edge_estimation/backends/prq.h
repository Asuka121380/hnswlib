#pragma once

#include "optional_backend.h"

namespace uq {
inline UnavailableBackend prqBackend() {
    return UnavailableBackend("prq", "faiss adapter is not compiled");
}
}  // namespace uq
