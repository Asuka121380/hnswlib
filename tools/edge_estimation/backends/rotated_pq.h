#pragma once

#include "optional_backend.h"

namespace uq {
inline UnavailableBackend opqBackend() {
    return UnavailableBackend("opq", "faiss adapter is not compiled");
}
inline UnavailableBackend jqBackend() {
    return UnavailableBackend("jq", "JQ source/artifact adapter has not been imported");
}
}  // namespace uq
