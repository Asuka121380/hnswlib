#pragma once

#include <stdexcept>
#include <string>

namespace uq {

class UnavailableBackend {
 public:
    UnavailableBackend(const char* name, const char* reason)
        : name_(name), reason_(reason) {}

    bool available() const { return false; }
    const char* reason() const { return reason_; }
    void load() const {
        throw std::runtime_error(
            std::string("backend unavailable: ") + name_ + ": " + reason_);
    }

 private:
    const char* name_;
    const char* reason_;
};

}  // namespace uq
