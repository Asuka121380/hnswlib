#pragma once

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct P0CounterEntry {
    const char* name;
    uint32_t type;
    uint64_t config;
    int fd;
    bool available;
    int error_number;
    uint64_t value;
    double running_ratio;
};

// Diagnostic-only counters around the complete timed query loop. Unsupported
// events are reported individually instead of aborting the QPS process.
class P0Counters {
 public:
    static const size_t kCount = 7U;

    explicit P0Counters(bool requested) : requested_(requested) {
        entries_[0] = makeEntry("cycles", hardwareType(), cyclesConfig());
        entries_[1] = makeEntry("instructions", hardwareType(), instructionsConfig());
        entries_[2] = makeEntry("branches", hardwareType(), branchesConfig());
        entries_[3] = makeEntry("branch_misses", hardwareType(), branchMissesConfig());
        entries_[4] = makeEntry("cache_references", hardwareType(), cacheReferencesConfig());
        entries_[5] = makeEntry("cache_misses", hardwareType(), cacheMissesConfig());
        entries_[6] = makeEntry("l1d_read_misses", cacheType(), l1dReadMissConfig());
        if (!requested_) return;
#ifdef __linux__
        for (size_t i = 0U; i < kCount; ++i) {
            perf_event_attr attributes;
            std::memset(&attributes, 0, sizeof(attributes));
            attributes.size = sizeof(attributes);
            attributes.disabled = 1;
            attributes.exclude_kernel = 1;
            attributes.exclude_hv = 1;
            attributes.type = entries_[i].type;
            attributes.config = entries_[i].config;
            attributes.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                PERF_FORMAT_TOTAL_TIME_RUNNING;
            entries_[i].fd = static_cast<int>(syscall(
                SYS_perf_event_open, &attributes, 0, -1, -1, 0));
            if (entries_[i].fd < 0) {
                entries_[i].error_number = errno;
            } else {
                entries_[i].available = true;
            }
        }
#else
        for (size_t i = 0U; i < kCount; ++i) {
            entries_[i].error_number = ENOTSUP;
        }
#endif
    }

    P0Counters(const P0Counters&) = delete;
    P0Counters& operator=(const P0Counters&) = delete;

    ~P0Counters() {
#ifdef __linux__
        for (size_t i = 0U; i < kCount; ++i) {
            if (entries_[i].fd >= 0) ::close(entries_[i].fd);
        }
#endif
    }

    void resetAndStart() {
#ifdef __linux__
        for (size_t i = 0U; i < kCount; ++i) {
            if (!entries_[i].available) continue;
            if (ioctl(entries_[i].fd, PERF_EVENT_IOC_RESET, 0) < 0 ||
                ioctl(entries_[i].fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
                disable(i, errno);
            }
        }
#endif
    }

    void stopAndCollect() {
#ifdef __linux__
        for (size_t i = 0U; i < kCount; ++i) {
            if (!entries_[i].available) continue;
            if (ioctl(entries_[i].fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
                disable(i, errno);
                continue;
            }
            uint64_t values[3] = {0U, 0U, 0U};
            if (::read(entries_[i].fd, values, sizeof(values)) !=
                    static_cast<ssize_t>(sizeof(values)) ||
                values[1] == 0U || values[2] == 0U) {
                disable(i, errno == 0 ? EIO : errno);
                continue;
            }
            entries_[i].running_ratio = static_cast<double>(values[2]) /
                static_cast<double>(values[1]);
            const long double scaled = static_cast<long double>(values[0]) *
                static_cast<long double>(values[1]) /
                static_cast<long double>(values[2]);
            entries_[i].value = static_cast<uint64_t>(std::llround(scaled));
        }
#endif
    }

    bool requested() const { return requested_; }
    const P0CounterEntry& entry(size_t index) const { return entries_[index]; }

 private:
    bool requested_;
    P0CounterEntry entries_[kCount];

    static P0CounterEntry makeEntry(
        const char* name, uint32_t type, uint64_t config) {
        P0CounterEntry entry = {
            name, type, config, -1, false, 0, 0U, 0.0
        };
        return entry;
    }

    void disable(size_t index, int error_number) {
        entries_[index].available = false;
        entries_[index].error_number = error_number;
    }

#ifdef __linux__
    static uint32_t hardwareType() { return PERF_TYPE_HARDWARE; }
    static uint32_t cacheType() { return PERF_TYPE_HW_CACHE; }
    static uint64_t cyclesConfig() { return PERF_COUNT_HW_CPU_CYCLES; }
    static uint64_t instructionsConfig() { return PERF_COUNT_HW_INSTRUCTIONS; }
    static uint64_t branchesConfig() { return PERF_COUNT_HW_BRANCH_INSTRUCTIONS; }
    static uint64_t branchMissesConfig() { return PERF_COUNT_HW_BRANCH_MISSES; }
    static uint64_t cacheReferencesConfig() { return PERF_COUNT_HW_CACHE_REFERENCES; }
    static uint64_t cacheMissesConfig() { return PERF_COUNT_HW_CACHE_MISSES; }
    static uint64_t l1dReadMissConfig() {
        return PERF_COUNT_HW_CACHE_L1D |
            (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_OP_READ) << 8U) |
            (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_RESULT_MISS) << 16U);
    }
#else
    static uint32_t hardwareType() { return 0U; }
    static uint32_t cacheType() { return 0U; }
    static uint64_t cyclesConfig() { return 0U; }
    static uint64_t instructionsConfig() { return 0U; }
    static uint64_t branchesConfig() { return 0U; }
    static uint64_t branchMissesConfig() { return 0U; }
    static uint64_t cacheReferencesConfig() { return 0U; }
    static uint64_t cacheMissesConfig() { return 0U; }
    static uint64_t l1dReadMissConfig() { return 0U; }
#endif
};
