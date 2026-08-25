#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hnswlib/visited_list_pool.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void testQueryIsolationAndWrap() {
    hnswlib::VisitedList list(16);
    list.reset();
    const hnswlib::vl_type first_generation = list.curV;
    list.mass[3] = first_generation;
    list.approx_pruned_mass[3] = first_generation;
    list.reset();
    require(list.curV != first_generation, "generation did not advance");
    require(list.mass[3] != list.curV, "visited state leaked across queries");
    require(list.approx_pruned_mass[3] != list.curV,
            "approx-pruned state leaked across queries");

    list.mass[5] = list.curV;
    list.approx_pruned_mass[5] = list.curV;
    const size_t resets_to_wrap =
        static_cast<size_t>(static_cast<hnswlib::vl_type>(-list.curV));
    for (size_t i = 0U; i < resets_to_wrap; ++i) list.reset();
    require(list.curV == 1U, "generation wrap did not restart at one");
    for (size_t i = 0U; i < list.numelements; ++i) {
        require(list.mass[i] == 0U, "visited mass was not cleared on wrap");
        require(list.approx_pruned_mass[i] == 0U,
                "approx-pruned mass was not cleared on wrap");
    }
}

void testConcurrentOwnership() {
    const size_t thread_count = 8U;
    hnswlib::VisitedListPool pool(2, 64);
    std::atomic<size_t> ready(0U);
    std::atomic<bool> release(false);
    std::vector<hnswlib::VisitedList*> acquired(thread_count, NULL);
    std::vector<std::thread> threads;
    for (size_t i = 0U; i < thread_count; ++i) {
        threads.push_back(std::thread([&pool, &ready, &release, &acquired, i]() {
            hnswlib::VisitedList* list = pool.getFreeVisitedList();
            acquired[i] = list;
            list->mass[i] = list->curV;
            list->approx_pruned_mass[i] = list->curV;
            ready.fetch_add(1U, std::memory_order_release);
            while (!release.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            require(list->mass[i] == list->curV,
                    "concurrent visited ownership was corrupted");
            require(list->approx_pruned_mass[i] == list->curV,
                    "concurrent retry ownership was corrupted");
            pool.releaseVisitedList(list);
        }));
    }
    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    for (size_t i = 0U; i < thread_count; ++i) {
        for (size_t j = i + 1U; j < thread_count; ++j) {
            require(acquired[i] != acquired[j],
                    "pool loaned one state buffer to concurrent queries");
        }
    }
    release.store(true, std::memory_order_release);
    for (size_t i = 0U; i < threads.size(); ++i) threads[i].join();
}

}  // namespace

int main() {
    testQueryIsolationAndWrap();
    testConcurrentOwnership();
    return 0;
}
