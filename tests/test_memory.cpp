#include "test_framework.hpp"
#include "llm/memory.hpp"

using namespace llm;

TEST(aligned_alloc_returns_aligned_pointer)
{
    void *p = aligned_malloc(1000, 64);
    REQUIRE(p != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);
    aligned_free(p);
}

TEST(arena_aligned_and_nonoverlapping)
{
    Arena a(1u << 20, 64); // 1 MiB
    void *x = a.allocate(100);
    void *y = a.allocate(100);
    CHECK(reinterpret_cast<uintptr_t>(x) % 64 == 0);
    CHECK(reinterpret_cast<uintptr_t>(y) % 64 == 0);
    CHECK(x != y);
    // 100 bytes rounds up to 128 for the next aligned slot, so y is >= 64 past x
    CHECK(reinterpret_cast<char *>(y) - reinterpret_cast<char *>(x) >= 64);
}

TEST(arena_reset_reuses_memory_and_tracks_peak)
{
    Arena a(1u << 20, 64);
    void *first = a.allocate(256);
    CHECK(a.used() == 256);
    a.reset();
    CHECK(a.used() == 0);
    void *again = a.allocate(256);
    CHECK(first == again);
    CHECK(a.high_water() >= 256); // same address after reset
    // peak remembered across the reset
}