#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "llm/common.hpp"

namespace llm
{
        inline bool is_power_of_two(size_t x) { return x != 0 && (x & (x - 1)) == 0; }

        inline void *aligned_malloc(size_t nbytes, size_t alignment = 64)
        {
                LLM_ASSERT(is_power_of_two(alignment), "alignment must be  a power of two");
#if defined(_WIN32)
                return _aligned_malloc(nbytes ? nbytes : 1, alignment);
#else
                const size_t rounded = (nbytes + alignment - 1) & ~(alignment - 1);
                return std::aligned_alloc(alignment, rounded ? rounded : alignment);
#endif
        }

        inline void aligned_free(void *p) noexcept
        {
#if defined(_WIN32)
                _aligned_free(p);
#else
                std::free(p);
#endif
        }

        class Arena
        {
        public:
                Arena(size_t capacity, size_t alignment = 64) : alignment_(alignment), capacity_(capacity)
                {
                        base_ = static_cast<std::byte *>(aligned_malloc(capacity, alignment));
                        LLM_CHECK(base_ != nullptr, "Storage: allocation failed");
                }
                Arena(const Arena &) = delete;
                Arena &operator=(const Arena &) = delete;
                ~Arena() { aligned_free(base_); }

                void *allocate(size_t nbytes)
                {
                        const size_t aligned_off = (offset_ + alignment_ - 1) & ~(alignment_ - 1);
                        LLM_CHECK(aligned_off + nbytes <= capacity_,
                                  "Arena: out of memory");
                        void *p = base_ + aligned_off;
                        offset_ = aligned_off + nbytes;
                        if (offset_ > high_water_)
                                high_water_ = offset_;
                        return p;
                }

                void reset() { offset_ = 0; }
                size_t used() { return offset_; }
                size_t high_water() { return high_water_; }
                size_t capacity() { return capacity_; }

        private:
                size_t offset_ = 0;
                std::byte *base_ = nullptr;
                size_t alignment_ = 0;
                size_t capacity_ = 0;
                size_t high_water_ = 0;
        };
}