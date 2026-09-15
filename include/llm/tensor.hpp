#include <vector>
#include <cstdint>
#include <numeric>
#include <functional>
#include <cstdlib>
#include <llm/dtype.hpp>
#include <llm/common.hpp>

namespace llm
{

    using Shape = std::vector<int64_t>;
    using Strides = std::vector<int64_t>;

    inline int64_t numel(const Shape &shape)
    {
        return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    }

    inline Strides contiguous_strides(const Shape &shape)
    {
        Strides strides(shape.size());
        std::exclusive_scan(shape.rbegin(), shape.rend(), strides.rbegin(), int64_t{1}, std::multiplies<int64_t>());
        return strides;
    }

    class Storage
    {
    private:
        void *buffer_ptr;

        void aligned_free();

    public:
        Storage(size_t alignment, size_t size);
        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;
        ~Storage();

        static int &alloc_count();
        void *data();
    };

    class Tensor
    {
    private:
        std::shared_ptr<Storage> storage;
        int offset_;
        Shape shape_;
        Strides strides_;
        DType dtype_;

    public:
        Tensor() = default;
        Tensor(const Tensor &other) = default;
        Tensor &operator=(const Tensor &) = default;
        Tensor(Tensor &&) = default;
        Tensor &operator=(Tensor &&) = default;
        ~Tensor() = default;

        static Tensor empty(const Shape &shape, const DType &dtype, int alignment = 64);

        Shape shape() const;
        Strides strides() const;
        DType dtype() const;
        int64_t ndim() const;
        int64_t numel() const;
        int64_t nbytes() const;
        bool is_contiguous() const;

        template <typename T>
        T *data_ptr() { return static_cast<T *>(storage->data()) + offset_; }

        template <typename T>
        const T *data_ptr() const { return static_cast<const T *>(storage->data()) + offset_; }

        template <typename T>
        T &at(std::initializer_list<int64_t> indices)
        {
            LLM_ASSERT(indices.size() == strides_.size(), "Unequal dimensions");
            T *ptr = data_ptr<T>();
            int64_t flat_location = std::inner_product(indices.begin(), indices.end(), strides_.begin(), int64_t{0});
            return ptr[flat_location];
        }

        template <typename T>
        const T &at(std::initializer_list<int64_t> indices) const
        {
            LLM_ASSERT(indices.size() == strides_.size(), "Unequal dimensions");
            const T *ptr = data_ptr<const T>();
            int64_t flat_location = std::inner_product(indices.begin(), indices.end(), strides_.begin(), int64_t{0});
            return ptr[flat_location];
        }
    };

}