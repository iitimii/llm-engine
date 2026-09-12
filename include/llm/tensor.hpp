#include <vector>
#include <cstdint>
#include <numeric>
#include <functional>
#include <cstdlib>
#include <llm/dtype.hpp>

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
        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;

        void *buffer_ptr;

        void aligned_free();

    public:
        Storage(size_t alignment, size_t size);

        ~Storage();

        static int &alloc_count();
        const void* data();
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
        Tensor(const Shape &shape, const DType &dtype, int alignment = 64);

        static Tensor empty(const Shape &shape, const DType &dtype, int alignment = 64);

        Shape shape();
        Strides strides();
        DType dtype();
        int64_t ndim();
        int64_t numel();
        int64_t nbytes();
        bool is_contiguous();

        template <typename T>
        T *data_ptr() { return static_cast<T *>(storage->data()) + offset_; }

        template <typename T>
        T *at(std::initializer_list<int64_t> coordinates)
        {
            return data_ptr<T>() + std::inner_product(coordinates.begin(), coordinates.end(), strides.begin(), int64_t{0});
        }
    };

}