#pragma once
#include <vector>
#include <cstdint>
#include <numeric>
#include <functional>
#include <cstdlib>
#include <memory>
#include <llm/dtype.hpp>
#include <llm/common.hpp>
#include <llm/memory.hpp>

namespace llm
{

    using Shape = std::vector<int64_t>;
    using Strides = std::vector<int64_t>;

    int64_t numel(const Shape &shape);
    Strides contiguous_strides(const Shape &shape);

    class Storage
    {
    private:
        std::byte *data_ = nullptr;
        size_t nbytes_ = 0;
        inline static uint64_t s_alloc_count = 0;

    public:
        explicit Storage(size_t nbytes, size_t alignment = 64);
        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;
        ~Storage();

        std::byte *data() noexcept { return data_; };
        const std::byte *data() const noexcept { return data_; };
        size_t nbytes() const noexcept { return nbytes_; }
        static uint64_t alloc_count() noexcept { return s_alloc_count; };
    };

    class Tensor
    {
    private:
        int64_t rel_offset(std::initializer_list<int64_t> idx) const;

        std::shared_ptr<Storage> storage_;
        int64_t offset_{0};
        Shape shape_;
        Strides strides_;
        DType dtype_ = DType::F32;

    public:
        Tensor() = default;
        Tensor(const Tensor &other) = default;
        Tensor &operator=(const Tensor &) = default;
        Tensor(Tensor &&) = default;
        Tensor &operator=(Tensor &&) = default;
        ~Tensor() = default;

        static Tensor empty(Shape shape, DType dtype, size_t alignment = 64);

        // layout queries
        const Shape &shape() const noexcept { return shape_; }
        const Strides &strides() const noexcept { return strides_; }
        DType dtype() const noexcept { return dtype_; }
        size_t ndim() const noexcept { return shape_.size(); }
        int64_t numel() const noexcept { return llm::numel(shape_); }
        size_t nbytes() const noexcept { return static_cast<size_t>(numel()) * dtype_size(dtype_); }
        bool is_contiguous() const noexcept;
        bool defined() const noexcept { return storage_ != nullptr; }

        template <typename T>
        T *data_ptr() { return reinterpret_cast<T *>(storage_->data()) + offset_; }

        template <typename T>
        const T *data_ptr() const { return reinterpret_cast<const T *>(storage_->data()) + offset_; }

        template <typename T>
        T &at(std::initializer_list<int64_t> idx)
        {
            return data_ptr<T>()[rel_offset(idx)];
        }

        template <typename T>
        const T &at(std::initializer_list<int64_t> idx) const
        {
            return data_ptr<T>()[rel_offset(idx)];
        }

        Tensor reshape(Shape new_shape) const;
        Tensor view(Shape new_shape) const {return reshape(std::move(new_shape));};
        Tensor permute(const std::vector<int> &dims) const;
        Tensor transpose(int dim0, int dim1) const;
        Tensor slice(int dim, int64_t start, int64_t stop) const;

        Tensor contiguous() const;

        const void *storage_id() const noexcept { return storage_.get(); }
        int64_t offset() const noexcept { return offset_; }
    };

}