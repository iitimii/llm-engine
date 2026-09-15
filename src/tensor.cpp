#include <llm/tensor.hpp>

namespace llm
{

    void Storage::aligned_free()
    {
        std::free(buffer_ptr);
    }

    Storage::Storage(size_t alignment, size_t size)
    {
        buffer_ptr = std::aligned_alloc(alignment, size);
        ++alloc_count();
    }

    Storage::~Storage()
    {
        aligned_free();
    }

    int &Storage::alloc_count()
    {
        static int object_count = 0;
        return object_count;
    }

    void *Storage::data() { return buffer_ptr; }

    Tensor Tensor::empty(const Shape &shape, const DType &dtype, int alignment)
    {
        Tensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        t.strides_ = contiguous_strides(t.shape_);
        t.offset_ = 0;
        t.storage = std::make_shared<Storage>(alignment, t.nbytes());
        return t;
    }

    Shape Tensor::shape() const { return shape_; }
    Strides Tensor::strides() const { return strides_; }
    DType Tensor::dtype() const { return dtype_; }
    int64_t Tensor::ndim() const { return shape_.size(); }
    int64_t Tensor::numel() const { return ::llm::numel(shape_); }
    int64_t Tensor::nbytes() const { return numel() * dtype_size(dtype_); }
    bool Tensor::is_contiguous() const { return strides_ == contiguous_strides(shape_); }
}