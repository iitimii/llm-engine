#include <llm/tensor.hpp>

namespace llm
{

    void Storage::aligned_free()
    {
        std::free(buffer_ptr);
    }

    Storage::Storage(size_t alignment, size_t size)
    {
        // std::vector<float> data(length);
        // auto data = std::make_shared<std::vector<float>>(length);

        buffer_ptr = std::aligned_alloc(alignment, size);

        ++alloc_count();
    }

    Storage::~Storage()
    {
        aligned_free();
    }

    int &Storage::alloc_count()
    {
        static int object_count;
        return object_count;
    }

    const void *Storage::data() { return buffer_ptr; }

    Tensor::Tensor(const Shape &shape, const DType &dtype, int alignment = 64) : shape_(shape), dtype_(dtype)
    {
        strides_ = contiguous_strides(shape);
    }

    Shape Tensor::shape() { return shape_; }

    Strides Tensor::strides() { return strides_; }

    DType Tensor::dtype() { return dtype_; }

    int64_t Tensor::ndim() { return shape_.size(); }

    int64_t Tensor::numel() { return ::llm::numel(shape_); }

    int64_t Tensor::nbytes() { return numel() * dtype_size(dtype_); }

    bool Tensor::is_contiguous() { return strides_ == contiguous_strides(shape_); }

}