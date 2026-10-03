#include <llm/tensor.hpp>
#include <ranges>

namespace llm
{
    int64_t numel(const Shape &shape)
    {
        int64_t n{1};
        for (int64_t d : shape)
            n *= d;
        return n;
    }

    Strides contiguous_strides(const Shape &shape)
    {
        Strides s(shape.size());
        int64_t acc{1};
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
        {
            s[i] = acc;
            acc *= shape[i];
        }
        return s;
    }

    Storage::Storage(size_t nbytes, size_t alignment) : nbytes_(nbytes)
    {
        data_ = static_cast<std::byte *>(aligned_malloc(nbytes, alignment));
        LLM_CHECK(data_ != nullptr, "Storage: allocation failed");
        ++s_alloc_count;
    }

    Storage::~Storage()
    {
        aligned_free(data_);
    }

    Tensor Tensor::empty(Shape shape, DType dtype, size_t alignment)
    {
        Tensor t;
        t.shape_ = std::move(shape);
        t.dtype_ = dtype;
        t.strides_ = contiguous_strides(t.shape_);
        t.offset_ = 0;
        t.storage_ = std::make_shared<Storage>(t.nbytes(), alignment);
        return t;
    }

    bool Tensor::is_contiguous() const noexcept
    {
        int64_t expected{1};
        for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i)
        {
            if (shape_[i] == 1)
                continue;
            if (strides_[i] != expected)
                return false;
            expected *= shape_[i];
        }
        return true;
    }

    int64_t Tensor::rel_offset(std::initializer_list<int64_t> idx) const
    {
        LLM_ASSERT(idx.size() == ndim(), "at(): wrong number of indices");
        uint64_t d{0};
        uint64_t off{0};
        for (int64_t i : idx)
        {
            LLM_ASSERT(i >= 0 && i < shape_[d], "at(): index out of range");
            off += i * strides_[d];
            ++d;
        }
        return off;
    }

    Tensor Tensor::permute(const std::vector<int> &dims) const
    {
        Tensor out = *this;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            std::swap(out.shape_[i], out.shape_[dims[i]]);
            std::swap(out.strides_[i], out.strides_[dims[i]]);
        }
        return out;
    }

    Tensor Tensor::reshape(Shape new_shape) const
    {
        LLM_ASSERT(numel() == llm::numel(new_shape), "reshape: incompatible shape");
        Tensor out = contiguous();
        out.shape_ = new_shape;
        out.strides_ = contiguous_strides(new_shape);
        return out;
    }

    Tensor Tensor::transpose(int dim0, int dim1) const
    {
        LLM_CHECK(ndim() > dim0 && ndim() > dim1, "dimension greater than existing");
        LLM_CHECK(dim0 >= 0 && dim1 >= 0, "dimension cannot be negative");
        Tensor out = *this;
        std::swap(out.shape_[dim0], out.shape_[dim1]);
        std::swap(out.strides_[dim0], out.strides_[dim1]);
        return out;
    }

    Tensor Tensor::slice(int dim, int64_t start, int64_t stop) const
    {
        LLM_CHECK(dim >= 0 && dim < static_cast<int>(ndim()), "slice: dim out of range");
        LLM_CHECK(0 <= start && start <= stop && stop <= shape_[dim], "slice: range out of bounds");
        Tensor out = *this;
        out.offset_ += start * strides_[dim];
        out.shape_[dim] = stop - start;
        return out;
    }

    Tensor Tensor::contiguous() const
    {
        if (is_contiguous())
            return *this;
        Tensor out = Tensor::empty(shape_, dtype_);
        const size_t es = dtype_size(dtype_);
        const std::byte *src = storage_->data();
        std::byte *dst = out.storage_->data();
        const int64_t n = numel();
        std::vector<int64_t> coord(ndim(), 0);
        for (int64_t linear = 0; linear < n; ++linear)
        {
            int64_t src_elem = offset_;
            for (size_t d = 0; d < ndim(); ++d)
                src_elem += coord[d] * strides_[d];
            std::memcpy(dst + static_cast<size_t>(linear) * es,
                        src + static_cast<size_t>(src_elem) * es, es);        
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d)
            {
                if (++coord[d] < shape_[d])
                    break;
                coord[d] = 0;
            }
        }
        return out;
    }

}