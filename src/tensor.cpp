#include <llm/tensor.hpp>

namespace llm
{


    void Storage::aligned_free()
    {
        std::free(buffer_ptr);
    }

    Storage::Storage(const Shape& shape, const DType& dtype)
    {
        int64_t length = numel(shape);
        // std::vector<float> data(length);
        // auto data = std::make_shared<std::vector<float>>(length);
        
        buffer_ptr = std::aligned_alloc(64, length*dtype_size(dtype));

        ++alloc_count();
    }

    Storage::~Storage()
    {
        aligned_free();
    }

    int& Storage::alloc_count()
    {
        static int object_count;
        return object_count;
    }

}