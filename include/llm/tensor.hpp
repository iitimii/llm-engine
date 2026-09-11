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
    return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

inline Strides contiguous_strides(const Shape& shape){

}

class Storage
{
    private:
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* buffer_ptr;

    void aligned_free();

    public:
    Storage(const Shape& shape, const DType& dtype);

    ~Storage();

    static int& alloc_count();

};


class Tensor
{
    private:
    std::shared_ptr<Storage> storage_ptr;
    int offset;
    Shape shape;
    Strides strides;
    DType dtype;

    public:
    Tensor(const Shape& shape, const DType& dtype, int alignment=64);
};

}