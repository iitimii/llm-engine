#include <cstddef>
#include <cstdint>

namespace llm
{
    enum class DType
    {
        F32,
        F16,
        I8
    };

    inline std::size_t dtype_size(DType dtype)
    {
        switch (dtype)
        {
        case DType::F32:
            return std::size_t(32);
        case DType::F16:
            return std::size_t(16);
        case DType::I8:
            return std::size_t(8);

            return std::size_t(0);
        }
    }

    inline char *dtype_name(DType dtype)
    {
        switch (dtype)
        {
        case DType::F32:
            return "F32";
        case DType::F16:
            return "F16";
        case DType::I8:
            return "I8";

            return "Invalid dtype";
        }
    }
}
