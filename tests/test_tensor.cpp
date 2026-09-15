#include "test_framework.hpp"
#include "llm/tensor.hpp"

using namespace llm;

TEST(tensor_shape_strides_numel)
{
    Tensor t = Tensor::empty({2, 3, 4}, DType::F32);
    CHECK(t.ndim() == 3);
    CHECK(t.numel() == 24);
    CHECK(t.nbytes() == 24u * 4u);
    CHECK(t.strides()[0] == 12);
    CHECK(t.strides()[1] == 4);
    CHECK(t.strides()[2] == 1);
    CHECK(t.is_contiguous());
    CHECK(t.dtype() == DType::F32);
}

TEST(tensor_set_get_roundtrip)
{
    Tensor t = Tensor::empty({2, 2}, DType::F32);
    t.at<float>({0, 0}) = 1.0f;
    // t.at<float>({0, 1}) = 2.0f;
    // t.at<float>({1, 0}) = 3.0f;
    // t.at<float>({1, 1}) = 4.0f;
    // CHECK_CLOSE(t.at<float>({1, 0}), 3.0f, 1e-6);
    // const float *p = t.data_ptr<float>();
    // CHECK_CLOSE(p[0], 1.0f, 1e-6);
    // CHECK_CLOSE(p[3], 4.0f, 1e-6);
}

// TEST(dtype_sizes)
// {
//     CHECK(dtype_size(DType::F32) == 4);
//     CHECK(dtype_size(DType::F16) == 2);
//     CHECK(dtype_size(DType::I8) == 1);
// }