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
    t.at<float>({0, 1}) = 2.0f;
    t.at<float>({1, 0}) = 3.0f;
    t.at<float>({1, 1}) = 4.0f;
    CHECK_CLOSE(t.at<float>({1, 0}), 3.0f, 1e-6);
    const float *p = t.data_ptr<float>();
    CHECK_CLOSE(p[0], 1.0f, 1e-6);
    CHECK_CLOSE(p[3], 4.0f, 1e-6);
}

TEST(dtype_sizes)
{
    CHECK(dtype_size(DType::F32) == 4);
    CHECK(dtype_size(DType::F16) == 2);
    CHECK(dtype_size(DType::I8) == 1);
}

TEST(reshape_zero_copy_preserves_data)
{
    Tensor t = Tensor::empty({2, 6}, DType::F32);
    for (int i = 0; i < 12; ++i)
        t.data_ptr<float>()[i] = float(i);
    Tensor r = t.reshape({3, 4});
    CHECK(r.storage_id() == t.storage_id()); // SAME storage -> no copy
    CHECK(r.is_contiguous());
    CHECK_CLOSE(r.at<float>({2, 3}), 11.0f, 1e-6); // last element preserved
    r.at<float>({0, 0}) = 99.0f;                   // mutate the view...
    CHECK_CLOSE(t.at<float>({0, 0}), 99.0f, 1e-6); // ...base sees it (same bytes)
}

TEST(transpose_swaps_strides_zero_copy)
{
    Tensor t = Tensor::empty({2, 3}, DType::F32);
    for (int i = 0; i < 6; ++i)
        t.data_ptr<float>()[i] = float(i);
    Tensor tt = t.transpose(0, 1); // shape {3,2}
    CHECK(tt.storage_id() == t.storage_id());
    CHECK(tt.shape()[0] == 3 && tt.shape()[1] == 2);
    CHECK(!tt.is_contiguous());
    for (int i = 0; i < 2; ++i) // A[i][j] == A^T[j][i]
        for (int j = 0; j < 3; ++j)
            CHECK_CLOSE(t.at<float>({i, j}), tt.at<float>({j, i}), 1e-6);
}

TEST(slice_views_subregion)
{
    Tensor t = Tensor::empty({4, 4}, DType::F32);
    for (int i = 0; i < 16; ++i)
        t.data_ptr<float>()[i] = float(i);
    Tensor rows = t.slice(0, 1, 3); // rows 1..2 -> shape {2,4}
    CHECK(rows.storage_id() == t.storage_id());
    CHECK(rows.shape()[0] == 2 && rows.shape()[1] == 4);
    CHECK_CLOSE(rows.at<float>({0, 0}), 4.0f, 1e-6);  // == t[1][0]
    CHECK_CLOSE(rows.at<float>({1, 3}), 11.0f, 1e-6); // == t[2][3]
}

TEST(contiguous_materializes_a_transpose)
{
    Tensor t = Tensor::empty({2, 3}, DType::F32);
    for (int i = 0; i < 6; ++i)
        t.data_ptr<float>()[i] = float(i);
    Tensor c = t.transpose(0, 1).contiguous(); // {3,2}, contiguous, real copy
    CHECK(c.is_contiguous());
    CHECK(c.storage_id() != t.storage_id()); // different storage -> a copy
    const float *p = c.data_ptr<float>();    // memory order now
    CHECK_CLOSE(p[0], 0.0f, 1e-6);
    CHECK_CLOSE(p[1], 3.0f, 1e-6);
    CHECK_CLOSE(p[2], 1.0f, 1e-6);
}

TEST(reshape_requires_contiguous)
{
    Tensor t = Tensor::empty({2, 3}, DType::F32);
    Tensor tt = t.transpose(0, 1);
    Tensor ok = tt.contiguous().reshape({6});
    CHECK(ok.numel() == 6);
    CHECK(ok.is_contiguous());
    // non-contiguous
    // the supported path
}