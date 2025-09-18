import pytest
import pandas as pd
import torch
from catlass_test import basic_matmul

cases = pd.read_csv("matmul.csv")

test_cases = [
    (
        row["M"],
        row["K"],
        row["N"],
        row["TransposeA"],
        row["TransposeB"],
        row["InputDtypes"],
        row["OutputDtype"],
    )
    for _, row in cases.iterrows()
]
test_cases = test_cases[0:20]


# 定义参数化装饰器
@pytest.mark.parametrize(
    "m, k, n, transpose_a, transpose_b, input_dtypes, output_dtype",
    test_cases,
)
def test_basic_matmul(
    m,
    k,
    n,
    transpose_a,
    transpose_b,
    input_dtypes,
    output_dtype,
):
    dtype = eval(f"torch.{output_dtype}")
    shape1 = (m, k) if not transpose_a else (k, m)
    shape2 = (k, n) if not transpose_b else (n, k)

    a = torch.rand(shape1, device="npu").to(dtype)
    b = torch.rand(shape2, device="npu").to(dtype)

    a = a if not transpose_a else a.T
    b = b if not transpose_b else b.T
    
    torch.npu.synchronize()
    result = basic_matmul(a, b, **{"TransA": transpose_a, "TransB": transpose_b})
    golden = torch.mm(a, b)
    torch.npu.synchronize()
    if dtype == torch.bfloat16:
        result = result.to(torch.float32)
        golden = golden.to(torch.float32)
    assert torch.allclose(result, golden, rtol=0.001, atol=0.001)
