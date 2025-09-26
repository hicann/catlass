from matmul_common import matmul_case_decorator, matmul_case_runner


@matmul_case_decorator
def test_00_basic_matmul(
    m,
    k,
    n,
    transpose_a,
    transpose_b,
    input_dtypes,
    output_dtype,
):
    matmul_case_runner(
        "basic_matmul", m, k, n, transpose_a, transpose_b, input_dtypes, output_dtype
    )
