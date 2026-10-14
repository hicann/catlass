# ruff: noqa: N803, N806

import torch
from torch import Tensor

_DTYPE_MAP = {
    "float16": torch.float16,
    "half": torch.float16,
}


def ascend950_batched_matmul_softmax(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
) -> Tensor:
    """Run CATLASS Ascend950 BatchedMatmulSoftmax on NPU tensors.

    This is the Python interface for example
    ``87_ascend950_batched_matmul_softmax``. It computes
    ``softmax(bmm(mat1, mat2), dim=-1)``. ``mat1`` must be RowMajor contiguous
    and ``mat2`` must be physically ColumnMajor with strides ``(K * N, 1, K)``
    for logical shape ``(B, K, N)``. The output is an FP16 NPU tensor with
    shape ``(B, M, N)``.
    """
    if isinstance(outDType, str):
        dtype_name = outDType.lower()
        if dtype_name not in _DTYPE_MAP:
            raise ValueError("outDType must be float16 or half")
        outDType = _DTYPE_MAP[dtype_name]
    if outDType is not torch.float16:
        raise ValueError("outDType must be torch.float16")
    return torch.ops.catlass.ascend950_batched_matmul_softmax(mat1, mat2, outDType)
