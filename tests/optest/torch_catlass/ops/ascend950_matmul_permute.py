# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# The public torch wrapper keeps the CATLASS argument names for API compatibility.
# ruff: noqa: E501, N803, N806

import torch
from torch import Tensor


def ascend950_matmul_permute(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 Matmul + Tensor4DPermute0213.

    Source: experimental/matmul/88_ascend950_matmul_permute.

    Computes ``C = mat1 @ mat2`` and writes the flattened output using
    ``reshape(M / 8, 8, 4, N / 4).permute(0, 2, 1, 3)`` layout.
    The returned tensor shape is ``(M / 8, 4, 8, N / 4)``.
    """
    if mat1.dtype != torch.float16 or mat2.dtype != torch.float16:
        raise ValueError("ascend950_matmul_permute currently requires float16 inputs")
    if transA or transB or formatA or formatB:
        raise ValueError(
            "ascend950_matmul_permute currently supports only RowMajor non-transposed inputs"
        )
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError("ascend950_matmul_permute expects 2D inputs")
    if not mat1.is_contiguous() or not mat2.is_contiguous():
        raise ValueError("ascend950_matmul_permute requires contiguous inputs")
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("ascend950_matmul_permute requires mat1.shape[1] == mat2.shape[0]")

    m = mat1.shape[0]
    n = mat2.shape[1]
    if m % 8 != 0:
        raise ValueError("ascend950_matmul_permute requires M % 8 == 0")
    if n % 4 != 0:
        raise ValueError("ascend950_matmul_permute requires N % 4 == 0")
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    if outDType != torch.float16:
        raise ValueError("ascend950_matmul_permute currently produces float16 output")

    out = torch.ops.catlass.ascend950_matmul_permute(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
    # The Ascend950 epilogue already writes the flattened output in 0213
    # order.  Reinterpret that storage directly; permuting here would apply
    # the layout transform a second time.
    return out.view(m // 8, 4, 8, n // 4)
