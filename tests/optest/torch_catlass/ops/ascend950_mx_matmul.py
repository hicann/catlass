# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import torch
from torch import Tensor


def ascend950_fp8_mx_matmul_aswt(
    mat1: Tensor,
    mat2: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
    transA: bool = False,
    transB: bool = True,
) -> Tensor:
    """Run CATLASS Ascend950 MX FP8 matmul (ASWT scheduler) on NPU tensors.

    Source: example 53_ascend950_fp8_mx_matmul_aswt.

    Uses ``BlockSchedulerAswt`` instead of ``GemmIdentityBlockSwizzle``.
    Computes ``C = (MxScaleA * A) @ (MxScaleB * B)`` with float8_e4m3fn inputs
    and float8_e8m0fnu block scales. Output is FP32.

    Args:
        mat1: Left input (float8_e4m3fn), shape ``(M, K)`` unless ``transA``.
        mat2: Right input (float8_e4m3fn), shape ``(N, K)`` when ``transB=True``.
        mx_scale_a: MX scale for A (float8_e8m0fnu).
        mx_scale_b: MX scale for B (float8_e8m0fnu).
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed (default matches example).

    Returns:
        FP32 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_fp8_mx_matmul_aswt(
        mat1, mat2, mx_scale_a, mx_scale_b, transA, transB
    )


def ascend950_fp4_mx_matmul_aswt(
    mat1: Tensor,
    mat2: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
    transA: bool = False,
    transB: bool = True,
) -> Tensor:
    """Run CATLASS Ascend950 MX FP4 matmul (ASWT scheduler) on NPU tensors.

    Source: example 54_ascend950_fp4_mx_matmul_aswt.

    Uses ``BlockSchedulerAswt`` instead of ``GemmIdentityBlockSwizzle``.
    Computes ``C = (MxScaleA * A) @ (MxScaleB * B)`` with float4_e2m1fn_x2 inputs
    and float8_e8m0fnu block scales. Output is FP32.

    Args:
        mat1: Left input (float4_e2m1fn_x2), shape ``(M, K)`` unless ``transA``.
        mat2: Right input (float4_e2m1fn_x2), shape ``(N, K)`` when ``transB=True``.
        mx_scale_a: MX scale for A (float8_e8m0fnu).
        mx_scale_b: MX scale for B (float8_e8m0fnu).
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed (default matches example).

    Returns:
        FP32 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_fp4_mx_matmul_aswt(
        mat1, mat2, mx_scale_a, mx_scale_b, transA, transB
    )
