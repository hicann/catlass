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


def ascend950_fp4_mx_quant_matmul(
    mat1: Tensor,
    mat2: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
    per_token_scale: Tensor,
    per_channel_scale: Tensor,
) -> Tensor:
    """Run CATLASS Ascend950 FP4 MX matmul with per-token/per-channel epilogue scaling.

    Source: experimental/matmul/ascend950_fp4_mx_quant_matmul.

    Args:
        mat1: Packed FP4 left input, logical shape ``(M, K)``.
        mat2: Packed FP4 right input, logical shape ``(K, N)``.
        mx_scale_a: MX scale for A, dtype ``torch.float8_e8m0fnu``, contiguous storage
            ordered as ``(M, ceil(K / 64), 2)``; each scale covers 32 K elements.
        mx_scale_b: MX scale for RowMajor B, dtype ``torch.float8_e8m0fnu``, contiguous
            storage ordered as ``(ceil(K / 64), N, 2)``.
        per_token_scale: Per-row epilogue scale, dtype ``torch.float8_e4m3fn``, shape ``(M,)``.
        per_channel_scale: Per-column epilogue scale, dtype ``torch.float8_e4m3fn``, shape ``(N,)``.

    Returns:
        FP32 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_fp4_mx_quant_matmul(
        mat1, mat2, mx_scale_a, mx_scale_b, per_token_scale, per_channel_scale
    )


def ascend950_fp8_e4m3_quant_matmul(
    mat1: Tensor,
    mat2: Tensor,
    per_token_scale: Tensor,
    per_channel_scale: Tensor,
) -> Tensor:
    """Run CATLASS Ascend950 FP8 E4M3 matmul with per-token/per-channel scaling.

    Source: experimental/matmul/ascend950_fp8_e4m3_quant_matmul.

    Args:
        mat1: FP8 left input, shape ``(M, K)``.
        mat2: FP8 right input, shape ``(K, N)``.
        per_token_scale: Per-row epilogue scale, dtype ``torch.float8_e4m3fn``, shape ``(M,)``.
        per_channel_scale: Per-column epilogue scale, dtype ``torch.float8_e4m3fn``, shape ``(N,)``.

    Returns:
        FP32 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_fp8_e4m3_quant_matmul(
        mat1, mat2, per_token_scale, per_channel_scale
    )
