# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import torch
import torch_catlass
import torch_npu

from common import only_on_3510
from mx_golden import prepare_fp4_mx_pertoken_perchannel_inputs


def _fp4_dtype_supported() -> bool:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return False
    try:
        packed = torch.zeros(4, dtype=torch.uint8)
        out = torch.empty((2, 2), dtype=torch.float4_e2m1fn_x2)
        out.view(torch.uint8).flatten()[: packed.numel()].copy_(packed)
        return out.shape == (2, 2)
    except (RuntimeError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _fp4_dtype_supported(),
    reason="torch.float4_e2m1fn_x2 tensor construction is unavailable in this PyTorch build",
)


@only_on_3510
def test_ascend950_fp4_mx_matmul_pertoken_perchannel():
    """Compare example 74 against the quantized CPU reference."""
    m, n, k = 256, 256, 128
    a, b, mx_scale_a, mx_scale_b, per_token, per_channel, expected = (
        prepare_fp4_mx_pertoken_perchannel_inputs(m, n, k, device="npu")
    )

    result = torch_catlass.ascend950_fp4_mx_matmul_pertoken_perchannel(
        a, b, mx_scale_a, mx_scale_b, per_token, per_channel
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"

    torch_npu.npu.synchronize()
    result_cpu = result.cpu().float()
    diff = (result_cpu - expected).abs()
    threshold = (1.0 / 256) * torch.clamp(expected.abs(), min=1.0)
    assert bool((diff <= threshold).all()), f"max diff = {diff.max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

