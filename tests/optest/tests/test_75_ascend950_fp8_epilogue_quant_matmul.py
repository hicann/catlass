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
from mx_golden import prepare_fp8_epilogue_quant_matmul_inputs


@only_on_3510
def test_ascend950_fp8_epilogue_quant_matmul():
    """Compare example 75 against the quantized CPU reference."""
    m, n, k = 256, 256, 128
    a, b, per_token, per_channel, expected = prepare_fp8_epilogue_quant_matmul_inputs(
        m, n, k, device="npu"
    )

    result = torch_catlass.ascend950_fp8_epilogue_quant_matmul(a, b, per_token, per_channel)

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

