# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ruff: noqa: E501

import re

import pytest
import torch
import torch_npu

import torch_catlass


def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))


pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 88_ascend950_matmul_permute requires Ascend 950 NPU",
)


def _golden_matmul_permute(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor, s1: int = 8, s2: int = 4
) -> torch.Tensor:
    m = a_cpu.shape[0]
    n = b_cpu.shape[1]
    return torch.matmul(a_cpu, b_cpu).reshape(m // s1, s1, s2, n // s2).permute(0, 2, 1, 3).contiguous()


@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 4, 16),
        (256, 512, 256),
        (400, 800, 1024),
        (2048, 512, 2048),
        # Exercise the host vector-pack route.
        (1024, 1024, 16),
    ],
)
def test_ascend950_matmul_permute(m, n, k):
    # Keep the reference independent from the NPU implementation under test.
    a_cpu = torch.randn(m, k, dtype=torch.float16)
    b_cpu = torch.randn(k, n, dtype=torch.float16)
    expected = _golden_matmul_permute(a_cpu, b_cpu)

    a = a_cpu.to("npu")
    b = b_cpu.to("npu")

    result = torch_catlass.ascend950_matmul_permute(a, b)
    result_cpu = result.cpu()

    assert result.shape == (m // 8, 4, 8, n // 4)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result_cpu.float(), expected.float(), rtol=1e-2, atol=1e-2), (
        f"max diff = {(result_cpu.float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
