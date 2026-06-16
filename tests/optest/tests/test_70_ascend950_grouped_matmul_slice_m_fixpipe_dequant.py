# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Test example 70_ascend950_grouped_matmul_slice_m_fixpipe_dequant."""

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
    reason="example 70_ascend950_grouped_matmul_slice_m_fixpipe_dequant requires Ascend 950 NPU",
)

def calc_gmm_fixpipe_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    group_sizes: torch.Tensor,
    scale: torch.Tensor,
    is_per_channel: bool = False,
) -> torch.Tensor: 
    scale = scale.npu() if scale.device.type != "npu" else scale
    expected_list = []
    offset = 0
    for i, size in enumerate(group_sizes):
        part = torch.matmul(a[offset:offset+size].float(), b[i].float())

        if is_per_channel:
            part = part * scale.reshape(1, -1).float()
        else:
            part = part * scale[0].float()
        expected_list.append(part)
        offset += size
    return torch.cat(expected_list, dim=0)

def test_fixpipe_dequant_per_channel():
    G, M, K, N = 2, 128, 512, 256

    a = torch.randint(-16, 16, (M, K), dtype=torch.int8).npu()
    b = torch.randint(-16, 16, (G, K, N), dtype=torch.int8).npu()

    group_sizes = torch.ones(G, dtype=torch.int64)*(M // G)
    group_list = torch.cumsum(group_sizes, dim=0).npu()

    per_tensor_scale = torch.tensor([0.5], dtype=torch.float32).npu()
    per_channel_scale = torch.rand((N,), dtype=torch.float32)
    per_channel_scale_uint64 = per_channel_scale.view(torch.uint32).to(torch.int64).view(torch.uint64).npu()

    expected = calc_gmm_fixpipe_golden(a, b, group_sizes, per_channel_scale, True)

    result = torch_catlass.grouped_matmul_slice_m_fixpipe_dequant(
        a, b, group_list, per_tensor_scale, per_channel_scale_uint64, per_channel_mode=True)

    assert result.shape == (M, N)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu().float(), expected.cpu(), rtol=1e-1, atol=1e-1)


def test_fixpipe_dequant_per_tensor():
    G, M, K, N = 2, 128, 512, 256

    a = torch.randint(-16, 16, (M, K), dtype=torch.int8).npu()
    b = torch.randint(-16, 16, (G, K, N), dtype=torch.int8).npu()

    group_sizes = torch.ones(G, dtype=torch.int64)*(M // G)
    group_list = torch.cumsum(group_sizes, dim=0).npu()

    per_tensor_scale = torch.tensor([0.5], dtype=torch.float32).npu()
    per_channel_scale_uint64 = torch.empty((N,), dtype=torch.uint64).npu()

    expected = calc_gmm_fixpipe_golden(a, b, group_sizes, per_tensor_scale, False)

    result = torch_catlass.grouped_matmul_slice_m_fixpipe_dequant(
        a, b, group_list, per_tensor_scale, per_channel_scale_uint64, per_channel_mode=False)

    assert result.shape == (M, N)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu().float(), expected.cpu(), rtol=1e-1, atol=1e-1)
