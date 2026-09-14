# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full license.

import torch
import torch_npu


QUANT_MATMUL_SHAPES = [
    (1, 2, 2),
    (31, 34, 30),
    (127, 130, 66),
    (255, 258, 130),
    (256, 256, 128),
    (257, 514, 514),
    (513, 258, 1026),
    (1024, 512, 2048),
    (1537, 1026, 578),
    (4097, 2050, 66),
]


def with_storage_offset(tensor):
    """创建带 64 字节偏移的连续视图，保持原数据和设备不变。"""
    raw = tensor.view(torch.uint8).reshape(-1)
    padded = torch.empty(raw.numel() + 64, dtype=torch.uint8, device=tensor.device)
    padded[:64].zero_()
    padded[64:].copy_(raw)
    view = padded[64:].view(tensor.dtype).reshape(tensor.shape)
    assert view.is_contiguous() and view.storage_offset() > 0
    return view


def signed_scales(per_token, per_channel, expected):
    """增加零 scale、负 scale；符号翻转不引入新的舍入误差。"""
    row_factor = torch.where(torch.arange(per_token.numel()) % 2 == 0, -1.0, 1.0)
    col_factor = torch.where(torch.arange(per_channel.numel()) % 3 == 0, -1.0, 1.0)
    row_factor[0] = 0.0
    col_factor[-1] = 0.0
    token = (per_token.cpu().float() * row_factor).to(per_token.dtype).to(per_token.device)
    channel = (per_channel.cpu().float() * col_factor).to(per_channel.dtype).to(per_channel.device)
    return token, channel, expected * row_factor[:, None] * col_factor[None, :]


def check_quant_result(result, expected, record_property):
    assert result.shape == expected.shape
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    torch_npu.npu.synchronize()
    actual = result.cpu().float()
    assert torch.isfinite(actual).all()
    diff = (actual - expected).abs()
    threshold = (1.0 / 256) * torch.clamp(expected.abs(), min=1.0)
    record_property("max_abs_error", diff.max().item())
    record_property("max_threshold_ratio", (diff / threshold).max().item())
    record_property("compared_elements", expected.numel())
    assert bool((diff <= threshold).all()), f"max diff = {diff.max().item()}"
