# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

"""MX matmul golden reference helpers for optest (aligned with example gen_data.py layout)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

MX_SCALE_GROUP_NUM = 32
_FP4_BLOCK_SIZE = 32
_FP4_EPSILON = 1e-12
_FP4_MIN_SCALE_EXP = -128
_FP4_MAX_SCALE_EXP = 127

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
}


def _round_up2(v: int) -> int:
    return ((v + 1) // 2) * 2


def mx_scale_k(k: int) -> int:
    return _round_up2(math.ceil(k / MX_SCALE_GROUP_NUM))


def _build_fp4_lut(format_name: str) -> torch.Tensor:
    config = _FP4_FORMATS[format_name]
    exp_bits = int(config["exp_bits"])
    mantissa_bits = int(config["mantissa_bits"])
    bias = float(config["bias"])

    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = i & ((1 << mantissa_bits) - 1)

        if exp == 0:
            if mantissa == 0:
                value = 0.0
            else:
                value = (mantissa / float(1 << mantissa_bits)) * (2.0 ** (1.0 - bias))
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (2.0 ** (float(exp) - bias))

        if sign == 1:
            value = -value
        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {"E2M1": _build_fp4_lut("E2M1")}


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]
    clamped = values.clamp(min_value, max_value)
    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    return lut[indices], indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [index_matrix, torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device)],
            dim=1,
        )
    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    return (low | high).to(torch.uint8)


def _quantize_fp4_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int = _FP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    padded_n = ((n + block_size - 1) // block_size) * block_size
    num_blocks = padded_n // block_size

    if padded_n != n:
        padded = torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(torch.clamp(max_abs, min=_FP4_EPSILON))) - _FP4_FORMATS[format_name]["emax"]
    exp = torch.where(max_abs < _FP4_EPSILON, torch.zeros_like(exp), exp)
    exp = exp.clamp(_FP4_MIN_SCALE_EXP, _FP4_MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)

    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, format_name)
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)

    quantized = quantized_blocks.reshape(m, padded_n)
    dequantized = dequant_blocks.reshape(m, padded_n)
    if padded_n != n:
        quantized = quantized[:, :n].contiguous()
        dequantized = dequantized[:, :n].contiguous()

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_fp4_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int = _FP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_fp4_axis_last(
        matrix.t().contiguous(), format_name, block_size
    )
    return quantized_t.t().contiguous(), scale_t.t().contiguous(), dequantized_t.t().contiguous()


def _quantize_fp4(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _FP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_fp4_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_fp4_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def _gen_fp4_e2m1(row: int, col: int, axis: int, trans: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate packed FP4 matrix, MX scale, and dequant FP32 (example 54 gen_data_fp4_e2m1)."""
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize_fp4(matrix, "E2M1", axis)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    packed_uint8 = _pack_fp4_nibbles(fp4_indices)
    return packed_uint8, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def _packed_uint8_to_fp4(packed_uint8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Materialize packed uint8 nibbles as float4_e2m1fn_x2 with logical shape (rows, cols)."""
    packed = packed_uint8.contiguous().flatten()
    out = torch.empty((rows, cols), dtype=torch.float4_e2m1fn_x2)
    out.view(torch.uint8).flatten()[: packed.numel()].copy_(packed)
    return out


def _quantize_axis_last(
    matrix: torch.Tensor, fp8_dtype: torch.dtype, emax: int, fp8_max: float, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    num_blocks = (n + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    if padded_n != n:
        padded = torch.zeros(m, padded_n, dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(max_abs.clamp(min=1e-12))) - emax
    exp = exp.clamp(-128, 127)
    scale = torch.exp2(exp.to(torch.float32))
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    quant_fp8 = scaled.to(fp8_dtype)
    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)

    if padded_n != n:
        quant_fp8 = quant_fp8.reshape(m, padded_n)[:, :n].contiguous()
        dequant = dequant.reshape(m, padded_n)[:, :n].contiguous()
    else:
        quant_fp8 = quant_fp8.reshape(m, n)
        dequant = dequant.reshape(m, n)

    padded_blocks = _round_up2(num_blocks)
    if padded_blocks != num_blocks:
        scale_padded = torch.ones(m, padded_blocks, dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quant_fp8, scale.to(torch.float8_e8m0fnu), dequant


def _quantize_axis_first(
    matrix: torch.Tensor, fp8_dtype: torch.dtype, emax: int, fp8_max: float, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qt, st, dt = _quantize_axis_last(matrix.t().contiguous(), fp8_dtype, emax, fp8_max, block_size)
    return qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous()


def _move_kernel_inputs_to_npu(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move MX kernel inputs to the current NPU device (call after set_device)."""
    return (
        a.contiguous().npu(),
        b.contiguous().npu(),
        a_scale.contiguous().npu(),
        b_scale.contiguous().npu(),
    )


def prepare_fp8_mx_inputs(
    m: int, n: int, k: int, device: str = "npu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build MX-FP8 inputs for trans_a=0, trans_b=1 (default example layout).

    Quantization and reference matmul run on CPU; kernel inputs are moved to NPU
    when ``device="npu"``. ``expected`` stays on CPU for numerical comparison.
    """
    fp8_dtype = torch.float8_e4m3fn
    emax, fp8_max = 8, 448.0

    a_fp32 = torch.randn(m, k, dtype=torch.float32) * 10.0 - 5.0
    b_fp32 = torch.randn(k, n, dtype=torch.float32) * 10.0 - 5.0

    a_fp8, a_scale, a_deq = _quantize_axis_last(a_fp32, fp8_dtype, emax, fp8_max)
    b_fp8, b_scale, b_deq = _quantize_axis_first(b_fp32, fp8_dtype, emax, fp8_max)

    a_scale = a_scale.reshape(m, a_scale.shape[1] // 2, 2).contiguous()
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1]).permute(2, 0, 1).contiguous()
    b_fp8 = b_fp8.t().contiguous()
    a_fp8 = a_fp8.contiguous()

    expected = a_deq @ b_deq
    if device == "npu":
        a_fp8, b_fp8, a_scale, b_scale = _move_kernel_inputs_to_npu(a_fp8, b_fp8, a_scale, b_scale)
    return a_fp8, b_fp8, a_scale, b_scale, expected


def prepare_fp4_mx_inputs(
    m: int, n: int, k: int, device: str = "npu", trans_a: int = 0, trans_b: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build MX-FP4 inputs for trans_a=0, trans_b=1 (aligned with example 54 gen_data.py).

    Quantization and reference matmul run on CPU; kernel inputs are moved to NPU
    when ``device="npu"``. ``expected`` stays on CPU for numerical comparison.
    """
    a_packed, a_scale, a_deq = _gen_fp4_e2m1(m, k, axis=1, trans=trans_a)
    b_packed, b_scale, b_deq = _gen_fp4_e2m1(k, n, axis=0, trans=trans_b)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2).contiguous()
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1]).contiguous()
    if trans_a == 1:
        a_scale = a_scale.permute(1, 0, 2).contiguous()
    if trans_b == 1:
        b_scale = b_scale.permute(2, 0, 1).contiguous()
    else:
        b_scale = b_scale.permute(0, 2, 1).contiguous()

    if trans_b == 1:
        b_fp4 = _packed_uint8_to_fp4(b_packed, n, k)
    else:
        b_fp4 = _packed_uint8_to_fp4(b_packed, k, n).t().contiguous()
    a_fp4 = _packed_uint8_to_fp4(a_packed, m, k)

    expected = a_deq @ b_deq
    if device == "npu":
        a_fp4, b_fp4, a_scale, b_scale = _move_kernel_inputs_to_npu(a_fp4, b_fp4, a_scale, b_scale)
    return a_fp4, b_fp4, a_scale, b_scale, expected
