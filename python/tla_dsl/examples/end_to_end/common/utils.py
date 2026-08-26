# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch  # noqa: F401
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Helper functions below
# ---------------------------------------------------------------------------


def get_block_num(block_num: int, device: int = 0, *, kind: str = "vector") -> int:
    """Get launch ``block_num``.

    Non-``-1`` uses the host argument. ``-1`` means full-device launch:
    pure vector → ``vector_core_num`` (AIV); cube/mix → ``cube_core_num`` (AIC).
    """
    if int(block_num) != -1:
        return max(1, int(block_num))
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


# Signed integer view types by storage width, for handing a buffer over as plain
# storage when DLPack cannot describe its real format.
_STORAGE_VIEW_BY_WIDTH = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}


def create_tla_tensor(buf, layout: str, element_type=None):
    """Wrap a device buffer as a TLA tensor.

    ``element_type`` overrides the element type derived from the DLPack dtype.
    torch cannot export fp8 over DLPack at all, so an fp8 buffer is handed over
    as a same-width integer view with the real type supplied here.
    """
    tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
    if element_type is not None:
        # Derive the view from the override's own width rather than assuming one
        # byte: from_dlpack accepts any same-width override, and viewing through
        # the wrong width silently reshapes the buffer.
        width = element_type.width
        if width not in _STORAGE_VIEW_BY_WIDTH:
            raise ValueError(
                f"no storage view for a {width}-bit element type {element_type.__name__}"
            )
        buf = buf.view(_STORAGE_VIEW_BY_WIDTH[width])
    return from_dlpack(
        buf.contiguous(), layout_tag=tag, element_type=element_type
    ).mark_layout_dynamic()


def to_hf32(
    x: torch.Tensor,
    hf32_mode: tla.params.HF32Mode,
) -> torch.Tensor:
    """Simulate HF32 rounding mode on f32 values.

    HF32 keeps the FP32 sign and 8-bit exponent, and reduces the mantissa to
    11 significant bits (10 explicit mantissa bits, close to FP16):

    - ``HF32_NEAREST_ZERO`` rounds to nearest, ties toward zero.
    - ``HF32_NEAREST_EVEN`` rounds them to nearest-even.
    """
    if not isinstance(hf32_mode, tla.params.HF32Mode):
        raise TypeError(
            f"hf32_mode must be a tla.params.HF32Mode, got {type(hf32_mode).__name__}"
        )
    x = x.float()
    if hf32_mode == tla.params.HF32Mode.HF32_DISABLE:
        return x

    bits = x.contiguous().view(torch.int32)
    if hf32_mode == tla.params.HF32Mode.HF32_NEAREST_ZERO:
        rounded_bits = (bits + 0x0FFF) & ~0x1FFF
    elif hf32_mode == tla.params.HF32Mode.HF32_NEAREST_EVEN:
        lsb = (bits >> 13) & 1
        rounded_bits = (bits + 0x0FFF + lsb) & ~0x1FFF
    else:
        raise ValueError(f"Unsupported HF32 mode: {hf32_mode!r}")

    return rounded_bits.view(torch.float32)
