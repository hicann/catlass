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

def create_tla_tensor(buf, layout: str):
    tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
    return from_dlpack(buf.contiguous(), layout_tag=tag).mark_layout_dynamic()

