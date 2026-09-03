# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch
from torch import Tensor


def ascend950_hstu_infer(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    actual_q_seqlen: Tensor,
    actual_kv_seqlen: Tensor,
    block_table: Tensor,
    input_layout: str = "TND",
    num_heads: int = 0,
    num_kv_heads: int = 0,
    paged_block_size: int = 0,
    silu_scale: float = 0.1,
    mask_type: int = 0,
) -> Tensor:
    """Run CATLASS Ascend950 HSTU inference attention on NPU tensors.

    Source: example 83_ascend950_hstu_infer.

    HSTU replaces softmax attention with a scaled SiLU activation:
    ``P = silu_scale * S * sigmoid(S)`` where ``S = Q @ K^T`` and
    ``O = P @ V``.

    Args:
        query: Query tensor. TND layout ``(total_q_tokens, num_heads,
            head_dim)`` or NTD layout ``(num_heads, total_q_tokens, head_dim)``.
        key: Key tensor. Non-paged: TND ``(total_kv_tokens, num_kv_heads,
            head_dim)`` or NTD ``(num_kv_heads, total_kv_tokens, head_dim)``.
            Paged: NHD ``(num_blocks, paged_block_size, num_kv_heads,
            head_dim)``.
        value: Value tensor with the same layout as ``key``.
        actual_q_seqlen: Cumulative Q sequence lengths (cu_seqlen), int64,
            shape ``(batch + 1,)`` with leading 0.
        actual_kv_seqlen: Cumulative KV sequence lengths, int64, shape
            ``(batch + 1,)`` with leading 0.
        block_table: Paged KV block table, int32 ``(batch, max_num_blocks)``.
            Pass an empty tensor when paged cache is not used.
        input_layout: ``TND`` or ``NTD`` for Q/O (and non-paged KV).
        num_heads: Number of query heads (must equal num_kv_heads).
        num_kv_heads: Number of KV heads (must equal num_heads).
        paged_block_size: ``0`` disables paged KV cache; ``> 0`` enables it
            with the given block size.
        silu_scale: SiLU activation scale.
        mask_type: ``0`` for no mask, ``1`` for causal multiplicative mask.

    Returns:
        Output tensor with the same shape and layout as ``query``.
    """
    return torch.ops.catlass.ascend950_hstu_infer(
        query,
        key,
        value,
        actual_q_seqlen,
        actual_kv_seqlen,
        block_table,
        input_layout,
        num_heads,
        num_kv_heads,
        paged_block_size,
        silu_scale,
        mask_type,
    )
