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


def ascend950_rain_fusion_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    select_idx: Tensor,
    select_num_idx: Tensor,
    block_shape: Tensor,
    actual_seq_lengths: Tensor,
    actual_seq_lengths_kv: Tensor,
    input_layout: str = "BNSD",
    num_heads: int = 0,
    num_key_value_heads: int = 0,
    is_varied_len: int = 0,
) -> Tensor:
    """Run CATLASS Rain Fusion Attention on NPU tensors.

    Source: example 81_ascend950_rain_fusion_attention.

    Args:
        query: Query tensor.
            TND: ``(total_q_tokens, num_heads, head_dim)``.
            BNSD: ``(batch, num_heads, max_q_seqlen, head_dim)``.
        key: Key tensor with same layout as query.
            TND: ``(total_kv_tokens, kv_heads, head_dim)``.
            BNSD: ``(batch, kv_heads, max_kv_seqlen, head_dim)``.
        value: Value tensor with same layout as key.
        select_idx: Sparse KV block indices, shape
            ``(total_qs_block_num, num_heads, max_kv_block_num)``.
        select_num_idx: Number of selected KV blocks per Q block, shape
            ``(total_qs_block_num, num_heads)``.
        block_shape: Block shape ``[block_shape_x, block_shape_y]``, shape ``(2,)``.
        actual_seq_lengths: Per-batch Q sequence lengths, shape ``(batch,)``.
        actual_seq_lengths_kv: Per-batch KV sequence lengths, shape ``(batch,)``.
        input_layout: ``"BNSD"`` or ``"TND"``.
        num_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        is_varied_len: Whether to use variable-length input (0 or 1).

    Returns:
        Output tensor with same shape as query.
    """
    return torch.ops.catlass.ascend950_rain_fusion_attention(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        block_shape,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        input_layout,
        num_heads,
        num_key_value_heads,
        is_varied_len,
    )
