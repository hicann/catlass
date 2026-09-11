# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
from torch import Tensor


def _optional_block_table(value: Optional[Tensor], query: Tensor) -> Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.int32, device=query.device)
    return value


def x_attention(
    query: Tensor,
    shared_key_block: Tensor,
    shared_value_block: Tensor,
    unshared_key_block: Tensor,
    unshared_value_block: Tensor,
    unshared_block_table: Optional[Tensor],
    shared_kv_lens: Tensor,
    decode_step: Tensor,
    shared_block_table: Optional[Tensor],
    scale_value: float = 0.0,
) -> Tensor:
    """Run CATLASS XAttention on NPU tensors.

    Source: example 78_x_attention. XAttention combines a batch-shared KV
    context with a beam-private decode cache in one softmax.

    Exactly one cache is paged:

    - ``shared_block_table`` is set and ``unshared_block_table`` is ``None``:
      shared paged cache plus unshared continuous cache.
    - ``shared_block_table`` is ``None`` and ``unshared_block_table`` is set:
      shared continuous cache plus unshared paged cache.

    Contract (see examples/78_x_attention/README.md): positive batch, beam_size,
    num_heads and kv_heads; num_heads % kv_heads == 0; GQA group size <= 128;
    head dimension == 128; 1 <= decode_step <= max_decode_step <= 256.
    All Q/K/V use the same float16 or bfloat16 dtype. All input tensors must
    be contiguous and on the same NPU. Attention masks are not supported.
    The 128-token block size applies only to the shared paged cache; the
    unshared paged cache is indexed by request and uses max_decode_step capacity.
    Length and block-index tensor values must be valid for the cache capacity;
    the wrapper does not check these device-resident values on the host.

    Args:
        query: Query in TND layout ``(batch * beam_size, num_heads, 128)``.
        shared_key_block: Shared key cache. Paged shape is
            ``(num_blocks, 128, kv_heads, 128)``; continuous shape is
            ``(batch * shared_kv_seq_len, kv_heads, 128)``.
        shared_value_block: Shared value cache with the same shape as the shared key cache.
        unshared_key_block: Beam-private key cache. Continuous shape is
            ``(batch * beam_size, kv_heads, max_decode_step, 128)``; paged shape is
            ``(request_count, beam_size, kv_heads, max_decode_step, 128)``.
        unshared_value_block: Unshared value cache with the same shape as the unshared key cache.
        unshared_block_table: Request index for each batch in unshared-paged mode, otherwise ``None``.
        shared_kv_lens: Positive actual shared KV length for each batch, int32
            shape ``(batch,)``. Paged lengths must not exceed table capacity.
            Continuous batches are packed by these lengths; example/ATK use
            equal lengths matching shared_kv_seq_len, without inter-batch padding.
        decode_step: Number of valid unshared cache tokens, int32 shape ``(1,)``;
            one value shared by all batches/beams, in ``[1, max_decode_step]``.
        shared_block_table: Shared paged-cache block table, int32 shape
            ``(batch, max_blocks_per_batch)``, otherwise ``None``.
        scale_value: Positive attention scale, or ``0.0`` for ``1 / sqrt(128)``.

    Returns:
        Tensor with the same shape and dtype as ``query``.
    """
    unshared_table = _optional_block_table(unshared_block_table, query)
    shared_table = _optional_block_table(shared_block_table, query)
    return torch.ops.catlass.x_attention(
        query,
        shared_key_block,
        shared_value_block,
        unshared_key_block,
        unshared_value_block,
        unshared_table,
        shared_kv_lens,
        decode_step,
        shared_table,
        scale_value,
    )
