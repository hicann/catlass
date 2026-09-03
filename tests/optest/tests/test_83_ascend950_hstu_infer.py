# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


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
    reason="example 83_ascend950_hstu_infer requires Ascend 950 NPU",
)


def _ref_hstu_attention(query, key, value, silu_scale: float, mask_type: int):
    """HSTU reference: P = silu_scale * S * sigmoid(S), O = P @ V.

    ``query``/``key``/``value`` are per-batch TND tensors
    ``(seq_len, num_heads, head_dim)`` on CPU.
    """
    query_t = query.transpose(0, 1).float()
    key_t = key.transpose(0, 1).float()
    value_t = value.transpose(0, 1).float()

    sim = torch.matmul(query_t, key_t.transpose(1, 2))
    p_high = silu_scale * sim * torch.sigmoid(sim)

    if mask_type == 1:
        q_len = sim.shape[-2]
        kv_len = sim.shape[-1]
        causal = (
            torch.arange(kv_len).unsqueeze(0) <= torch.arange(q_len).unsqueeze(1)
        ).to(p_high.dtype)
        p_high = p_high * causal

    p = p_high.to(query.dtype).float()
    out = torch.matmul(p, value_t)
    return out.transpose(0, 1).to(query.dtype)


def _run_case(
    batch,
    q_seqlen,
    kv_seqlen,
    num_heads,
    head_dim,
    silu_scale,
    mask_type,
    layout,
    paged_block_size,
):
    torch.manual_seed(1)

    q_seqlen_list = [q_seqlen] * batch
    kv_seqlen_list = [kv_seqlen] * batch
    num_tokens = sum(q_seqlen_list)
    total_kv_tokens = sum(kv_seqlen_list)

    # TND layout inputs on NPU.
    query = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16, device="npu")

    expected_batches = []
    if paged_block_size == 0:
        key = torch.randn(total_kv_tokens, num_heads, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(total_kv_tokens, num_heads, head_dim, dtype=torch.float16, device="npu")
        block_table = torch.empty(0, dtype=torch.int32, device="npu")

        q_cpu = query.cpu()
        k_cpu = key.cpu()
        v_cpu = value.cpu()
        cu_q = 0
        cu_kv = 0
        for q_len, kv_len in zip(q_seqlen_list, kv_seqlen_list):
            expected_batches.append(
                _ref_hstu_attention(
                    q_cpu[cu_q : cu_q + q_len],
                    k_cpu[cu_kv : cu_kv + kv_len],
                    v_cpu[cu_kv : cu_kv + kv_len],
                    silu_scale,
                    mask_type,
                )
            )
            cu_q += q_len
            cu_kv += kv_len
    else:
        # Paged KV cache: NHD layout (num_blocks, block_size, kv_heads, head_dim).
        blocks_per_batch = (kv_seqlen + paged_block_size - 1) // paged_block_size
        num_blocks = batch * blocks_per_batch
        key_cache = torch.randn(
            num_blocks, paged_block_size, num_heads, head_dim, dtype=torch.float16, device="npu"
        )
        value_cache = torch.randn(
            num_blocks, paged_block_size, num_heads, head_dim, dtype=torch.float16, device="npu"
        )
        block_table = torch.tensor(
            [[blocks_per_batch * i + j for j in range(blocks_per_batch)] for i in range(batch)],
            dtype=torch.int32,
            device="npu",
        )
        key = key_cache
        value = value_cache

        q_cpu = query.cpu()
        k_cpu = key_cache.cpu()
        v_cpu = value_cache.cpu()
        table_cpu = block_table.cpu()
        cu_q = 0
        for i, q_len in enumerate(q_seqlen_list):
            kv_len = kv_seqlen_list[i]
            keys = []
            values = []
            for j in range(kv_len):
                block_number = int(table_cpu[i, j // paged_block_size].item())
                block_offset = j % paged_block_size
                keys.append(k_cpu[block_number, block_offset])
                values.append(v_cpu[block_number, block_offset])
            key_batch = torch.stack(keys, dim=0)
            value_batch = torch.stack(values, dim=0)
            expected_batches.append(
                _ref_hstu_attention(
                    q_cpu[cu_q : cu_q + q_len], key_batch, value_batch, silu_scale, mask_type
                )
            )
            cu_q += q_len

    expected = torch.cat(expected_batches, dim=0)

    # Cumulative sequence lengths (cu_seqlen, batch + 1 int64 entries).
    q_cu = [0]
    for q_len in q_seqlen_list:
        q_cu.append(q_cu[-1] + q_len)
    kv_cu = [0]
    for kv_len in kv_seqlen_list:
        kv_cu.append(kv_cu[-1] + kv_len)
    actual_q_seqlen = torch.tensor(q_cu, dtype=torch.int64, device="npu")
    actual_kv_seqlen = torch.tensor(kv_cu, dtype=torch.int64, device="npu")

    if layout == "NTD":
        query = query.transpose(0, 1).contiguous()
        expected = expected.transpose(0, 1).contiguous()
        if paged_block_size == 0:
            key = key.transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()

    result = torch_catlass.ascend950_hstu_infer(
        query,
        key,
        value,
        actual_q_seqlen,
        actual_kv_seqlen,
        block_table,
        layout,
        num_heads,
        num_heads,
        paged_block_size,
        silu_scale,
        mask_type,
    )

    assert result.shape == query.shape
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result.cpu().float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.cpu().float() - expected.float()).abs().max().item()}"
    )


def test_ascend950_hstu_infer_tnd():
    """TND layout, dense KV, no mask."""
    _run_case(
        batch=2,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=8,
        head_dim=256,
        silu_scale=0.1,
        mask_type=0,
        layout="TND",
        paged_block_size=0,
    )


def test_ascend950_hstu_infer_tnd_causal():
    """TND layout, dense KV, causal mask."""
    _run_case(
        batch=2,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=8,
        head_dim=256,
        silu_scale=0.1,
        mask_type=1,
        layout="TND",
        paged_block_size=0,
    )


def test_ascend950_hstu_infer_ntd():
    """NTD layout, dense KV, no mask."""
    _run_case(
        batch=2,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=8,
        head_dim=256,
        silu_scale=0.1,
        mask_type=0,
        layout="NTD",
        paged_block_size=0,
    )


def test_ascend950_hstu_infer_paged():
    """TND Q layout with paged KV cache (NHD) and causal mask."""
    _run_case(
        batch=2,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=8,
        head_dim=256,
        silu_scale=0.1,
        mask_type=1,
        layout="TND",
        paged_block_size=128,
    )


def test_ascend950_hstu_infer_head_dim_min():
    """Boundary: head_dim = 1 is the minimum supported embedding size."""
    _run_case(
        batch=1,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=4,
        head_dim=1,
        silu_scale=0.1,
        mask_type=1,
        layout="TND",
        paged_block_size=0,
    )


def test_ascend950_hstu_infer_head_dim_mid():
    """Boundary: head_dim inside the supported range (e.g. 128) with paged KV cache."""
    _run_case(
        batch=2,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=8,
        head_dim=128,
        silu_scale=0.1,
        mask_type=1,
        layout="TND",
        paged_block_size=128,
    )


def test_ascend950_hstu_infer_head_dim_max():
    """Boundary: head_dim = 256 is the maximum supported embedding size."""
    _run_case(
        batch=1,
        q_seqlen=256,
        kv_seqlen=256,
        num_heads=4,
        head_dim=256,
        silu_scale=0.1,
        mask_type=1,
        layout="NTD",
        paged_block_size=0,
    )


def _call_op_with_head_dim(head_dim, layout, paged_block_size):
    """Build minimal inputs with the given head_dim and call the operator."""
    batch, q_seqlen, kv_seqlen, num_heads = 1, 128, 128, 2
    query = torch.randn(
        batch * q_seqlen, num_heads, head_dim, dtype=torch.float16, device="npu"
    )
    if paged_block_size == 0:
        total_kv = batch * kv_seqlen
        key = torch.randn(total_kv, num_heads, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(total_kv, num_heads, head_dim, dtype=torch.float16, device="npu")
        block_table = torch.empty(0, dtype=torch.int32, device="npu")
    else:
        blocks_per_batch = (kv_seqlen + paged_block_size - 1) // paged_block_size
        num_blocks = batch * blocks_per_batch
        key = torch.randn(
            num_blocks, paged_block_size, num_heads, head_dim,
            dtype=torch.float16, device="npu",
        )
        value = torch.randn(
            num_blocks, paged_block_size, num_heads, head_dim,
            dtype=torch.float16, device="npu",
        )
        block_table = torch.tensor(
            [[blocks_per_batch * i + j for j in range(blocks_per_batch)] for i in range(batch)],
            dtype=torch.int32,
            device="npu",
        )
    if layout == "NTD":
        query = query.transpose(0, 1).contiguous()
        if paged_block_size == 0:
            key = key.transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()
    actual_q_seqlen = torch.tensor([0, q_seqlen], dtype=torch.int64, device="npu")
    actual_kv_seqlen = torch.tensor([0, kv_seqlen], dtype=torch.int64, device="npu")
    return torch_catlass.ascend950_hstu_infer(
        query,
        key,
        value,
        actual_q_seqlen,
        actual_kv_seqlen,
        block_table,
        layout,
        num_heads,
        num_heads,
        paged_block_size,
        0.1,
        1,
    )


@pytest.mark.parametrize("head_dim", [257, 512])
def test_ascend950_hstu_infer_head_dim_out_of_range(head_dim):
    """Boundary: head_dim > 256 must be rejected (dense KV, TND layout)."""
    with pytest.raises(RuntimeError, match="only supports head_dim in range"):
        _call_op_with_head_dim(head_dim, layout="TND", paged_block_size=0)


@pytest.mark.parametrize("head_dim", [257, 512])
def test_ascend950_hstu_infer_head_dim_out_of_range_paged(head_dim):
    """Boundary: head_dim > 256 must be rejected (paged KV cache)."""
    with pytest.raises(RuntimeError, match="only supports head_dim in range"):
        _call_op_with_head_dim(head_dim, layout="TND", paged_block_size=128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
