# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import numpy as np
import pytest
import torch

import torch_catlass
from common import only_on_3510


def _ref_rfa_single(q_blk, k_gathered, v_gathered, softmax_scale):
    """Compute one Q-block attention with gathered KV (CPU reference)."""
    q_blk_fp32 = q_blk.astype(np.float32)
    k_g_fp32 = k_gathered.astype(np.float32)
    v_g_fp32 = v_gathered.astype(np.float32)

    qk = np.matmul(q_blk_fp32, k_g_fp32.transpose(1, 0)) * softmax_scale
    row_max = np.max(qk, axis=-1, keepdims=True)
    sim_sub = qk - row_max
    sim_exp = np.exp(sim_sub)
    row_sum = np.sum(sim_exp, axis=-1, keepdims=True)
    attn = sim_exp / row_sum
    out = np.matmul(attn, v_g_fp32)
    return out.astype(np.float16)


def _reference_rain_fusion_attention(
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
):
    """CPU reference implementation for Rain Fusion Attention."""
    batch = actual_seq_lengths.shape[0]
    head_dim = query.shape[-1]
    block_shape_x, block_shape_y = int(block_shape[0]), int(block_shape[1])
    softmax_scale = 1.0 / np.sqrt(head_dim)
    group_size = num_heads // num_key_value_heads

    out = np.zeros_like(query, dtype=np.float32)

    # select_idx has shape [total_q_s_block_num, num_heads, max_kv_block_num]
    # We iterate per batch and track a running block_offset into select_idx's first dim.
    block_offset = 0
    for b in range(batch):
        q_seqlen = int(actual_seq_lengths[b])
        kv_seqlen = int(actual_seq_lengths_kv[b])
        q_s_block_num = (q_seqlen + block_shape_x - 1) // block_shape_x

        for q_s_blk_idx in range(q_s_block_num):
            q_start = q_s_blk_idx * block_shape_x
            q_end = min(q_start + block_shape_x, q_seqlen)
            if q_start >= q_seqlen:
                break

            for n1 in range(num_heads):
                n2 = n1 // group_size
                if input_layout == "BNSD":
                    q_blk = query[b, n1, q_start:q_end, :].astype(np.float16)
                else:
                    q_offset = int(actual_seq_lengths[:b].sum()) if b > 0 else 0
                    q_blk = query[q_offset + q_start : q_offset + q_end, n1, :].astype(np.float16)

                # Index into select_idx with the global block offset
                cur_block_idx = block_offset + q_s_blk_idx
                num_selected = int(select_num_idx[cur_block_idx, n1])
                selected_kv_blocks = select_idx[cur_block_idx, n1, :num_selected]

                k_tiles = []
                v_tiles = []
                for idx in selected_kv_blocks:
                    idx = int(idx)
                    kv_start = idx * block_shape_y
                    kv_end = min(kv_start + block_shape_y, kv_seqlen)
                    if input_layout == "BNSD":
                        k_tile = key[b, n2, kv_start:kv_end, :]
                        v_tile = value[b, n2, kv_start:kv_end, :]
                    else:
                        kv_offset = int(actual_seq_lengths_kv[:b].sum()) if b > 0 else 0
                        k_tile = key[kv_offset + kv_start : kv_offset + kv_end, n2, :]
                        v_tile = value[kv_offset + kv_start : kv_offset + kv_end, n2, :]
                    k_tiles.append(k_tile)
                    v_tiles.append(v_tile)

                if len(k_tiles) == 0:
                    continue

                k_gathered = np.concatenate(k_tiles, axis=0)
                v_gathered = np.concatenate(v_tiles, axis=0)

                out_slice = _ref_rfa_single(q_blk, k_gathered, v_gathered, softmax_scale)

                if input_layout == "BNSD":
                    out[b, n1, q_start:q_end, :] = out_slice
                else:
                    q_offset = int(actual_seq_lengths[:b].sum()) if b > 0 else 0
                    out[q_offset + q_start : q_offset + q_end, n1, :] = out_slice

        block_offset += q_s_block_num

    return out.astype(np.float16)


def _gen_select_idx(batch, num_heads, q_seqlen_list, kv_seqlen_list, block_shape_x, block_shape_y, max_kv_block_num):
    """Generate select_idx and select_num_idx for RFA test.

    select_idx: [total_q_s_block_num, num_heads, max_kv_block_num], int64
    select_num_idx: [total_q_s_block_num, num_heads], int64
    """
    total_q_s_block_num = sum((q_s + block_shape_x - 1) // block_shape_x for q_s in q_seqlen_list)
    select_idx = np.zeros((total_q_s_block_num, num_heads, max_kv_block_num), dtype=np.int64)
    select_num_idx = np.zeros((total_q_s_block_num, num_heads), dtype=np.int64)

    block_offset = 0
    for b in range(batch):
        kv_seqlen = int(kv_seqlen_list[b])
        kv_block_num = (kv_seqlen + block_shape_y - 1) // block_shape_y
        q_s_block_num = (int(q_seqlen_list[b]) + block_shape_x - 1) // block_shape_x
        num_sel = min(kv_block_num, max_kv_block_num)
        for n1 in range(num_heads):
            for q_s_idx in range(q_s_block_num):
                select_num_idx[block_offset + q_s_idx, n1] = num_sel
                select_idx[block_offset + q_s_idx, n1, :num_sel] = np.arange(num_sel, dtype=np.int64)
        block_offset += q_s_block_num

    return select_idx, select_num_idx


def _gen_inputs_bnsd(
    batch,
    num_heads,
    kv_heads,
    head_dim,
    max_q_seqlen,
    max_kv_seqlen,
    block_shape_x,
    block_shape_y,
    dtype=torch.float16,
    device="npu",
):
    q_seqlen_list = [max_q_seqlen] * batch
    kv_seqlen_list = [max_kv_seqlen] * batch
    max_kv_block_num = (max_kv_seqlen + block_shape_y - 1) // block_shape_y

    query = torch.randn(batch, num_heads, max_q_seqlen, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, kv_heads, max_kv_seqlen, head_dim, dtype=dtype, device=device)
    value = torch.randn(batch, kv_heads, max_kv_seqlen, head_dim, dtype=dtype, device=device)

    actual_seq_lengths = torch.tensor(q_seqlen_list, dtype=torch.int64, device=device)
    actual_seq_lengths_kv = torch.tensor(kv_seqlen_list, dtype=torch.int64, device=device)

    select_idx_np, select_num_idx_np = _gen_select_idx(
        batch, num_heads, q_seqlen_list, kv_seqlen_list, block_shape_x, block_shape_y, max_kv_block_num
    )
    select_idx = torch.from_numpy(select_idx_np).to(device)
    select_num_idx = torch.from_numpy(select_num_idx_np).to(device)

    block_shape = torch.tensor([block_shape_x, block_shape_y], dtype=torch.int64, device="cpu")

    return (query, key, value, select_idx, select_num_idx, block_shape, actual_seq_lengths, actual_seq_lengths_kv)


def _gen_inputs_tnd(
    batch,
    num_heads,
    kv_heads,
    head_dim,
    max_q_seqlen,
    max_kv_seqlen,
    block_shape_x,
    block_shape_y,
    dtype=torch.float16,
    device="npu",
):
    q_seqlen_list = [max_q_seqlen] * batch
    kv_seqlen_list = [max_kv_seqlen] * batch
    total_q = batch * max_q_seqlen
    total_kv = batch * max_kv_seqlen
    max_kv_block_num = (max_kv_seqlen + block_shape_y - 1) // block_shape_y

    query = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(total_kv, kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(total_kv, kv_heads, head_dim, dtype=dtype, device=device)

    actual_seq_lengths = torch.tensor(q_seqlen_list, dtype=torch.int64, device=device)
    actual_seq_lengths_kv = torch.tensor(kv_seqlen_list, dtype=torch.int64, device=device)

    select_idx_np, select_num_idx_np = _gen_select_idx(
        batch, num_heads, q_seqlen_list, kv_seqlen_list, block_shape_x, block_shape_y, max_kv_block_num
    )
    select_idx = torch.from_numpy(select_idx_np).to(device)
    select_num_idx = torch.from_numpy(select_num_idx_np).to(device)

    block_shape = torch.tensor([block_shape_x, block_shape_y], dtype=torch.int64, device="cpu")

    return (query, key, value, select_idx, select_num_idx, block_shape, actual_seq_lengths, actual_seq_lengths_kv)


@only_on_3510
@pytest.mark.parametrize(
    "batch,num_heads,kv_heads,head_dim,max_q_seqlen,max_kv_seqlen,block_shape_x,block_shape_y",
    [
        (1, 8, 2, 128, 128, 128, 128, 128),
        (2, 8, 1, 128, 128, 128, 128, 128),
        (1, 4, 1, 128, 128, 256, 128, 128),
    ],
)
def test_ascend950_rain_fusion_attention_bnsd(
    batch,
    num_heads,
    kv_heads,
    head_dim,
    max_q_seqlen,
    max_kv_seqlen,
    block_shape_x,
    block_shape_y,
):
    device = "npu"
    dtype = torch.float16

    inputs = _gen_inputs_bnsd(
        batch,
        num_heads,
        kv_heads,
        head_dim,
        max_q_seqlen,
        max_kv_seqlen,
        block_shape_x,
        block_shape_y,
        dtype=dtype,
        device=device,
    )
    (query, key, value, select_idx, select_num_idx, block_shape, actual_seq_lengths, actual_seq_lengths_kv) = inputs

    output = torch_catlass.ascend950_rain_fusion_attention(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        block_shape,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        input_layout="BNSD",
        num_heads=num_heads,
        num_key_value_heads=kv_heads,
        is_varied_len=0,
    )

    ref_output = _reference_rain_fusion_attention(
        query.cpu().numpy(),
        key.cpu().numpy(),
        value.cpu().numpy(),
        select_idx.cpu().numpy(),
        select_num_idx.cpu().numpy(),
        block_shape.numpy(),
        actual_seq_lengths.cpu().numpy(),
        actual_seq_lengths_kv.cpu().numpy(),
        "BNSD",
        num_heads,
        kv_heads,
        0,
    )

    # shape / dtype / device checks (skill checklist)
    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(output.cpu().float(), torch.from_numpy(ref_output).float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = "
        f"{(output.cpu().float() - torch.from_numpy(ref_output).float()).abs().max().item()}"
    )


@only_on_3510
@pytest.mark.parametrize(
    "batch,num_heads,kv_heads,head_dim,max_q_seqlen,max_kv_seqlen,block_shape_x,block_shape_y",
    [
        (1, 8, 1, 128, 128, 128, 128, 128),
        (2, 8, 2, 128, 128, 128, 128, 128),
        (1, 4, 1, 128, 128, 256, 128, 128),
    ],
)
def test_ascend950_rain_fusion_attention_tnd(
    batch,
    num_heads,
    kv_heads,
    head_dim,
    max_q_seqlen,
    max_kv_seqlen,
    block_shape_x,
    block_shape_y,
):
    device = "npu"
    dtype = torch.float16

    inputs = _gen_inputs_tnd(
        batch,
        num_heads,
        kv_heads,
        head_dim,
        max_q_seqlen,
        max_kv_seqlen,
        block_shape_x,
        block_shape_y,
        dtype=dtype,
        device=device,
    )
    (query, key, value, select_idx, select_num_idx, block_shape, actual_seq_lengths, actual_seq_lengths_kv) = inputs

    output = torch_catlass.ascend950_rain_fusion_attention(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        block_shape,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        input_layout="TND",
        num_heads=num_heads,
        num_key_value_heads=kv_heads,
        is_varied_len=0,
    )

    ref_output = _reference_rain_fusion_attention(
        query.cpu().numpy(),
        key.cpu().numpy(),
        value.cpu().numpy(),
        select_idx.cpu().numpy(),
        select_num_idx.cpu().numpy(),
        block_shape.numpy(),
        actual_seq_lengths.cpu().numpy(),
        actual_seq_lengths_kv.cpu().numpy(),
        "TND",
        num_heads,
        kv_heads,
        0,
    )

    # shape / dtype / device checks (skill checklist)
    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(output.cpu().float(), torch.from_numpy(ref_output).float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = "
        f"{(output.cpu().float() - torch.from_numpy(ref_output).float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
