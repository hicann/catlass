# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu

import torch_catlass
from common import only_on_2201

# Temporarily disable x_attention optest until the device issue (507033) is resolved.
pytestmark = pytest.mark.skip(reason="x_attention: device issue 507033; re-enable after recovery")


BLOCK_SIZE = 128
HEAD_DIM = 128


def _reference_x_attention(
    query,
    shared_key,
    shared_value,
    unshared_key,
    unshared_value,
    shared_block_table,
    unshared_block_table,
    shared_kv_lens,
    decode_step,
    batch,
    beam_size,
    num_heads,
    kv_heads,
):
    group_size = num_heads // kv_heads
    scale = HEAD_DIM**-0.5
    output = torch.empty_like(query, device="cpu")

    query = query.cpu().float().reshape(batch, beam_size, num_heads, HEAD_DIM)
    shared_key = shared_key.cpu().float()
    shared_value = shared_value.cpu().float()
    unshared_key = unshared_key.cpu().float()
    unshared_value = unshared_value.cpu().float()
    shared_kv_lens = shared_kv_lens.cpu()
    decode_count = int(decode_step.cpu().item())

    shared_paged = shared_block_table is not None
    if shared_paged:
        shared_block_table = shared_block_table.cpu()
    else:
        unshared_block_table = unshared_block_table.cpu()

    output = output.reshape(batch, beam_size, num_heads, HEAD_DIM)
    for batch_idx in range(batch):
        shared_len = int(shared_kv_lens[batch_idx].item())
        if shared_paged:
            block_count = (shared_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_ids = shared_block_table[batch_idx, :block_count].long()
            shared_key_batch = shared_key.index_select(0, block_ids).reshape(-1, kv_heads, HEAD_DIM)[:shared_len]
            shared_value_batch = shared_value.index_select(0, block_ids).reshape(-1, kv_heads, HEAD_DIM)[:shared_len]
            request_idx = batch_idx
        else:
            start = batch_idx * shared_len
            shared_key_batch = shared_key[start : start + shared_len]
            shared_value_batch = shared_value[start : start + shared_len]
            request_idx = int(unshared_block_table[batch_idx].item())

        if shared_paged:
            unshared_k = unshared_key[
                batch_idx * beam_size : (batch_idx + 1) * beam_size, :, :decode_count
            ]
            unshared_v = unshared_value[
                batch_idx * beam_size : (batch_idx + 1) * beam_size, :, :decode_count
            ]
        else:
            unshared_k = unshared_key[request_idx, :, :, :decode_count]
            unshared_v = unshared_value[request_idx, :, :, :decode_count]

        shared_k = shared_key_batch.repeat_interleave(group_size, dim=1)
        shared_v = shared_value_batch.repeat_interleave(group_size, dim=1)
        unshared_k = unshared_k.repeat_interleave(group_size, dim=1)
        unshared_v = unshared_v.repeat_interleave(group_size, dim=1)
        query_batch = query[batch_idx]

        shared_scores = torch.einsum("bhd,shd->bhs", query_batch, shared_k)
        unshared_scores = torch.einsum("bhd,bhtd->bht", query_batch, unshared_k)
        weights = torch.softmax(torch.cat((shared_scores, unshared_scores), dim=-1) * scale, dim=-1)
        shared_weights = weights[..., :shared_len]
        unshared_weights = weights[..., shared_len:]
        output[batch_idx] = torch.einsum("bhs,shd->bhd", shared_weights, shared_v)
        output[batch_idx] += torch.einsum("bht,bhtd->bhd", unshared_weights, unshared_v)

    return output.reshape(batch * beam_size, num_heads, HEAD_DIM)


def _make_inputs(cache_mode, dtype, seed, batch, beam_size, shared_kv_seq_len, decode_count):
    torch.manual_seed(seed)
    num_heads = 32
    kv_heads = 8
    max_decode_step = 3

    query = torch.empty(
        batch * beam_size, num_heads, HEAD_DIM, dtype=dtype, device="npu"
    ).uniform_(-0.5, 0.5)
    blocks_per_batch = (shared_kv_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * blocks_per_batch

    if cache_mode == 0:
        shared_shape = (num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM)
        unshared_shape = (batch * beam_size, kv_heads, max_decode_step, HEAD_DIM)
        shared_block_table = torch.arange(
            num_blocks, dtype=torch.int32, device="npu"
        ).reshape(batch, blocks_per_batch)
        unshared_block_table = None
    else:
        shared_shape = (batch * shared_kv_seq_len, kv_heads, HEAD_DIM)
        unshared_shape = (batch, beam_size, kv_heads, max_decode_step, HEAD_DIM)
        shared_block_table = None
        unshared_block_table = torch.arange(batch, dtype=torch.int32, device="npu")

    shared_key = torch.empty(shared_shape, dtype=dtype, device="npu").uniform_(-0.5, 0.5)
    shared_value = torch.empty(shared_shape, dtype=dtype, device="npu").uniform_(-0.5, 0.5)
    unshared_key = torch.empty(unshared_shape, dtype=dtype, device="npu").uniform_(-0.5, 0.5)
    unshared_value = torch.empty(unshared_shape, dtype=dtype, device="npu").uniform_(-0.5, 0.5)
    shared_kv_lens = torch.full(
        (batch,), shared_kv_seq_len, dtype=torch.int32, device="npu"
    )
    decode_step = torch.tensor([decode_count], dtype=torch.int32, device="npu")

    return {
        "query": query,
        "shared_key": shared_key,
        "shared_value": shared_value,
        "unshared_key": unshared_key,
        "unshared_value": unshared_value,
        "unshared_block_table": unshared_block_table,
        "shared_kv_lens": shared_kv_lens,
        "decode_step": decode_step,
        "shared_block_table": shared_block_table,
        "batch": batch,
        "beam_size": beam_size,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
    }


@only_on_2201
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "cache_mode,seed,batch,beam_size,shared_kv_seq_len,decode_count",
    [
        (0, 0, 1, 4, 1024, 2),
        (1, 1, 1, 16, 1024, 2),
        (1, 2, 1, 64, 2048, 3),
        (1, 3, 2, 128, 1024, 2),
    ],
)
def test_x_attention(cache_mode, dtype, seed, batch, beam_size, shared_kv_seq_len, decode_count):
    inputs = _make_inputs(
        cache_mode, dtype, seed, batch, beam_size, shared_kv_seq_len, decode_count
    )
    result = torch_catlass.x_attention(
        inputs["query"],
        inputs["shared_key"],
        inputs["shared_value"],
        inputs["unshared_key"],
        inputs["unshared_value"],
        inputs["unshared_block_table"],
        inputs["shared_kv_lens"],
        inputs["decode_step"],
        inputs["shared_block_table"],
    )
    torch.npu.synchronize()
    actual = result.cpu().float()
    expected = _reference_x_attention(
        inputs["query"],
        inputs["shared_key"],
        inputs["shared_value"],
        inputs["unshared_key"],
        inputs["unshared_value"],
        inputs["shared_block_table"],
        inputs["unshared_block_table"],
        inputs["shared_kv_lens"],
        inputs["decode_step"],
        inputs["batch"],
        inputs["beam_size"],
        inputs["num_heads"],
        inputs["kv_heads"],
    )

    assert result.shape == inputs["query"].shape
    assert result.dtype == dtype
    assert result.device.type == "npu"
    assert torch.allclose(actual, expected.float(), rtol=1e-3, atol=1e-3), (
        f"cache_mode={cache_mode}, dtype={dtype}, seed={seed}, "
        f"batch={batch}, shared_kv_seq_len={shared_kv_seq_len}, "
        f"max diff={(actual - expected.float()).abs().max().item()}"
    )


def _invoke_contract_case(inputs, scale=0.0):
    names = ("query", "shared_key", "shared_value", "unshared_key", "unshared_value",
             "unshared_block_table", "shared_kv_lens", "decode_step", "shared_block_table")
    return torch_catlass.x_attention(*(inputs[name] for name in names), scale)


@only_on_2201
@pytest.mark.parametrize("case,match", [
    ("zero_heads", "num_heads"),
    ("zero_kv_heads", "kv_heads"),
    ("empty_requests", "request_count"),
    ("scalar_table", "must be 2D"),
    ("scalar_decode", "shape"),
    ("matrix_lengths", "one value per batch"),
    ("dtype", "same dtype"),
    ("both_tables", "exactly one"),
    ("no_tables", "exactly one"),
    ("capacity", "max_decode_step"),
    ("group", "GQA"),
    ("noncontiguous", "contiguous"),
    ("cpu", "NPU tensor"),
])
def test_x_attention_can_implement(case, match):
    mode = 1 if case in ("empty_requests", "zero_kv_heads") else 0
    inputs = _make_inputs(mode, torch.float16, 0, 1, 2, 33, 1)
    if case == "zero_heads":
        inputs["query"] = inputs["query"][:, :0].contiguous()
    elif case == "zero_kv_heads":
        for name in ("shared_key", "shared_value"):
            inputs[name] = inputs[name][:, :0].contiguous()
    elif case == "empty_requests":
        for name in ("unshared_key", "unshared_value"):
            inputs[name] = inputs[name][:0]
    elif case == "scalar_table":
        inputs["shared_block_table"] = inputs["shared_block_table"].reshape(())
    elif case == "scalar_decode":
        inputs["decode_step"] = inputs["decode_step"].reshape(())
    elif case == "matrix_lengths":
        inputs["shared_kv_lens"] = inputs["shared_kv_lens"].reshape(1, 1)
    elif case == "dtype":
        inputs["shared_value"] = inputs["shared_value"].bfloat16()
    elif case == "both_tables":
        inputs["unshared_block_table"] = torch.zeros(1, dtype=torch.int32, device="npu")
    elif case == "no_tables":
        inputs["shared_block_table"] = None
    elif case == "capacity":
        for name in ("unshared_key", "unshared_value"):
            inputs[name] = torch.empty(2, 8, 257, 128, dtype=torch.float16, device="npu")
    elif case == "group":
        inputs["query"] = torch.empty(2, 8 * 129, 128, dtype=torch.float16, device="npu")
    elif case == "noncontiguous":
        inputs["query"] = inputs["query"].transpose(0, 1)
    elif case == "cpu":
        inputs["decode_step"] = inputs["decode_step"].cpu()
    with pytest.raises(RuntimeError, match=match):
        _ = _invoke_contract_case(inputs)


@only_on_2201
@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf"), 1e100])
def test_x_attention_invalid_scale(scale):
    inputs = _make_inputs(0, torch.float16, 0, 1, 2, 33, 1)
    with pytest.raises(RuntimeError, match="scale_value"):
        _ = _invoke_contract_case(inputs, scale)


@only_on_2201
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_x_attention_storage_offset(mode, dtype):
    inputs = _make_inputs(mode, dtype, 0, 1, 2, 33, 1)
    expected = _invoke_contract_case(inputs).cpu()
    for name, tensor in list(inputs.items()):
        if isinstance(tensor, torch.Tensor):
            # A contiguous slice may start after the beginning of its storage.
            padded = torch.zeros(tensor.numel() + 16, dtype=tensor.dtype, device=tensor.device)
            sliced = padded[16:].view(tensor.shape)
            sliced.copy_(tensor)
            assert sliced.is_contiguous() and sliced.storage_offset() == 16
            inputs[name] = sliced
    actual = _invoke_contract_case(inputs).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
