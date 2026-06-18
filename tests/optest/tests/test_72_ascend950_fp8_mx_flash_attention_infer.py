# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import math
import re

import numpy as np
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
    reason="example 72_ascend950_fp8_mx_flash_attention_infer requires Ascend 950 NPU",
)

MX_BASEK_FACTOR = 64


def _fp8_e4m3_quantize_dequantize(data_fp32, axis, block_size=32):
    M, N = data_fp32.shape
    data = torch.from_numpy(data_fp32)

    if axis == 0:
        num_blocks = (M + block_size - 1) // block_size
        quantized = torch.zeros_like(data)
        scales = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)
        for bi in range(num_blocks):
            sr = bi * block_size
            er = min(sr + block_size, M)
            block = data[sr:er, :]
            for c in range(N):
                col = block[:, c]
                max_abs = torch.max(torch.abs(col)).item()
                if max_abs < 1e-12:
                    scale = 1.0
                else:
                    exp = int(math.floor(math.log2(max_abs))) - 8
                    exp = max(-128, min(exp, 127))
                    scale = 2.0 ** exp
                scaled = col / scale
                scaled = torch.clamp(scaled, -448.0, 448.0)
                quantized[sr:er, c] = scaled.to(torch.float8_e4m3fn).to(torch.float32)
                scales[bi, c] = scale
    else:
        num_blocks = (N + block_size - 1) // block_size
        quantized = torch.zeros_like(data)
        scales = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)
        for bi in range(num_blocks):
            sc = bi * block_size
            ec = min(sc + block_size, N)
            block = data[:, sc:ec]
            for r in range(M):
                row = block[r, :]
                max_abs = torch.max(torch.abs(row)).item()
                if max_abs < 1e-12:
                    scale = 1.0
                else:
                    exp = int(math.floor(math.log2(max_abs))) - 8
                    exp = max(-128, min(exp, 127))
                    scale = 2.0 ** exp
                scaled = row / scale
                scaled = torch.clamp(scaled, -448.0, 448.0)
                quantized[r, sc:ec] = scaled.to(torch.float8_e4m3fn).to(torch.float32)
                scales[r, bi] = scale

    data_fp8 = torch.tensor(quantized.to(torch.float8_e4m3fn).flatten().untyped_storage(), dtype=torch.int8)
    scale_fp8 = torch.tensor(scales.to(torch.float8_e8m0fnu).flatten().untyped_storage(), dtype=torch.int8)

    if axis == 0:
        scales_expanded = scales.repeat_interleave(block_size, dim=0)[:M, :]
    else:
        scales_expanded = scales.repeat_interleave(block_size, dim=1)[:, :N]
    data_dequant = quantized * scales_expanded

    return data_fp8, scale_fp8, data_dequant


def _ref_mxfp8_flash_attention(
    query_fp32,
    key_cache_fp32,
    value_cache_fp32,
    q_seqlen_list,
    kv_seqlen_list,
    block_table,
    mask,
    num_heads,
    kv_heads,
    head_dim,
    block_size,
    use_p_scale,
):
    batch = len(q_seqlen_list)
    outputs = []
    scale = 1.0 / (head_dim ** 0.5)
    post_mask_factor = -3e38

    cu_q = 0
    for i in range(batch):
        q_len = int(q_seqlen_list[i])
        kv_len = int(kv_seqlen_list[i])
        q = query_fp32[cu_q : cu_q + q_len]
        q = q.reshape(q_len, num_heads, head_dim)

        keys = []
        values = []
        table = block_table[i]
        for j in range(kv_len):
            block_number = int(table[j // block_size])
            block_offset = j % block_size
            keys.append(key_cache_fp32[block_number, block_offset])
            values.append(value_cache_fp32[block_number, block_offset])
        key = np.stack(keys, axis=0)
        value = np.stack(values, axis=0)

        batch_mask = None
        if mask is not None:
            batch_mask = mask[cu_q : cu_q + q_len, :kv_len]

        q_t = q.transpose(1, 0, 2)
        k_t = key.transpose(1, 2, 0)
        v_t = value.transpose(1, 0, 2)

        out = np.zeros_like(q_t)
        group_num = num_heads // kv_heads

        for n_i in range(num_heads):
            kv_head_i = n_i // group_num
            num_s1_block = (q_len + 127) // 128
            for s1_i in range(num_s1_block):
                s1_start = s1_i * 128
                s1_end = min((s1_i + 1) * 128, q_len)
                s1_len = s1_end - s1_start
                max_i = np.ones((s1_len, 1)) * (-3e38)
                sum_i = np.zeros((s1_len, 1))
                o_i = np.zeros((s1_len, head_dim))

                num_s2_block = (kv_len + 127) // 128
                for s2_i in range(num_s2_block):
                    s2_start = s2_i * 128
                    s2_end = min((s2_i + 1) * 128, kv_len)

                    sblock = np.matmul(
                        q_t[n_i, s1_start:s1_end, :].astype(np.float64),
                        k_t[kv_head_i, :, s2_start:s2_end].astype(np.float64),
                    ).astype(np.float32)
                    sblock *= scale

                    if batch_mask is not None:
                        sblock = sblock + batch_mask[s1_start:s1_end, s2_start:s2_end].astype(np.float32) * post_mask_factor

                    row_max = np.max(sblock, axis=-1, keepdims=True)
                    max_i_new = np.maximum(row_max, max_i)
                    pblock = np.exp(sblock - max_i_new)

                    if use_p_scale:
                        pblock *= 0.5

                    sum_i = sum_i * np.exp(max_i - max_i_new) + np.sum(pblock, axis=-1, keepdims=True)

                    pblock_fp8 = torch.from_numpy(pblock.astype(np.float32))
                    pblock_fp8 = pblock_fp8.to(torch.float8_e4m3fn).to(torch.float32)
                    pblock = pblock_fp8.numpy()

                    o_temp = np.matmul(
                        pblock.astype(np.float64),
                        v_t[kv_head_i, s2_start:s2_end, :].astype(np.float64),
                    ).astype(np.float32)
                    o_i = o_i * np.exp(max_i - max_i_new) + o_temp
                    max_i = max_i_new

                o_i = o_i / sum_i
                out[n_i, s1_start:s1_end, :] = o_i

        out = out.transpose(1, 0, 2)
        outputs.append(out.reshape(q_len, num_heads, head_dim))
        cu_q += q_len

    return np.concatenate(outputs, axis=0)


def _make_mxfp8_test_data(batch, q_seqlen, kv_seqlen, num_heads, kv_heads, head_dim, block_size, mask_type, use_p_scale, dtype):
    q_seqlen_list = [q_seqlen] * batch
    kv_seqlen_list = [kv_seqlen] * batch
    num_tokens = sum(q_seqlen_list)
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    max_num_blocks_per_seq = (max_kv_seqlen + block_size - 1) // block_size

    np_dtype = np.float16 if dtype == torch.float16 else np.float32

    query_fp32 = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_dim)).astype(np.float32)
    key_cache_fp32 = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_dim)).astype(np.float32)
    value_cache_fp32 = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_dim)).astype(np.float32)

    q_2d = query_fp32.reshape(num_tokens * num_heads, head_dim)
    q_fp8_raw, q_scale_raw, q_dequant = _fp8_e4m3_quantize_dequantize(q_2d, axis=1)
    query_dequant = q_dequant.reshape(num_tokens, num_heads, head_dim).numpy().astype(np_dtype)

    k_2d = key_cache_fp32.reshape(num_blocks * block_size, kv_heads * head_dim)
    k_fp8_raw, k_scale_raw, k_dequant = _fp8_e4m3_quantize_dequantize(k_2d, axis=1)
    key_dequant = k_dequant.reshape(num_blocks, block_size, kv_heads, head_dim).numpy().astype(np_dtype)

    v_2d = value_cache_fp32.reshape(num_blocks * block_size, kv_heads * head_dim)
    v_fp8_raw, v_scale_raw, v_dequant = _fp8_e4m3_quantize_dequantize(v_2d, axis=0)
    value_dequant = v_dequant.reshape(num_blocks, block_size, kv_heads, head_dim).numpy().astype(np_dtype)

    block_table = []
    for i in range(batch):
        block_table.append([max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)])
    block_table_np = np.array(block_table, dtype=np.int32)

    mask_np = None
    atten_mask_np = None
    if mask_type > 0:
        mask_np = np.zeros((num_tokens, max_kv_seqlen), dtype=np_dtype)
        atten_mask_np = np.zeros((num_tokens, max_kv_seqlen), dtype=np.uint8)
        cu_q = 0
        for i in range(batch):
            q_len = q_seqlen_list[i]
            k_len = kv_seqlen_list[i]
            max_seq_len = max(q_len, k_len)
            tri = np.triu(np.ones((max_seq_len, max_seq_len)), 1).astype(np_dtype)
            mask_np[cu_q : cu_q + q_len, 0 : k_len] = tri[0 : q_len, 0 : k_len]
            atten_mask_np[cu_q : cu_q + q_len, 0 : k_len] = tri[0 : q_len, 0 : k_len].astype(np.uint8)
            cu_q += q_len

    p_scale_np = None
    if use_p_scale:
        p_scale_np = np.array([0.5], dtype=np.float32)

    q_fp8 = torch.from_numpy(np.frombuffer(q_fp8_raw.numpy().tobytes(), dtype=np.int8)).reshape(num_tokens, num_heads, head_dim)
    k_fp8 = torch.from_numpy(np.frombuffer(k_fp8_raw.numpy().tobytes(), dtype=np.int8)).reshape(num_blocks, block_size, kv_heads, head_dim)
    v_fp8 = torch.from_numpy(np.frombuffer(v_fp8_raw.numpy().tobytes(), dtype=np.int8)).reshape(num_blocks, block_size, kv_heads, head_dim)

    q_scale = torch.from_numpy(np.frombuffer(q_scale_raw.numpy().tobytes(), dtype=np.int8)).reshape(num_tokens, num_heads, head_dim // MX_BASEK_FACTOR, 2)
    k_scale = torch.from_numpy(np.frombuffer(k_scale_raw.numpy().tobytes(), dtype=np.int8)).reshape(num_blocks, block_size, kv_heads, head_dim // MX_BASEK_FACTOR, 2)
    v_scale = torch.from_numpy(np.frombuffer(v_scale_raw.numpy().tobytes(), dtype=np.int8)).reshape(num_blocks, block_size // MX_BASEK_FACTOR, 2, kv_heads, head_dim).permute(0, 1, 3, 4, 2)

    return {
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "v_fp8": v_fp8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "p_scale": p_scale_np,
        "query_dequant": query_dequant,
        "key_dequant": key_dequant,
        "value_dequant": value_dequant,
        "q_seqlen_list": q_seqlen_list,
        "kv_seqlen_list": kv_seqlen_list,
        "block_table": block_table_np,
        "mask": mask_np,
        "atten_mask": atten_mask_np,
        "num_tokens": num_tokens,
        "num_blocks": num_blocks,
        "max_kv_seqlen": max_kv_seqlen,
    }


def _to_npu(t):
    if isinstance(t, np.ndarray):
        return torch.from_numpy(t).to("npu")
    return t.to("npu")


class TestAscend950MxFp8FlashAttentionInfer:

    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("mask_type", [0, 1])
    @pytest.mark.parametrize("use_p_scale", [False, True])
    @pytest.mark.parametrize("q_seqlen", [128])
    @pytest.mark.parametrize("kv_seqlen", [256])
    @pytest.mark.parametrize("num_heads", [8])
    def test_mxfp8_flash_attention_infer_paged(self, batch, mask_type, use_p_scale, q_seqlen, kv_seqlen, num_heads):
        torch.manual_seed(1)
        np.random.seed(1)

        kv_heads = 1
        head_dim = 128
        block_size = 128
        dtype = torch.float16

        data = _make_mxfp8_test_data(
            batch, q_seqlen, kv_seqlen, num_heads, kv_heads,
            head_dim, block_size, mask_type, use_p_scale, dtype,
        )

        q_fp8_npu = _to_npu(data["q_fp8"])
        k_fp8_npu = _to_npu(data["k_fp8"])
        v_fp8_npu = _to_npu(data["v_fp8"])
        q_scale_npu = _to_npu(data["q_scale"])
        k_scale_npu = _to_npu(data["k_scale"])
        v_scale_npu = _to_npu(data["v_scale"])
        block_table_npu = _to_npu(data["block_table"])
        actual_seq_lengths = _to_npu(np.array(data["q_seqlen_list"], dtype=np.int64))
        actual_seq_lengths_kv = _to_npu(np.array(data["kv_seqlen_list"], dtype=np.int64))

        atten_mask_npu = _to_npu(data["atten_mask"]) if data["atten_mask"] is not None else torch.zeros(1, dtype=torch.uint8, device="npu")

        p_scale_npu = None
        if use_p_scale:
            p_scale_npu = _to_npu(data["p_scale"])

        result = torch_catlass.ascend950_fp8_mx_flash_attention_infer(
            q_fp8_npu,
            k_fp8_npu,
            v_fp8_npu,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            atten_mask_npu,
            block_table_npu,
            q_scale_npu,
            k_scale_npu,
            v_scale_npu,
            p_scale_npu,
            "TND",
            num_heads,
            kv_heads,
            mask_type,
        )

        expected = _ref_mxfp8_flash_attention(
            data["query_dequant"],
            data["key_dequant"],
            data["value_dequant"],
            data["q_seqlen_list"],
            data["kv_seqlen_list"],
            data["block_table"],
            data["mask"],
            num_heads,
            kv_heads,
            head_dim,
            block_size,
            use_p_scale,
        )

        num_tokens = data["num_tokens"]
        assert result.shape == (num_tokens, num_heads, head_dim)
        assert result.dtype == dtype
        assert result.device.type == "npu"

        expected_t = torch.from_numpy(expected.astype(np.float16))

        rtol = 1e-2
        atol = 1e-2
        result_cpu = result.cpu().float()
        expected_cpu = expected_t.float()
        max_diff = (result_cpu - expected_cpu).abs().max().item()
        assert torch.allclose(result_cpu, expected_cpu, rtol=rtol, atol=atol), (
            f"Results not close: max diff = {max_diff}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
