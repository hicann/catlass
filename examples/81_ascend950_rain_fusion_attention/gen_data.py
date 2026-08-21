#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import random
import sys
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

np.random.seed(1)
random.seed(1)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for i in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(q_seq, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    print(f"q_seqlen_list.shape: {len(q_seqlen_list)}, q_seqlen_list: {q_seqlen_list}")
    print(f"kv_seqlen_list.shape: {len(kv_seqlen_list)}, kv_seqlen_list: {kv_seqlen_list}")

    return q_seqlen_list, kv_seqlen_list


def gen_select_idx_pattern(
    q_seqlen_list: any,
    kv_seqlen_list: any,
    max_kv_seqlen_origin: int,
    block_shape: any,
    batch: int,
    num_heads: int,
    sparsity_ratio: float,
    is_varied_sparsity: int,
):
    block_x = block_shape[0]
    block_y = block_shape[1]
    max_kv_s_block_num = (max_kv_seqlen_origin + block_y - 1) // block_y
    total_q_s_block_num = sum((q_s + block_x - 1) // block_x for q_s in q_seqlen_list)

    select_idx = -np.ones((total_q_s_block_num, num_heads, max_kv_s_block_num), dtype=np.int64)
    select_num_idx = np.zeros((total_q_s_block_num, num_heads), dtype=np.int64)

    block_offset = 0
    total_valid_kv_s_blocks = 0
    for b in range(batch):
        q_s_block_num_aval = (q_seqlen_list[b] + block_x - 1) // block_x
        kv_s_block_num_aval = (kv_seqlen_list[b] + block_y - 1) // block_y
        selected_kv_s_block_num = max(1, int(kv_s_block_num_aval * sparsity_ratio))

        for q_blk_id in range(q_s_block_num_aval):
            for n1 in range(num_heads):
                rsvd_kv_s_block_num = selected_kv_s_block_num
                if is_varied_sparsity:
                    random.seed(b * num_heads * q_s_block_num_aval + n1 * q_s_block_num_aval + q_blk_id)
                    rsvd_kv_s_block_num = random.randint(1, selected_kv_s_block_num)
                cur_row_sparse_idx = sorted(
                    np.random.choice(kv_s_block_num_aval, size=rsvd_kv_s_block_num, replace=False)
                )
                select_idx[block_offset + q_blk_id, n1, 0:rsvd_kv_s_block_num] = cur_row_sparse_idx
                select_num_idx[block_offset + q_blk_id, n1] = rsvd_kv_s_block_num
            total_valid_kv_s_blocks += rsvd_kv_s_block_num
        block_offset += q_s_block_num_aval

    return select_idx, select_num_idx


class TestRainFusionAttention:
    @dataclass
    class AttentionInputs:
        query: any
        key: any
        value: any
        select_idx: any
        select_num_idx: any
        block_shape: any
        q_seqlen_list: any
        kv_seqlen_list: any
        shape_param: any  # GenDataParams

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        kv_seqlen_list: list
        max_kv_seqlen_origin: int
        num_heads: int
        kv_heads: int
        embedding_size: int
        block_shape: any
        dtype: any
        q_layout: str
        kv_layout: str
        inner_prec: int

    def base_tile_mm(self, left, right, mm_k_tile):
        res = None
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (left.shape[1] - mm1_k_loop_idx * mm_k_tile) if (mm1_k_loop_idx == mm_k_loop - 1) else mm_k_tile
            left_slice = left[:, mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k]
            right_slice = right[mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k, :]
            res_slice = np.matmul(left_slice, right_slice).astype(np.float32)
            res = res_slice if res is None else (res + res_slice)
        return res

    def online_softmax(self, qk_tile_res, gm, is_first_tile, interm_dtype_sm=np.float32):
        sim = qk_tile_res.astype(interm_dtype_sm)  # [s1_base, cur_kv_s_tile]
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first_tile:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        p = np.exp(sim_sub.astype(interm_dtype_sm))
        ll = np.sum(p, axis=-1, keepdims=True)
        return p, ll, dm, gm

    def rescale_o(self, lo, ll, dm, go, gl, is_first_tile, interm_dtype_re=np.float32):
        if is_first_tile:
            gl = ll
            go = lo
        else:
            dm = np.exp(dm)
            gl = gl * dm
            gl = gl + ll
            go = go * dm.astype(interm_dtype_re)
            go = go + lo
        return go, gl.astype(interm_dtype_re)

    def ref_flash_rain_fusion_attention(self, query, key, value, softmax_scale, inner_prec, attn_mask=None):
        cur_kv_s_gathered = key.shape[1]
        kv_s_base_tile = 256
        interm_dtype_sm = query.dtype
        interm_dtype_re = np.float32
        softmax_scale = np.asarray(softmax_scale, dtype=query.dtype)
        gl = None
        go = None

        for kv_s_start in range(0, cur_kv_s_gathered, kv_s_base_tile):
            cur_kv_s_tile = (
                (cur_kv_s_gathered - kv_s_start)
                if (kv_s_start + kv_s_base_tile > cur_kv_s_gathered)
                else kv_s_base_tile
            )
            key_tile = key[:, kv_s_start : kv_s_start + cur_kv_s_tile]  # [D, cur_kv_s_tile]
            value_tile = value[kv_s_start : kv_s_start + cur_kv_s_tile, :]  # [cur_kv_s_tile, D]

            # atten_mask暂不支持
            # q*k^t
            mm1_k_tile = 128
            qk_tile_res = self.base_tile_mm(query, key_tile, mm1_k_tile).astype(interm_dtype_sm)
            qk_tile_res = qk_tile_res * softmax_scale

            # online softmax
            if kv_s_start == 0:
                gm = None
            p, ll, dm, gm = self.online_softmax(qk_tile_res, gm, kv_s_start == 0, interm_dtype_sm)
            p = p.astype(query.dtype)  # [s1_base, cur_kv_s_tile]

            # p*v
            mm2_k_tile = 128
            lo = self.base_tile_mm(p, value_tile, mm2_k_tile).astype(interm_dtype_re)
            # rescale O
            go, gl = self.rescale_o(lo, ll, dm, go, gl, kv_s_start == 0, interm_dtype_re)

        go = go / gl
        go = go.astype(query.dtype)  # [S1,D]
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32)
        return go, lse

    def ref_attention(self, query, key, value, softmax_scale, inner_prec, attn_mask=None):
        s = np.matmul(query.astype(np.float32), key.astype(np.float32)).astype(np.float32)
        s = s * np.float32(softmax_scale)

        row_max = np.max(s, axis=-1, keepdims=True)
        s_sub = s - row_max
        s_sub = np.exp(s_sub)
        row_sum = np.sum(s_sub, axis=-1, keepdims=True)
        p = s_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

        o = np.matmul(p.astype(np.float32), value.astype(np.float32)).astype(np.float32)

        return o, lse

    def compute_output(self, attention_inputs: AttentionInputs, scale: float):
        attn_out_bm = np.zeros_like(attention_inputs.query)
        attn_out_gt = np.zeros_like(attention_inputs.query).astype(np.float32)

        query = attention_inputs.query
        key = attention_inputs.key
        value = attention_inputs.value
        select_idx = attention_inputs.select_idx
        select_num_idx = attention_inputs.select_num_idx
        block_shape = attention_inputs.block_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        q_layout = attention_inputs.shape_param.q_layout
        kv_layout = attention_inputs.shape_param.kv_layout
        inner_prec = attention_inputs.shape_param.inner_prec

        softmax_scale = scale
        batch = len(q_seqlen_list)
        group_size = int(num_heads / kv_heads)

        block_offset = 0
        for b in range(batch):
            q_s_block_num_aval = (q_seqlen_list[b] + block_shape[0] - 1) // block_shape[0]
            kv_s_block_num_aval = (kv_seqlen_list[b] + block_shape[1] - 1) // block_shape[1]

            for q_s_blk_idx in range(q_s_block_num_aval):
                cur_q_s_blk_size = (
                    (q_seqlen_list[b] - q_s_blk_idx * block_shape[0])
                    if q_s_blk_idx == (q_s_block_num_aval - 1)
                    else block_shape[0]
                )
                for n1 in range(num_heads):
                    n2 = int(n1 // group_size)
                    q_blk = None
                    if q_layout == "TND":
                        t_offset = sum(q_seqlen_list[:b]) + q_s_blk_idx * block_shape[0]
                        q_blk = query[t_offset : t_offset + cur_q_s_blk_size, n1, :]  # [cur_q_s_blk_size, D]
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_shape[0]
                        q_blk = query[b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :]  # [cur_q_s_blk_size, D]

                    key_gathered = None
                    value_gathered = None
                    select_num = select_num_idx[block_offset + q_s_blk_idx, n1]
                    for count, idx in enumerate(range(select_num)):
                        kv_s_blk_idx = select_idx[block_offset + q_s_blk_idx, n1, idx]
                        cur_kv_s_blk_size = (
                            (kv_seqlen_list[b] - kv_s_blk_idx * block_shape[1])
                            if kv_s_blk_idx == (kv_s_block_num_aval - 1)
                            else block_shape[1]
                        )
                        cur_k_blk = None
                        cur_v_blk = None
                        if kv_layout == "TND":
                            t_offset = sum(kv_seqlen_list[:b]) + kv_s_blk_idx * block_shape[1]
                            cur_k_blk = key[t_offset : t_offset + cur_kv_s_blk_size, n2, :]
                            cur_v_blk = value[t_offset : t_offset + cur_kv_s_blk_size, n2, :]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))
                        elif kv_layout == "BNSD":
                            s2_offset = kv_s_blk_idx * block_shape[1]
                            cur_k_blk = key[
                                b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                            ]  # [cur_kv_s_blk_size, D]
                            cur_v_blk = value[b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))  # [D, cur_kv_s_blk_size]

                        if count == 0:
                            key_gathered = cur_k_blk
                            value_gathered = cur_v_blk
                        else:
                            key_gathered = np.concatenate([key_gathered, cur_k_blk], axis=1)  # [D, kv_s_valid]
                            value_gathered = np.concatenate([value_gathered, cur_v_blk], axis=0)  # [kv_s_valid, D]

                    attn_out_bm_slice, _lse_bm_slice = self.ref_flash_rain_fusion_attention(
                        q_blk, key_gathered, value_gathered, softmax_scale, inner_prec
                    )
                    attn_out_gt_slice, _lse_gt_slice = self.ref_attention(
                        q_blk, key_gathered, value_gathered, softmax_scale, inner_prec
                    )

                    if q_layout == "TND":
                        t_offset = sum(q_seqlen_list[:b]) + q_s_blk_idx * block_shape[0]
                        attn_out_bm[t_offset : t_offset + cur_q_s_blk_size, n1, :] = attn_out_bm_slice
                        attn_out_gt[t_offset : t_offset + cur_q_s_blk_size, n1, :] = attn_out_gt_slice
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_shape[0]
                        attn_out_bm[b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :] = attn_out_bm_slice
                        attn_out_gt[b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :] = attn_out_gt_slice

            block_offset += q_s_block_num_aval

        return attn_out_bm, attn_out_gt

    def calc_data(self, gen_data_params: GenDataParams):
        os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

        q_seqlen_list = gen_data_params.q_seqlen_list
        kv_seqlen_list = gen_data_params.kv_seqlen_list
        max_kv_seqlen_origin = gen_data_params.max_kv_seqlen_origin
        num_heads = gen_data_params.num_heads
        kv_heads = gen_data_params.kv_heads
        embedding_size = gen_data_params.embedding_size
        block_shape = gen_data_params.block_shape
        dtype = gen_data_params.dtype
        q_layout = gen_data_params.q_layout
        kv_layout = gen_data_params.kv_layout

        batch = len(q_seqlen_list)
        num_tokens = sum(q_seqlen_list)
        num_kv_tokens = sum(kv_seqlen_list)
        max_q_seqlen = max(q_seqlen_list)
        max_kv_seqlen = max(kv_seqlen_list)

        scale = 1.0 / (embedding_size**0.5)

        q_min_value, q_max_value = -1.0, 1.0
        kv_min_value, kv_max_value = -1.0, 1.0

        if q_layout == "TND":
            q_shape = (num_tokens, num_heads, embedding_size)
        elif q_layout == "BNSD":
            q_shape = (batch, num_heads, max_q_seqlen, embedding_size)
        query = np.random.uniform(q_min_value, q_max_value, size=q_shape).astype(dtype)

        if kv_layout == "TND":
            kv_shape = (num_kv_tokens, kv_heads, embedding_size)
        elif kv_layout == "BNSD":
            kv_shape = (batch, kv_heads, max_kv_seqlen, embedding_size)
        key = np.random.uniform(kv_min_value, kv_max_value, size=kv_shape).astype(dtype)
        value = np.random.uniform(kv_min_value, kv_max_value, size=kv_shape).astype(dtype)

        sparsity_ratio = 0.5
        is_varied_sparsity = 0
        select_idx, select_num_idx = gen_select_idx_pattern(
            q_seqlen_list,
            kv_seqlen_list,
            max_kv_seqlen_origin,
            block_shape,
            batch,
            num_heads,
            sparsity_ratio,
            is_varied_sparsity,
        )

        attention_inputs = self.AttentionInputs(
            query,
            key,
            value,
            select_idx,
            select_num_idx,
            block_shape,
            q_seqlen_list,
            kv_seqlen_list,
            gen_data_params,
        )

        attn_out_bm, attn_out_gt = self.compute_output(attention_inputs, scale)

        np.array(num_tokens).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
        np.array(num_kv_tokens).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens.bin"))

        query.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
        key.tofile(os.path.join(WORKSPACE, "data", "k.bin"))
        value.tofile(os.path.join(WORKSPACE, "data", "v.bin"))

        select_idx.tofile(os.path.join(WORKSPACE, "data", "select_idx.bin"))
        select_num_idx.tofile(os.path.join(WORKSPACE, "data", "select_num_idx.bin"))

        np.array([select_idx.shape[0]], dtype=np.int32).tofile(
            os.path.join(WORKSPACE, "data", "total_qs_block_num.bin")
        )

        np.array(q_seqlen_list).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "q_seqlen_list.bin"))
        np.array(kv_seqlen_list).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "kv_seqlen_list.bin"))

        attn_out_bm.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden.bin"))
        attn_out_gt.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu.bin"))


if __name__ == "__main__":
    # 参数顺序: batch, qSeqlen, kvSeqlen, numHeads, kvHeads, headSize,
    #          blockShapeX, blockShapeY, dtype, qInputLayout, kvInputLayout, isVariedLen
    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    head_size = int(sys.argv[6])
    block_shape_x = int(sys.argv[7])
    block_shape_y = int(sys.argv[8])
    str_dtype = str(sys.argv[9])
    print("str_dtype: ", str_dtype)
    dtype = np.float16
    if str_dtype == "half":  # "half", "bf16"
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        print("[ERROR] dtype must be half or bf16")
        sys.exit()
    q_input_layout = str(sys.argv[10])  # 0:TND, 1:BNSD
    kv_input_layout = str(sys.argv[11])  # 0:TND, 1:BNSD
    is_varied_len = int(sys.argv[12])  # 0: not varied, 1: varied
    if q_input_layout != "TND" and q_input_layout != "BNSD":
        print("[ERROR] q_input_layout must be TND or BNSD")
        sys.exit()
    if kv_input_layout != "TND" and kv_input_layout != "BNSD":
        print("[ERROR] kv_input_layout must be TND or BNSD")
        sys.exit()
    if q_input_layout != kv_input_layout:
        print("[ERROR] q_input_layout and kv_input_layout must be the same")
        sys.exit()

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, batch)

    testObj = TestRainFusionAttention()
    gen_data_params = testObj.GenDataParams(
        q_seqlen_list,
        kv_seqlen_list,
        kv_seqlen,  # max_kv_seqlen_origin
        num_head,
        kv_heads,
        head_size,
        (block_shape_x, block_shape_y),
        dtype,
        q_input_layout,
        kv_input_layout,
        inner_prec=0,
    )
    testObj.calc_data(gen_data_params)
