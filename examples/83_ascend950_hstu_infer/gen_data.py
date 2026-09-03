#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass

def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    accu_q_seqlen = 0
    accu_kv_seqlen = 0
    for i in range(batch + 1):
        # kv_seq >= q_seq
        if i == 0:
            q_seq = 0
            kv_seq = 0
        else:
            if is_varied_len == 0:
                q_seq = max_q_seqlen
                kv_seq = max_kv_seqlen
            else:
                q_seq = random.randint(1, max_q_seqlen)
                kv_seq = random.randint(q_seq, max_q_seqlen + max_kv_seqlen)
        accu_q_seqlen += q_seq
        accu_kv_seqlen += kv_seq

        q_seqlen_list.append(accu_q_seqlen)
        kv_seqlen_list.append(accu_kv_seqlen)

    return q_seqlen_list, kv_seqlen_list

class TestHstuInfer():

    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        q_seqlen_list: any
        k_seqlen_list: any
        mask_type: any
        param: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        silu_scale: float
        layout: str
        dtype: any
        data_path: str
        mask_type: int
        paged_block_size: int = 0

    @classmethod
    def silu(cls, score, silu_scale):
        return silu_scale * score * (1 / (1 + np.exp(-score)))

    def silu_scale(x: np.ndarray, scale: float) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return scale * x * sigmoid

    def create_causal_mult_mask(self, n: int, m: int):
        """
        返回 shape (n,m) 0/1 矩阵，乘法掩码
        mult_mask[i,j] = 1 if j <= i else 0
        """
        i = np.arange(n)[:, np.newaxis]  # (n,1)
        j = np.arange(m)[np.newaxis, :]  # (1,m)
        mult_mask = (j <= i).astype(np.float32)
        return mult_mask

    def ref_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            silu_scale: float,
            enable_causal_mask: int = 0,
    ):
        # Q * K.T
        query = query
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))

        sim_high = np.matmul(query.astype(np.float32), key.astype(np.float32))  # (head_num, q_seqlen, k_seqlen)

        # silu
        sim_high = sim_high.astype(np.float32)
        p_high = self.silu(sim_high, silu_scale)

        if enable_causal_mask == 1:
            # causal mask
            causal_mask = self.create_causal_mult_mask(p_high.shape[1], p_high.shape[2])

            for i in range(p_high.shape[0]):
                p_high[i, :, :] = p_high[i, :, :] * causal_mask

        # P * V
        value = np.transpose(value, (1, 0, 2))

        p = p_high.astype(query.dtype)

        out = np.matmul(p.astype(np.float32), value.astype(np.float32))
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(query.dtype)

        p_high = p_high.astype(np.float32)
        out_high = np.matmul(p_high, value.astype(np.float32))
        out_high = np.transpose(out_high, (1, 0, 2))

        # print(f"O: {out.reshape(-1)}")
        return out, out_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, true_out) -> None:
        num_heads = attention_inputs.param.num_heads
        kv_heads = attention_inputs.param.kv_heads
        head_size_qk = attention_inputs.param.head_size
        head_size_vo = attention_inputs.param.head_size
        batch = len(attention_inputs.param.q_seqlen_list) - 1
        silu_scale = attention_inputs.param.silu_scale
        enable_causal_mask = attention_inputs.param.mask_type
        print(f"enable_causal_mask: {enable_causal_mask}")
        print(f"attention_inputs.q_seqlen_list: {attention_inputs.q_seqlen_list}")
        print(f"attention_inputs.k_seqlen_list: {attention_inputs.k_seqlen_list}")

        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i + 1] - attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i + 1] - attention_inputs.k_seqlen_list[i])
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            keys = attention_inputs.key_cache[kv_seqlen_now:(kv_seqlen_now + k_seqlen), :, :]
            values = attention_inputs.value_cache[kv_seqlen_now:(kv_seqlen_now + k_seqlen), :, :]

            out, out_high = self.ref_attention(q, keys, values, silu_scale, enable_causal_mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = gen_data_params.q_seqlen_list[-1]
        num_kv_tokens = gen_data_params.k_seqlen_list[-1]
        batch_size = len(gen_data_params.q_seqlen_list) - 1

        debug = 0

        if debug == 1:
            # 列维度生成 1,2,1,2... 序列
            col_arr = np.tile([1, 2], (head_size_qk + 1) // 2)[:head_size_qk]
            # 扩维广播填充整个数组
            query = np.broadcast_to(col_arr[None, None, :],
                                    shape=(num_tokens, gen_data_params.num_heads, head_size_qk)
                                ).astype(gen_data_params.dtype)

            col_arr = np.tile([-1, 1], (head_size_qk + 1) // 2)[:head_size_qk]
            key_cache = np.broadcast_to(col_arr[None, None, :],
                                                shape=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)
                                            ).astype(gen_data_params.dtype)

            col_arr = np.tile([-2, 3], (head_size_vo + 1) // 2)[:head_size_vo]
            value_cache = np.broadcast_to(col_arr[None, None, :],
                                                shape=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)
                                            ).astype(gen_data_params.dtype)

            query = np.ones(shape=(num_tokens, gen_data_params.num_heads, head_size_qk), dtype=gen_data_params.dtype)
            key_cache = np.ones(shape=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk), dtype=gen_data_params.dtype)
            value_cache = np.ones(shape=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo), dtype=gen_data_params.dtype)

        else :
            query = np.random.uniform(q_min_range, q_max_range,
                size=(num_tokens, gen_data_params.num_heads, head_size_qk)).astype(gen_data_params.dtype)
            key_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
            value_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = np.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = np.zeros(shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(query, key_cache, value_cache,
            gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, gen_data_params.mask_type, gen_data_params)

        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            ref_output,
            true_out,
        )

        if gen_data_params.layout == "NTD":
            query = np.transpose(query, (1, 0, 2))
            ref_output = np.transpose(ref_output, (1, 0, 2))
            true_out = np.transpose(true_out, (1, 0, 2))

        if gen_data_params.paged_block_size == 0:
            if gen_data_params.layout == "NTD":
                key_cache = np.transpose(key_cache, (1, 0, 2))
                value_cache = np.transpose(value_cache, (1, 0, 2))
        else:
            paged_block_size = gen_data_params.paged_block_size
            # 直接差分得到numpy数组（假设k_seqlen_list本身是list/np数组）
            k_seqlen_arr = np.array(gen_data_params.k_seqlen_list, dtype=np.int32)
            kv_seq_len_arr = k_seqlen_arr[1:batch+1] - k_seqlen_arr[:batch]

            max_kv_seq_len = kv_seq_len_arr.max()
            kv_block_num_arr = (kv_seq_len_arr + paged_block_size - 1) // paged_block_size
            max_kv_block_num = kv_block_num_arr.max()
            total_kv_block_num = kv_block_num_arr.sum()

            key_cache_paged = np.zeros(shape=(total_kv_block_num, paged_block_size, gen_data_params.kv_heads, head_size_qk), dtype=gen_data_params.dtype)
            value_cache_paged = np.zeros(shape=(total_kv_block_num, paged_block_size, gen_data_params.kv_heads, head_size_vo), dtype=gen_data_params.dtype)

            block_table = np.zeros(shape=(batch, max_kv_block_num), dtype=np.int32)

            blocks = 0
            kv_seq_sum = 0
            for i in range(batch):
                block_num = kv_block_num_arr[i]
                for j in range(block_num):
                    len_tmp = min(paged_block_size, kv_seq_len_arr[i] - j * paged_block_size)

                    block_idx = blocks + j
                    src_kv_seq_start = kv_seq_sum + j * paged_block_size
                    key_cache_paged[block_idx, :len_tmp, :, :] = key_cache[src_kv_seq_start:src_kv_seq_start + len_tmp, :, :]
                    value_cache_paged[block_idx, :len_tmp, :, :] = value_cache[src_kv_seq_start:src_kv_seq_start + len_tmp, :, :]

                    block_table[i, j] = block_idx

                blocks = blocks + block_num
                kv_seq_sum = kv_seq_sum + kv_seq_len_arr[i]

        np.array(num_tokens).astype(np.int32).tofile(os.path.join(gen_data_params.data_path, "q_ntokens.bin"))
        np.array(num_kv_tokens).astype(np.int32).tofile(os.path.join(gen_data_params.data_path, "kv_ntokens.bin"))
        query.tofile(os.path.join(gen_data_params.data_path, "q.bin"))

        if gen_data_params.paged_block_size == 0:
            key_cache.tofile(os.path.join(gen_data_params.data_path, "k.bin"))
            value_cache.tofile(os.path.join(gen_data_params.data_path, "v.bin"))
        else :
            key_cache_paged.tofile(os.path.join(gen_data_params.data_path, "k.bin"))
            value_cache_paged.tofile(os.path.join(gen_data_params.data_path, "v.bin"))
            block_table.tofile(os.path.join(gen_data_params.data_path, "block_table.bin"))

        np.array(gen_data_params.q_seqlen_list).astype(np.int64).tofile(
            os.path.join(gen_data_params.data_path, "q_seqlen.bin"))
        np.array(gen_data_params.k_seqlen_list).astype(np.int64).tofile(
            os.path.join(gen_data_params.data_path, "kv_seqlen.bin"))
        ref_output.astype(np.float32).tofile(os.path.join(gen_data_params.data_path, "golden.bin"))
        ref_output.tofile(os.path.join(gen_data_params.data_path, "golden_dtype.bin"))

import torch
def set_seed(seedValue):
    np.random.seed(seedValue)
    random.seed(seedValue)
    torch.manual_seed(seedValue)


if __name__ == "__main__":
    # set_seed(1)

    torch.set_printoptions(
        precision=6,        # 浮点数保留小数位数
        threshold=128,     # 元素超过该数量才简略显示(...)
        edgeitems=5,        # 头尾各展示多少个元素
        linewidth=11 * 31,      # 单行字符宽度，自动换行
        sci_mode=False      # False=普通小数，True=强制科学计数法
    )

    np.set_printoptions(
        precision=6,        # 浮点数保留几位小数
        threshold=128,    # 元素总数超过该值就缩略显示(...)
        edgeitems=5,       # 矩阵首尾各展示几个元素
        linewidth=11 * 30,     # 单行最大字符长度，超出自动换行
        suppress=True,     # True：极小值不用科学计数法；False：很小的数自动e计数
        sign=" ",          # " "正数前面留空格，"+"强制显示正号，"-"只显示负号
        formatter={"float": lambda x: f"{x:.6f},"}
    )

    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    embedding_size = int(sys.argv[6])
    is_varied_len = int(sys.argv[7])
    silu_scale = float(sys.argv[8])
    layout = str(sys.argv[9])
    str_dtype = str(sys.argv[10])
    mask_type = int(sys.argv[11])
    paged_block_size = int(sys.argv[12])
    data_path = str(sys.argv[13])

    os.makedirs(data_path, exist_ok=True)

    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, batch)
    testObj = TestHstuInfer()
    gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                            kv_heads, embedding_size,
                                            silu_scale, layout, dtype, data_path, mask_type,
                                            paged_block_size)
    testObj.calc_data(gen_data_params)
