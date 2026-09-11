# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np


BLOCK_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Generate inputs and golden output for the x_attention example")
    parser.add_argument("batch", type=int)
    parser.add_argument("beam_size", type=int)
    parser.add_argument("shared_kv_seq_len", type=int)
    parser.add_argument("num_heads", type=int)
    parser.add_argument("kv_heads", type=int)
    parser.add_argument("embedding_size", type=int)
    parser.add_argument("max_decode_step", type=int)
    parser.add_argument("decode_step", type=int)
    parser.add_argument("cache_mode", type=int, choices=(0, 1))
    parser.add_argument("dtype", choices=("half", "bf16"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "data"))
    return parser.parse_args()


def validate(args):
    positive = (
        args.batch,
        args.beam_size,
        args.shared_kv_seq_len,
        args.num_heads,
        args.kv_heads,
        args.embedding_size,
        args.max_decode_step,
        args.decode_step,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all shape arguments must be positive")
    if args.num_heads % args.kv_heads != 0:
        raise ValueError("num_heads must be divisible by kv_heads")
    if args.embedding_size != 128:
        raise ValueError("embedding_size must be 128")
    if args.max_decode_step > 256 or args.decode_step > args.max_decode_step:
        raise ValueError("require 1 <= decode_step <= max_decode_step <= 256")
    if args.num_heads // args.kv_heads > 128:
        raise ValueError("GQA group size must not exceed 128")


def softmax(scores):
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    return weights / np.sum(weights)


def write_array(output_dir, name, value):
    np.asarray(value).tofile(output_dir / name)


def generate(args):
    validate(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    if args.dtype == "half":
        data_type = np.float16
    else:
        from ml_dtypes import bfloat16

        data_type = bfloat16
    group_size = args.num_heads // args.kv_heads
    max_blocks_per_batch = (args.shared_kv_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = args.batch * max_blocks_per_batch

    query = rng.uniform(
        -0.5,
        0.5,
        size=(args.batch, args.beam_size, args.num_heads, args.embedding_size),
    ).astype(data_type)

    shared_paged = args.cache_mode == 0
    if shared_paged:
        shared_key = rng.uniform(
            -0.5,
            0.5,
            size=(num_blocks, BLOCK_SIZE, args.kv_heads, args.embedding_size),
        ).astype(data_type)
        shared_value = rng.uniform(
            -0.5,
            0.5,
            size=(num_blocks, BLOCK_SIZE, args.kv_heads, args.embedding_size),
        ).astype(data_type)
    else:
        shared_key = rng.uniform(
            -0.5,
            0.5,
            size=(args.batch, args.shared_kv_seq_len, args.kv_heads, args.embedding_size),
        ).astype(data_type)
        shared_value = rng.uniform(
            -0.5,
            0.5,
            size=(args.batch, args.shared_kv_seq_len, args.kv_heads, args.embedding_size),
        ).astype(data_type)

    unshared_shape = (
        args.batch,
        args.beam_size,
        args.kv_heads,
        args.max_decode_step,
        args.embedding_size,
    )
    unshared_key = rng.uniform(-0.5, 0.5, size=unshared_shape).astype(data_type)
    unshared_value = rng.uniform(-0.5, 0.5, size=unshared_shape).astype(data_type)

    shared_block_table = np.arange(num_blocks, dtype=np.int32).reshape(args.batch, max_blocks_per_batch)
    unshared_block_table = np.arange(args.batch, dtype=np.int32)
    shared_kv_lens = np.full(args.batch, args.shared_kv_seq_len, dtype=np.int32)
    decode_step = np.asarray([args.decode_step], dtype=np.int32)

    golden = np.empty_like(query)
    scale = 1.0 / np.sqrt(float(args.embedding_size))
    for batch_idx in range(args.batch):
        if shared_paged:
            block_ids = shared_block_table[batch_idx]
            shared_key_batch = shared_key[block_ids].reshape(-1, args.kv_heads, args.embedding_size)
            shared_value_batch = shared_value[block_ids].reshape(-1, args.kv_heads, args.embedding_size)
            shared_key_batch = shared_key_batch[: args.shared_kv_seq_len]
            shared_value_batch = shared_value_batch[: args.shared_kv_seq_len]
        else:
            shared_key_batch = shared_key[batch_idx]
            shared_value_batch = shared_value[batch_idx]

        request_idx = unshared_block_table[batch_idx] if not shared_paged else batch_idx
        for beam_idx in range(args.beam_size):
            for head_idx in range(args.num_heads):
                kv_head_idx = head_idx // group_size
                shared_k = shared_key_batch[:, kv_head_idx].astype(np.float32)
                shared_v = shared_value_batch[:, kv_head_idx].astype(np.float32)
                unshared_k = unshared_key[
                    request_idx, beam_idx, kv_head_idx, : args.decode_step
                ].astype(np.float32)
                unshared_v = unshared_value[
                    request_idx, beam_idx, kv_head_idx, : args.decode_step
                ].astype(np.float32)
                key = np.concatenate((shared_k, unshared_k), axis=0)
                value = np.concatenate((shared_v, unshared_v), axis=0)
                q = query[batch_idx, beam_idx, head_idx].astype(np.float32)
                probability = softmax(np.matmul(key, q) * scale)
                golden[batch_idx, beam_idx, head_idx] = np.matmul(probability, value).astype(data_type)

    write_array(output_dir, "query.bin", query)
    write_array(output_dir, "shared_key.bin", shared_key)
    write_array(output_dir, "shared_value.bin", shared_value)
    write_array(output_dir, "unshared_key.bin", unshared_key)
    write_array(output_dir, "unshared_value.bin", unshared_value)
    write_array(output_dir, "shared_block_table.bin", shared_block_table)
    write_array(output_dir, "unshared_block_table.bin", unshared_block_table)
    write_array(output_dir, "shared_kv_lens.bin", shared_kv_lens)
    write_array(output_dir, "decode_step.bin", decode_step)
    write_array(output_dir, "golden.bin", golden.astype(np.float32))

    print(f"Generated x_attention data in {output_dir}")
    print(
        f"shape: batch={args.batch}, beam={args.beam_size}, shared_kv={args.shared_kv_seq_len}, "
        f"heads={args.num_heads}/{args.kv_heads}, max_decode={args.max_decode_step}, "
        f"decode={args.decode_step}, cache_mode={args.cache_mode}, dtype={args.dtype}"
    )


if __name__ == "__main__":
    generate(parse_args())
