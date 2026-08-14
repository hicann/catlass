# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, List, Optional, Tuple

BASE_KV_SIZE = 128
Q_TILE_CEIL = 128
WS_FLOOR = 1024 * 1024 * 32 * 4
HN_MAX = 16
TILE_M = 128
TILE_N = 128

_BASIC_CONST = dict(
    innerPrec=0, actSeqAval=0,
    qBaseTile=128, kvBaseTile=128,
    qkL1TileM=128, qkL1TileN=256, qkL1TileKLeft=192, qkL1TileKRight=192,
    pvL1TileM=128, pvL1TileN=128, pvL1TileKLeft=256, pvL1TileKRight=128,
    qL1BufNum=1, kL1BufNum=2, vL1BufNum=2, pL1BufNum=3,
)

class Format:
    TND = 0
    BSND = 1

class MaskType:
    NO_MASK = 0

@dataclass
class FAInferTilingData:
    numHeads: int = 0
    embeddingSize: int = 0
    embeddingSizeV: int = 0
    numBlocks: int = 0
    blockSize: int = 0
    maxQSeqlen: int = 0
    maxKvSeqlen: int = 0
    kvHeads: int = 0
    batch: int = 0
    maxNumBlocksPerBatch: int = 0
    firstBatchTaskNum: int = 0
    totalTaskNum: int = 0
    maskType: int = 0
    workSpaceSize: int = 0
    scaleValue: float = 0.0
    cacheLayout: str = "nd"
    qSeqlenAligned: int = 0
    kvSeqlenAligned: int = 0
    innerPrec: int = 0
    actSeqAval: int = 0
    qBaseTile: int = 0
    kvBaseTile: int = 0
    qkL1TileM: int = 0
    qkL1TileN: int = 0
    qkL1TileKLeft: int = 0
    qkL1TileKRight: int = 0
    pvL1TileM: int = 0
    pvL1TileN: int = 0
    pvL1TileKLeft: int = 0
    pvL1TileKRight: int = 0
    qL1BufNum: int = 0
    kL1BufNum: int = 0
    vL1BufNum: int = 0
    pL1BufNum: int = 0
    qFormat: int = Format.BSND
    kvFormat: int = Format.BSND
    anyMaskEnabled: int = 0
    holeMaxNum: int = 0
    holeNumAddr: int = 0
    tileRangeAddr: int = 0
    sparseComputeAddr: int = 0
    sparseMaskAddr: int = 0
    maskrAddr: int = 0
    holelAddr: int = 0
    holesAddr: int = 0
    kvSeqlenList: Optional[List[int]] = None

TILING_INT_FIELDS: Tuple[str, ...] = (
    "batch",
    "numHeads",
    "kvHeads",
    "embeddingSize",
    "embeddingSizeV",
    "maxQSeqlen",
    "maxKvSeqlen",
    "firstBatchTaskNum",
    "totalTaskNum",
    "qBaseTile",
    "kvBaseTile",
    "maskType",
    "qFormat",
    "kvFormat",
    "anyMaskEnabled",
    "holeMaxNum",
)

def ceil_div(x: int, n: int) -> int:
    return (x + n - 1) // n

def _size(t: Any, dim: int) -> int:
    return tuple(getattr(t, "shape", ()))[dim]

def _to_int_list(t: Any) -> List[int]:
    if t is None:
        return []
    if hasattr(t, "tolist"):
        return [int(x) for x in t.tolist()]
    return [int(x) for x in t]

def build_q_seqlen_list(is_varlen_q: bool, host_cu_seqlens_q: Any,
                        batch_size: int, seqlen_q: int) -> List[int]:
    if is_varlen_q:
        return _to_int_list(host_cu_seqlens_q)
    return [seqlen_q for _ in range(batch_size)]

def build_kv_seqlen_list(is_varlen_q: bool, paged_KV: bool,
                         host_seqused_k: Any) -> List[int]:
    per_batch = _to_int_list(host_seqused_k)
    if is_varlen_q and not paged_KV:
        cum = [0]
        for x in per_batch:
            cum.append(cum[-1] + x)
        return cum
    return per_batch

def max_q_seqlen(is_varlen_q: bool, q_list: List[int]) -> int:
    if is_varlen_q:
        return max((q_list[i + 1] - q_list[i] for i in range(len(q_list) - 1)), default=0)
    return max(q_list, default=0)

def max_kv_seqlen(is_varlen_q: bool, paged_KV: bool, kv_list: List[int]) -> int:
    if is_varlen_q and not paged_KV:
        return max((kv_list[i + 1] - kv_list[i] for i in range(len(kv_list) - 1)), default=0)
    return max(kv_list, default=0)

def count_tasks(is_varlen_q: bool, num_heads: int,
                q_list: List[int], seqlen_q_bsnd: int, batch_size: int) -> Tuple[int, int]:
    first_batch_task = 0
    total_task = 0
    for b in range(batch_size):
        if is_varlen_q:
            q_seqlen = q_list[b + 1] - q_list[b]
        else:
            q_seqlen = seqlen_q_bsnd
        cur_task = num_heads * ceil_div(q_seqlen, Q_TILE_CEIL)
        if b == 0:
            first_batch_task = cur_task
        total_task += cur_task
    return first_batch_task, total_task

def run_host_tiling(q: Any, k: Any, v: Any,
                    cu_seqlens_q: Optional[Any] = None,
                    seqused_k: Optional[Any] = None,
                    max_seqlen_q: Optional[int] = None,
                    softmax_scale: Optional[float] = None,
                    paged_KV: bool = False,
                    is_bf16: bool = False,
                    hole_num: Optional[Any] = None,
                    tile_range: Optional[Any] = None,
                    sparse_compute: Optional[Any] = None,
                    sparse_mask: Optional[Any] = None,
                    maskr: Optional[Any] = None,
                    holel: Optional[Any] = None,
                    holes: Optional[Any] = None,
                    ) -> FAInferTilingData:
    is_varlen_q = q.dim() == 3

    if is_varlen_q:
        batch_size = _size(cu_seqlens_q, 0) - 1
        seqlen_q = max_seqlen_q
        num_heads = _size(q, 1)
        head_size_q = _size(q, 2)
        num_heads_k = _size(k, 2) if paged_KV else _size(k, 1)
    else:
        batch_size = _size(q, 0)
        seqlen_q = _size(q, 1)
        num_heads = _size(q, 2)
        head_size_q = _size(q, 3)
        num_heads_k = _size(k, 2)
    head_size_v = _size(v, -1)

    q_list = build_q_seqlen_list(is_varlen_q, cu_seqlens_q, batch_size, seqlen_q)
    kv_list = build_kv_seqlen_list(is_varlen_q, paged_KV, seqused_k)
    max_q = max_q_seqlen(is_varlen_q, q_list) if is_varlen_q else seqlen_q
    max_kv = max_kv_seqlen(is_varlen_q, paged_KV, kv_list)

    td = FAInferTilingData()
    td.numHeads = num_heads
    td.embeddingSize = head_size_q
    td.embeddingSizeV = head_size_v
    td.kvHeads = num_heads_k
    td.batch = batch_size
    td.maxQSeqlen = max_q
    td.maxKvSeqlen = max_kv
    td.blockSize = BASE_KV_SIZE
    td.numBlocks = 0
    td.maxNumBlocksPerBatch = 0
    td.scaleValue = (softmax_scale if softmax_scale is not None
                     else 1.0 / math.sqrt(float(head_size_q)))
    td.maskType = MaskType.NO_MASK
    td.cacheLayout = "nd"
    td.qFormat = Format.TND if is_varlen_q else Format.BSND
    td.kvFormat = td.qFormat
    for name, val in _BASIC_CONST.items():
        setattr(td, name, val)

    first_batch_task, total_task = count_tasks(is_varlen_q, num_heads, q_list, seqlen_q, batch_size)
    td.firstBatchTaskNum = first_batch_task
    td.totalTaskNum = total_task

    td.workSpaceSize = WS_FLOOR

    td.kvSeqlenList = kv_list

    any_mask_tensors = (hole_num, tile_range, sparse_compute, sparse_mask, maskr, holel, holes)
    any_mask_present = any(t is not None for t in any_mask_tensors)
    td.anyMaskEnabled = 1 if any_mask_present else 0
    if any_mask_present:
        Tq = ceil_div(seqlen_q, TILE_M)
        Tk = ceil_div(max_kv, TILE_N)
        Wk = ceil_div(Tk, 32)
        Hn = 0
        if hole_num is not None:
            Hn = _size(hole_num, 0)
        elif holel is not None:
            Hn = _size(holel, 2)
        td.holeMaxNum = Hn
        if tile_range is not None:
            td.tileRangeAddr = 1
        if maskr is not None:
            td.maskrAddr = 1
        if holel is not None:
            td.holelAddr = 1
        if holes is not None:
            td.holesAddr = 1
        if hole_num is not None:
            td.holeNumAddr = 1

    return td

def pack_tiling_int(td: FAInferTilingData) -> List[int]:
    return [int(getattr(td, name)) for name in TILING_INT_FIELDS]

def pack_tiling_scale(td: FAInferTilingData) -> List[float]:
    return [float(td.scaleValue)]

def unpack_index(field_name: str) -> int:
    return TILING_INT_FIELDS.index(field_name)

class _FakeTensor:
    def __init__(self, shape, data=None):
        self.shape = tuple(shape)
        self._data = data

    def tolist(self):
        return list(self._data) if self._data is not None else []

def _selfcheck() -> None:
    cu_q = _FakeTensor((3,), [0, 4, 10])
    seqused = _FakeTensor((3,), [3, 5, 2])
    q = _FakeTensor((10, 8, 128))
    k = _FakeTensor((10, 8, 128))
    v = _FakeTensor((10, 8, 128))
    td = run_host_tiling(q, k, v, cu_seqlens_q=cu_q, seqused_k=seqused,
                         max_seqlen_q=6, softmax_scale=None, is_bf16=False)

    q2 = _FakeTensor((2, 256, 8, 128))
    k2 = _FakeTensor((2, 256, 8, 128))
    v2 = _FakeTensor((2, 256, 8, 128))
    seqused2 = _FakeTensor((2,), [200, 256])
    td2 = run_host_tiling(q2, k2, v2, seqused_k=seqused2, softmax_scale=None)

    ti = pack_tiling_int(td2)

    print("bsa_tiling self-check passed:")
    print(f"  TND : batch={td.batch} maxQ={td.maxQSeqlen} maxKv={td.maxKvSeqlen} "
          f"first={td.firstBatchTaskNum} total={td.totalTaskNum}")
    print(f"  BSND: batch={td2.batch} maxQ={td2.maxQSeqlen} maxKv={td2.maxKvSeqlen} "
          f"first={td2.firstBatchTaskNum} total={td2.totalTaskNum}")

if __name__ == "__main__":
    _selfcheck()
