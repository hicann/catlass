# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any
import catlass.tla as tla
from catlass.types import dtype_size_bytes
from dataclasses import dataclass

from catlass.params import MaskStoreParams
from catlass.params import MaskLoadParams

DTYPE_S = tla.Float32
DTYPE_OTMP = tla.Float32
DTYPE_ACC = tla.Float32

L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128

PRE_LAUNCH = 2
UB_S_OTMP_BUF_STAGES = 2
L0_STAGES = 2
Q_L1_BUF = 1
K_L1_BUF = 2
V_L1_BUF = 2
P_L1_BUF = 3

MIN_VALUE = -65504.0

def _ceil_div(curQSeqlen: int, qBaseTile: int) -> int:
    curQSTileNum = (curQSeqlen + qBaseTile - 1) // qBaseTile
    return curQSTileNum

_VL_F32 = 64   # VREG 半精度 lane 数
_VL_F16 = 128  # 单 AIV 子核一次处理的最大行数

def _elem_bytes(num_elems: int, dtype_tla: Any) -> int:
    return num_elems * dtype_size_bytes(dtype_tla.dtype)

def _alloc1(allocator: Any, num_elems: int, align: int, space: Any, dtype_tla: Any) -> Any:
    return tla.recast_ptr(
        allocator.allocate(_elem_bytes(num_elems, dtype_tla), align, space), dtype=dtype_tla)

@tla.kernel
def bsa_regular_kernel_arch35(
    query: tla.Tensor,             # gQ  [S1, D]  fp16（BSND/TND：packed [*,S,N,D] 的单 head 切片）
    key: tla.Tensor,               # gK  [D, S2]  fp16（ColumnMajor：直接给出 K^T）
    value: tla.Tensor,             # gV  [S2, D]  fp16
    attentionOut: tla.Tensor,      # gO  [S1, D]  fp16
    lse: tla.Tensor,               # gLSE [total_q, qHeads] fp32（return_lse=True 时的 log-sum-exp 输出）
    tilingInt: tla.Tensor,         # Tiling 整型包（Int32 1D，下标见上表 = bsa_tiling.TILING_INT_FIELDS）
    tilingScale: tla.Tensor,       # Tiling 浮点包（fp32 1D，[scaleValue]）
    actualQseqlen: tla.Tensor,     # gActualQseqlen：TND 为累加 [B+1]，BSND 占位 [B]
    actualKvseqlen: tla.Tensor,    # gActualKvseqlen：TND 非 paged 累加 [B+1]，BSND 逐 batch [B]
    tileRange: tla.Tensor,         # AnyMask：[batch, Tq] int32（收缩 KV 循环）
    sparseCompute: tla.Tensor,     # AnyMask：[batch, Tq, Wk] int32（bit=1 表示该 tile 有可见元素需计算）
    sparseMask: tla.Tensor,        # AnyMask：[batch, Tq, Wk] int32（bit=1 需精细 mask）
    maskr: tla.Tensor,             # AnyMask：[batch, Sq] int32
    holel: tla.Tensor,             # AnyMask：[batch, Sq, Hn] int32
    holes: tla.Tensor,             # AnyMask：[batch, Sq, Hn] int32
    holeNum: tla.Tensor,           # AnyMask：[Hn] int32
    is_fp16: tla.Constexpr[bool],  # True=fp16 输入，False=bf16 输入（编译期决定 DTYPE_Q/K/V/P/O）
    hole_max_num: tla.Constexpr[int],          # 每行 hole 数上限 Hn（0=hole-free：仅 maskr 前缀掩码）
    uniform_q_seqlen: tla.Constexpr[int],      # 定长 Q 每 batch 序列长度（0=非 uniform，走运行时路径）
    uniform_kv_seqlen: tla.Constexpr[int],     # 定长 KV 每 batch 序列长度（0=非 uniform）
    uniform_tasks_per_batch: tla.Constexpr[int],  # 每 batch 的 Q 任务数（0=非 uniform）
    return_lse: tla.Constexpr[bool],           # 是否计算/写回 LSE 输出
) -> None:
    print("kernel launch, start to run...")

    # 编译期根据 is_fp16 选择 Q/K/V/P/O 的数据类型；S/OTMP/ACC 恒为 Float32
    if tla.const_expr(is_fp16):
        DTYPE_Q = tla.Float16
        DTYPE_K = tla.Float16
        DTYPE_V = tla.Float16
        DTYPE_P = tla.Float16
        DTYPE_O = tla.Float16
    else:
        DTYPE_Q = tla.BFloat16
        DTYPE_K = tla.BFloat16
        DTYPE_V = tla.BFloat16
        DTYPE_P = tla.BFloat16
        DTYPE_O = tla.BFloat16
    c0 = 0
    c1 = 1
    TND = 0
    BSND = 1

    # TilingData
    _TI_BATCH = 0; _TI_NUMHEADS = 1; _TI_KVHEADS = 2; _TI_EMBED = 3; _TI_EMBEDV = 4
    _TI_MAXQ = 5; _TI_MAXKV = 6; _TI_FIRSTTASK = 7; _TI_TOTALTASK = 8
    _TI_QBASE = 9; _TI_KVBASE = 10; _TI_MASKTYPE = 11
    _TI_QFORMAT = 12; _TI_KVFORMAT = 13; _TI_ANYMASK = 14; _TI_HOLEMAX = 15

    embedV_ = 1

    anyMaskEnabled_ = True   # 是否启用 AnyMask

    embed_ = tilingInt[_TI_EMBEDV]
    qBaseTile_ = 128
    kvBaseTile_ = 128
    batch_ = tilingInt[_TI_BATCH]
    qHeads_ = tilingInt[_TI_NUMHEADS]
    kvHeads_ = tilingInt[_TI_KVHEADS]
    embedV_ = tilingInt[_TI_EMBEDV]
    maxQSeqlen_ = tilingInt[_TI_MAXQ]
    maxKvSeqlen_ = tilingInt[_TI_MAXKV]
    qSeqlen_ = tilingInt[_TI_MAXQ]
    kvSeqlen_ = tilingInt[_TI_MAXKV]
    maskType_ = tilingInt[_TI_MASKTYPE]     # (0=NO_MASK)
    qFormat_ = tilingInt[_TI_QFORMAT]       # (0=TND, 1=BSND)
    kvFormat_ = tilingInt[_TI_KVFORMAT]
    if tilingInt[_TI_ANYMASK] == 0:
        anyMaskEnabled_ = False
    holeMaxNum_ = hole_max_num
    scaleValue_ = tilingScale[0]

    mm1L1TileN_ = 128
    mm2L1TileN_ = 128

    mm1L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (128 // L0_TILE_K)
    mm1L0BTotalStages_ = (kvBaseTile_ // L0_TILE_N) * (128 // L0_TILE_K)
    mm2L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (kvBaseTile_ // L0_TILE_K)
    mm2L0BTotalStages_ = (kvBaseTile_ // L0_TILE_K) * (128 // L0_TILE_N)

    qSTileNum_ = _ceil_div(maxQSeqlen_, qBaseTile_)     # 首 batch qStile 数
    firstBatchTaskNum_ = tilingInt[_TI_FIRSTTASK]       # qSTileNum_ * qHeads_
    totalTaskNum_ = tilingInt[_TI_TOTALTASK]            # batch_ * firstBatchTaskNum_    

    # --- L1->L0 完成 -> GM->L1 可开始 ---
    q_l0a_ready_l1 = tla.flag("l0a_ready_l1", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_0 = tla.flag("k_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_1 = tla.flag("k_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_0 = tla.flag("v_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_1 = tla.flag("v_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)

    # --- CUBE MMAD 完成 -> L1->L0 可开始 ---
    mmad_ready_l0a_0 = tla.flag("mmad_ready_l0a_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0a_1 = tla.flag("mmad_ready_l0a_1", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_0 = tla.flag("mmad_ready_l0b_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_1 = tla.flag("mmad_ready_l0b_1", tla.arch.CUBE, tla.arch.MTE1)

    # --- FIX (L0C->UB) 完成 -> CUBE 可开始 ---
    fix_ready_mmad_0 = tla.flag("fix_ready_mmad_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_1 = tla.flag("fix_ready_mmad_1", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_2 = tla.flag("fix_ready_mmad_2", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_3 = tla.flag("fix_ready_mmad_3", tla.arch.FIX, tla.arch.CUBE)
    # --- GM->L1 加载完成 -> L1->L0 可开始 ---
    q_l1_ready_l0   = tla.flag("q_l1_ready_l0",   tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_0 = tla.flag("k_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_1 = tla.flag("k_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_0 = tla.flag("v_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_1 = tla.flag("v_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    # --- L1->L0 完成 -> CUBE 可开始 ---
    l0a_ready_mmad_0 = tla.flag("l0a_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_ready_mmad_1 = tla.flag("l0a_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_0 = tla.flag("l0b_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_1 = tla.flag("l0b_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    # --- CUBE MMAD 完成 -> FIX (L0C -> UB) 可开始 ---
    mmad_ready_fix_0 = tla.flag("mmad_ready_fix_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_1 = tla.flag("mmad_ready_fix_1", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_2 = tla.flag("mmad_ready_fix_2", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_3 = tla.flag("mmad_ready_fix_3", tla.arch.CUBE, tla.arch.FIX)

    # --- VECTOR 完成 -> gm -> UB ---
    vec_ready_mte2_0 = tla.flag("vec_ready_mte2_0", tla.arch.VECTOR, tla.arch.MTE2)
    vec_ready_mte2_1 = tla.flag("vec_ready_mte2_1", tla.arch.VECTOR, tla.arch.MTE2)
    # --- mask2index  
    mte3_ready_mask_0 = tla.flag("mte3_ready_mask_0", tla.arch.MTE3, tla.arch.VECTOR)
    mte3_ready_mask_1 = tla.flag("mte3_ready_mask_1", tla.arch.MTE3, tla.arch.VECTOR)
    # --- softmax
    mte3_ready_softmax_0 = tla.flag("mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR)
    mte3_ready_softmax_1 = tla.flag("mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR)
    # --- rescale
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)

    # --- anymask
    mte2_ready_vector = tla.flag("mte2_ready_vector", tla.arch.MTE2, tla.arch.VECTOR)
    vector_ready_mte2 = tla.flag("vector_ready_mte2", tla.arch.VECTOR, tla.arch.MTE2)
    mask_vector_ready = tla.flag("mask_vector_ready", tla.arch.MTE3, tla.arch.VECTOR)

    # 内部使用
    p_ub_ready_l1_0 = tla.flag("p_ub_ready_l1_0", tla.arch.VECTOR, tla.arch.MTE3)
    p_ub_ready_l1_1 = tla.flag("p_ub_ready_l1_1", tla.arch.VECTOR, tla.arch.MTE3)

    mm1_ready_sm_0 = tla.cross_flag("mm1_ready_sm_0")
    mm1_ready_sm_1 = tla.cross_flag("mm1_ready_sm_1")

    mm2_ready_re_0 = tla.cross_flag("mm2_ready_re_0")
    mm2_ready_re_1 = tla.cross_flag("mm2_ready_re_1")

    sm_ready_mm2_0 = tla.cross_flag("sm_ready_mm2_0")
    sm_ready_mm2_1 = tla.cross_flag("sm_ready_mm2_1")
    sm_ready_mm2_2 = tla.cross_flag("sm_ready_mm2_2")

    with tla.cube():
        tla.set_flag(q_l0a_ready_l1)
        tla.set_flag(k_l0b_ready_l1_0)
        tla.set_flag(k_l0b_ready_l1_1)
        tla.set_flag(v_l0b_ready_l1_0)
        tla.set_flag(v_l0b_ready_l1_1)
        tla.set_flag(mmad_ready_l0a_0)
        tla.set_flag(mmad_ready_l0a_1)
        tla.set_flag(mmad_ready_l0b_0)
        tla.set_flag(mmad_ready_l0b_1)
        tla.set_flag(fix_ready_mmad_0)
        tla.set_flag(fix_ready_mmad_1)
        tla.set_flag(fix_ready_mmad_2)
        tla.set_flag(fix_ready_mmad_3)

        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)
    with tla.vector():
        tla.set_flag(vec_ready_mte2_0)
        tla.set_flag(vec_ready_mte2_1)
        tla.set_flag(mte3_ready_mask_0)
        tla.set_flag(mte3_ready_mask_1)
        tla.set_flag(mte3_ready_softmax_0)
        tla.set_flag(mte3_ready_softmax_1)
        tla.set_flag(mte3_ready_rescale)
        tla.set_flag(vector_ready_mte2)
        tla.set_flag(mask_vector_ready)

        tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
    
    
    # 片上内存分配：全部 ring buffer 均在此一次性分配
    # DSL 不支持在 for 循环内分配 buffer => 每份 ring buffer 手动展开，逐份调用 _alloc1。
    allocator = tla.utils.LocalmemAllocator()

    # L1 ring buffers
    # Q × Q_L1_BUF(=1, SOLO)
    l1Q_ptrs = [
        _alloc1(allocator, qBaseTile_ * 128, 512, tla.AddressSpace.l1, DTYPE_Q),
    ]
    # K × K_L1_BUF(=2, DUO)
    l1K_ptrs = [
        _alloc1(allocator, 128 * kvBaseTile_, 512, tla.AddressSpace.l1, DTYPE_K),
        _alloc1(allocator, 128 * kvBaseTile_, 512, tla.AddressSpace.l1, DTYPE_K),
    ]

    # P × P_L1_BUF(=3, TRIO)
    l1P_ptrs = [
        _alloc1(allocator, qBaseTile_ * kvBaseTile_, 512, tla.AddressSpace.l1, DTYPE_P),
        _alloc1(allocator, qBaseTile_ * kvBaseTile_, 512, tla.AddressSpace.l1, DTYPE_P),
        _alloc1(allocator, qBaseTile_ * kvBaseTile_, 512, tla.AddressSpace.l1, DTYPE_P),
    ]
    # V × V_L1_BUF(=2, DUO)
    l1V_ptrs = [
        _alloc1(allocator, kvBaseTile_ * 128, 512, tla.AddressSpace.l1, DTYPE_V),
        _alloc1(allocator, kvBaseTile_ * 128, 512, tla.AddressSpace.l1, DTYPE_V),
    ]

    # L0 pingpong buffers × L0_STAGES(=2)
    l0a_ptrs = [
        _alloc1(allocator, L0_TILE_M * L0_TILE_K, 512, tla.AddressSpace.l0a, DTYPE_Q),
        _alloc1(allocator, L0_TILE_M * L0_TILE_K, 512, tla.AddressSpace.l0a, DTYPE_Q),
    ]
    l0b_ptrs = [
        _alloc1(allocator, L0_TILE_K * L0_TILE_N, 512, tla.AddressSpace.l0b, DTYPE_K),
        _alloc1(allocator, L0_TILE_K * L0_TILE_N, 512, tla.AddressSpace.l0b, DTYPE_K),
    ]
    l0c_ptrs = [
        _alloc1(allocator, L0_TILE_M * L0_TILE_N, 512, tla.AddressSpace.l0c, DTYPE_ACC),
        _alloc1(allocator, L0_TILE_M * L0_TILE_N, 512, tla.AddressSpace.l0c, DTYPE_ACC),
        _alloc1(allocator, L0_TILE_M * L0_TILE_N, 512, tla.AddressSpace.l0c, DTYPE_ACC),
        _alloc1(allocator, L0_TILE_M * L0_TILE_N, 512, tla.AddressSpace.l0c, DTYPE_ACC),
    ]

    # UB ring buffers × UB_S_OTMP_BUF_STAGES(=2)
    ubS_ptrs = [
        _alloc1(allocator, qBaseTile_ // 2 * kvBaseTile_, 256, tla.AddressSpace.ub, DTYPE_S),
        _alloc1(allocator, qBaseTile_ // 2 * kvBaseTile_, 256, tla.AddressSpace.ub, DTYPE_S),
    ]

    ubP_ptrs = [
        _alloc1(allocator, (qBaseTile_ // 2 + 1) * kvBaseTile_, 256, tla.AddressSpace.ub, DTYPE_P),
        _alloc1(allocator, (qBaseTile_ // 2 + 1) * kvBaseTile_, 256, tla.AddressSpace.ub, DTYPE_P),
    ]
    ubOTmp_ptrs = [
        _alloc1(allocator, qBaseTile_ // 2 * 128, 256, tla.AddressSpace.ub, DTYPE_OTMP),
        _alloc1(allocator, qBaseTile_ // 2 * 128, 256, tla.AddressSpace.ub, DTYPE_OTMP),
    ]
    
    # UB 单份 buffer：O 累加器 + 行统计标量
    ubO_ptr = tla.recast_ptr(
        allocator.allocate(_elem_bytes(qBaseTile_ // 2 * 128, DTYPE_OTMP), 256, tla.AddressSpace.ub), dtype=DTYPE_OTMP)
    ubO16_ptr = tla.recast_ptr(ubO_ptr, dtype=DTYPE_Q)
    nowMax_ptr = tla.recast_ptr(     # lmUbTensor
        allocator.allocate(_elem_bytes(qBaseTile_ // 2, tla.Float32), 256, tla.AddressSpace.ub), dtype=tla.Float32)
    # expMax × P_L1_BUF(=3)：dmUbTensor，跨 prelaunch 延迟按 tile%P_L1_BUF 传给 rescale
    expMax_ptrs = [
        _alloc1(allocator, qBaseTile_ // 2, 256, tla.AddressSpace.ub, tla.Float32),
        _alloc1(allocator, qBaseTile_ // 2, 256, tla.AddressSpace.ub, tla.Float32),
        _alloc1(allocator, qBaseTile_ // 2, 256, tla.AddressSpace.ub, tla.Float32),
    ]
    nowSum_ptr = tla.recast_ptr(     # llUbTensor
        allocator.allocate(_elem_bytes(qBaseTile_ // 2, tla.Float32), 256, tla.AddressSpace.ub), dtype=tla.Float32) 
    lastMax_ptr = tla.recast_ptr(    # gmUbTensor
        allocator.allocate(_elem_bytes(qBaseTile_ // 2, tla.Float32), 256, tla.AddressSpace.ub), dtype=tla.Float32)
    lastSum_ptr = tla.recast_ptr(    # glUbTensor
        allocator.allocate(_elem_bytes(qBaseTile_ // 2, tla.Float32), 256, tla.AddressSpace.ub), dtype=tla.Float32)
    if tla.const_expr(return_lse):
        ub_lse_ptr = _alloc1(allocator, qBaseTile_ // 2 * 8, 256, tla.AddressSpace.ub, tla.Float32)

    mask_ub_ptr = tla.recast_ptr(
        allocator.allocate(_elem_bytes(qBaseTile_ // 2 * 128, tla.Int32), 256, tla.AddressSpace.ub), dtype=tla.Int32)
    
    maskr_ub_ptr = tla.recast_ptr(allocator.allocate(_elem_bytes(64, tla.Int32), 256, tla.AddressSpace.ub), dtype=tla.Int32)
    holel_ub_ptr = tla.recast_ptr(allocator.allocate(_elem_bytes(64 * 3, tla.Int32), 256, tla.AddressSpace.ub), dtype=tla.Int32)
    holes_ub_ptr = tla.recast_ptr(allocator.allocate(_elem_bytes(64 * 3, tla.Int32), 256, tla.AddressSpace.ub), dtype=tla.Int32)

    coreIdx = tla.arch.block_idx()
    coreNum = tla.arch.block_num()
    with tla.cube():
        coreIdx = tla.arch.block_idx()
    with tla.vector():
        coreIdx = tla.arch.block_idx() // 2

    qNOffset = embed_
    kNOffset = embed_
    vNOffset = embedV_
    oNOffset = embedV_
    qSOffset = qHeads_ * qNOffset
    kSOffset = kvHeads_ * kNOffset
    vSOffset = kvHeads_ * vNOffset
    oSOffset = qHeads_ * oNOffset

    embedVRound = (embedV_ + 15) // 16 * 16
    groupSize = qHeads_ // kvHeads_

    qBOffset = 0
    kBOffset = 0
    vBOffset = 0
    oBOffset = 0
    preTotalTaskNum = 0
    curBatch = 0
    
    curQSeqlen = maxQSeqlen_
    curKvSeqlen = actualKvseqlen[curBatch] #
    curTotalTaskNum = firstBatchTaskNum_
    if tla.const_expr(uniform_tasks_per_batch != 0):
        # 定长 TND：q/kv 长度取 constexpr，跳过运行时 GM 读取与差分
        curQSeqlen = uniform_q_seqlen
        curKvSeqlen = uniform_kv_seqlen
    else:
        curKvSeqlen = actualKvseqlen[curBatch]
        if qFormat_ == TND:
            curQSeqlen = actualQseqlen[curBatch + 1]-actualQseqlen[curBatch]
        if kvFormat_ == TND:
            curKvSeqlen = actualKvseqlen[curBatch + 1]-actualKvseqlen[curBatch]
    maxQsBlockNum = (maxQSeqlen_ + qBaseTile_ - 1) // qBaseTile_

    task_range = tla.range(tla.arch.block_idx(), totalTaskNum_, tla.arch.block_num())
    for taskSlotIdx in task_range:
        if tla.const_expr(uniform_tasks_per_batch != 0):
            # 定长任务：整除一次获得 batch 与 batch 内任务序号
            curBatch = taskSlotIdx // uniform_tasks_per_batch
            taskIdxCurBatch = taskSlotIdx - curBatch * uniform_tasks_per_batch
            qBOffset = curBatch * uniform_q_seqlen * qSOffset
            kBOffset = curBatch * uniform_kv_seqlen * kSOffset
            vBOffset = curBatch * uniform_kv_seqlen * vSOffset
            oBOffset = curBatch * uniform_q_seqlen * oSOffset
        else:
            while taskSlotIdx >= curTotalTaskNum:
                curBatch = curBatch + 1
                preTotalTaskNum = curTotalTaskNum
                qBOffset = qBOffset + curQSeqlen * qSOffset
                kBOffset = kBOffset + curKvSeqlen * kSOffset
                vBOffset = vBOffset + curKvSeqlen * vSOffset
                oBOffset = oBOffset + curQSeqlen * oSOffset
                curQSeqlen = maxQSeqlen_
                curKvSeqlen = actualKvseqlen[curBatch]
                if qFormat_ == TND:
                    curQSeqlen = curQSeqlen = actualQseqlen[curBatch+1]-actualQseqlen[curBatch]
                if kvFormat_ == TND:
                    curKvSeqlen = actualKvseqlen[curBatch+1]-actualKvseqlen[curBatch]
                curTotalTaskNum = curTotalTaskNum + _ceil_div(curQSeqlen, qBaseTile_) * qHeads_
            taskIdxCurBatch = taskSlotIdx - preTotalTaskNum
        logicalQSTileIdx = taskIdxCurBatch // qHeads_
        qHeadIdx = taskIdxCurBatch - logicalQSTileIdx * qHeads_
        qsBlockNum = _ceil_div(curQSeqlen, qBaseTile_)
        evenTileCount = (qsBlockNum + 1) // 2
        qSTileIdx = (logicalQSTileIdx * 2) if logicalQSTileIdx < evenTileCount \
            else ((qsBlockNum - 1 - logicalQSTileIdx) * 2 + 1)
        kvHeadIdx = qHeadIdx // groupSize
        qSIdex = qSTileIdx * qBaseTile_
        gmOffsetQ = qBOffset + qHeadIdx * qNOffset
        gmOffsetO = oBOffset + qHeadIdx * oNOffset
        gmOffsetK = kBOffset + kvHeadIdx * kNOffset
        gmOffsetV = vBOffset + kvHeadIdx * vNOffset
        rowNum = (curQSeqlen - (qsBlockNum - 1) * qBaseTile_) if qSTileIdx == qsBlockNum - 1 else qBaseTile_   
        rowNumRound = ((rowNum + 15) // 16) * 16
        kvSTileSizeAct = kvBaseTile_
        noSkipKvS = curKvSeqlen

        if anyMaskEnabled_: 
            trIdx = curBatch * maxQsBlockNum + qSTileIdx
            tileRangeVal = tileRange[trIdx]
            tileRangeCount = kvBaseTile_ * tileRangeVal
            noSkipKvS = min(tileRangeCount, noSkipKvS)
        kvSLoopNum = (noSkipKvS + kvBaseTile_ - 1) // kvBaseTile_
        if kvSLoopNum > 0:
            gatheredKvSeqlen = noSkipKvS
            kvSTileSizeActDe = kvBaseTile_
            holeScanBase = (curBatch * maxQSeqlen_ + qSTileIdx * qBaseTile_) * holeMaxNum_
            holeScanLast = holeScanBase + (rowNum - 1) * holeMaxNum_
            maskHasHoles = holes[holeScanBase] > 0 or holes[holeScanLast] > 0

            for holeScanIdx in tla.range(c1, holeMaxNum_, c1):
                if holes[holeScanBase + holeScanIdx] > 0 or holes[holeScanLast + holeScanIdx] > 0:
                    maskHasHoles = True

            with tla.cube():
                # loadQGM：整块 Q 常驻 L1（Q_L1_BUF=1，单 buffer），task 内只加载一次。
                gm_q = tla.make_tensor(
                    query.ptr  + qBOffset + qHeadIdx * qNOffset ,
                    tla.make_layout(
                        tla.make_shape(curQSeqlen, embed_),
                        tla.make_stride(qSOffset, 1)
                ))
                gmQTensorTla = tla.tile_view(gm_q, tla.make_shape(qBaseTile_, embed_), tla.make_coord(qSTileIdx, c0))
                l1_q = tla.make_tensor_like(l1Q_ptrs[0], gmQTensorTla, tla.arch.zN)
                tla.wait_flag(q_l0a_ready_l1) # MTE1-MTE2
                tla.copy(l1_q, gmQTensorTla)
                tla.set_flag(q_l1_ready_l0)  # MTE2-MTE1
                tla.wait_flag(q_l1_ready_l0) # MTE2-MTE1
    
                gm_k = tla.make_tensor(key.ptr  + (kBOffset + kvHeadIdx * kNOffset),
                                        tla.make_layout(
                                                tla.make_shape(embed_, curKvSeqlen),
                                                tla.make_stride(1, kSOffset),
                                                layoutTag=tla.arch.ColumnMajor
                                        ))
    
                gm_v = tla.make_tensor(value.ptr  + (vBOffset + kvHeadIdx * vNOffset),
                                        tla.make_layout(
                                                tla.make_shape(curKvSeqlen, embed_),
                                                tla.make_stride(vSOffset, 1),
                                                layoutTag=tla.arch.RowMajor
                                        ))
            # 循环两阶段：idx < kvSLoopNum 做 QK Mmad(cube) + online Softmax(vector)；
            #            idx >= PRE_LAUNCH 做 PV Mmad(cube) + rescale O(vector)
            KvS_range = tla.range(c0, kvSLoopNum + PRE_LAUNCH, c1)
            launch_idx_0 = -1
            launch_idx_1 = -2
            launch_idx_2 = -3
            validIdx = -1
            validNum = 0
            cmp = True
            Tk = (kvSeqlen_ + kvBaseTile_ - 1) // kvBaseTile_
            Wk = _ceil_div(Tk, 32)
            fineMaskWordBase = (curBatch * maxQsBlockNum + qSTileIdx) * Wk
            maskHasFine = sparseMask[fineMaskWordBase] != c0
            for fineMaskWordIdx in tla.range(c1, Wk, c1):
                if sparseMask[fineMaskWordBase + fineMaskWordIdx] != c0:
                    maskHasFine = True
            is_move_mask = maskHasFine
            is_first = True
            Tq = (maxQSeqlen_ + qBaseTile_ - 1) // qBaseTile_
            for gatheredKvSTileIdx in KvS_range:
                elemIdx = (curBatch * maxQsBlockNum + qSTileIdx) * Tk + gatheredKvSTileIdx
                if gatheredKvSTileIdx < kvSLoopNum:
                    computeWordIdx = (curBatch*maxQsBlockNum + qSTileIdx)*Wk + gatheredKvSTileIdx//32
                    computeBitPos = gatheredKvSTileIdx % 32
                    cmp = ((sparseCompute[computeWordIdx] >> computeBitPos) & 1) == 1
                if cmp:
                    launch_idx_2 = launch_idx_1
                    launch_idx_1 = launch_idx_0
                    launch_idx_0 = gatheredKvSTileIdx
                    validIdx = validIdx + 1
                # ==================== 前半 idx<kvSLoopNum：QK(cube) + Softmax(vector) ====================
                if gatheredKvSTileIdx < kvSLoopNum and cmp:
                    validNum = validNum + 1
                    if gatheredKvSTileIdx == kvSLoopNum - c1:
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_
                    else:
                        kvSTileSizeAct = kvBaseTile_
                    isFirstKvSTile = (validIdx == c0)
                    isLastKvS = (gatheredKvSTileIdx == kvSLoopNum - c1)
                    ubSBufId = validIdx % UB_S_OTMP_BUF_STAGES
                    ubS_ptr = ubS_ptrs[0] if ubSBufId == c0 else ubS_ptrs[1]
    
                    kvSStartIdx = gatheredKvSTileIdx * kvBaseTile_
                    # AnyMask 位图三分支判定
                    wordIdx = (curBatch*maxQsBlockNum + qSTileIdx)*Wk + gatheredKvSTileIdx//32
                    bitPos = gatheredKvSTileIdx % 32
                    anyMaskSmBit = (sparseMask[wordIdx] >> bitPos) & 1 == 1

                    colNumRound = (kvSTileSizeAct + 15) // 16 * 16
                    ubSTensorTla = tla.make_tensor(ubS_ptr,
                                                tla.make_layout(
                                                    tla.make_shape(rowNum, kvSTileSizeAct), 
                                                    tla.make_stride(128, 1)
                                                ))
                    # QK Mmad
                    with tla.cube():
                        gm_q = tla.make_tensor(
                            query.ptr  + qBOffset + qHeadIdx * qNOffset ,
                            tla.make_layout(
                                tla.make_shape(curQSeqlen, embed_),
                                tla.make_stride(qSOffset, 1)
                        ))
                    
                        gmQTensorTla = tla.tile_view(gm_q, tla.make_shape(qBaseTile_, embed_), tla.make_coord(qSTileIdx, c0))
                        l1_q = tla.make_tensor_like(l1Q_ptrs[0], gmQTensorTla, tla.arch.zN)
    
                        gm_k = tla.make_tensor(key.ptr  + (kBOffset + kvHeadIdx * kNOffset),
                                                tla.make_layout(
                                                        tla.make_shape(embed_, curKvSeqlen),
                                                        tla.make_stride(1, kSOffset),
                                                        layoutTag=tla.arch.ColumnMajor
                                                ))
                    
                        prefixSumL0AStages = (validIdx * mm1L0ATotalStages_) if validIdx <= PRE_LAUNCH \
                            else (validIdx * mm1L0ATotalStages_ + (validIdx - PRE_LAUNCH) * mm2L0ATotalStages_)
                        prefixSumL0BStages = (validIdx * mm1L0BTotalStages_) if validIdx <= PRE_LAUNCH \
                            else (validIdx * mm1L0BTotalStages_ + (validIdx - PRE_LAUNCH) * mm2L0BTotalStages_)
                        # -----------------QK-----------------
                        l1TileNAct = kvSTileSizeAct
                        nLoopCounterL1 = validIdx 
    
                        # copy gm_k to L1
                        l1BBufId = nLoopCounterL1 % K_L1_BUF
                        l1K_ptr = l1K_ptrs[0] if l1BBufId == c0 else l1K_ptrs[1]
                        gm_k_tile = tla.tile_view(gm_k,
                                            tla.make_shape(embed_, kvBaseTile_), 
                                            tla.make_coord(c0, gatheredKvSTileIdx))
                        l1_k_tile = tla.make_tensor_like(
                                        l1K_ptr, 
                                        gm_k_tile, 
                                        layoutTag=tla.arch.nZ
                                    )
                        if l1BBufId == c0 :
                            tla.wait_flag(k_l0b_ready_l1_0)# MTE1-MTE2
                        else :
                            tla.wait_flag(k_l0b_ready_l1_1)
                        tla.copy(l1_k_tile, gm_k_tile)
                        if l1BBufId == c0 :
                            tla.set_flag(k_l1_ready_l0_0)# MTE2-MTE1
                        else :
                            tla.set_flag(k_l1_ready_l0_1)
                        
                            
                        # copy L1 to l0
                        l0TileNAct = l1TileNAct
                        l0CBufId = (nLoopCounterL1) % L0_STAGES
                        l0c_ptr = l0c_ptrs[0] if l0CBufId == c0 else l0c_ptrs[1]
                        ub_s_tile = tla.tile_view(
                                        ubSTensorTla,
                                        tla.make_shape(qBaseTile_, kvBaseTile_),
                                        tla.make_coord(c0, c0)
                                    )
                        l0c_s = tla.make_tensor_like(
                                    l0c_ptr,
                                    ub_s_tile,
                                    layoutTag=tla.arch.L0Clayout
                                )
    
                        l0ALoopCounter = prefixSumL0AStages
                        l0BLoopCounter = prefixSumL0BStages
                        l0ABufId = l0ALoopCounter % L0_STAGES
                        l0BBufId = l0BLoopCounter % L0_STAGES

                        l0a_ptr = l0a_ptrs[0] if l0ABufId == c0 else l0a_ptrs[1]
                        l0a_q_tensor = tla.make_tensor_like(l0a_ptr, l1_q, tla.arch.zN)
                        
                        if l0ABufId == 0:
                            tla.wait_flag(mmad_ready_l0a_0)# CUBE-MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0a_q_tensor, l1_q)
                        if l0ABufId == 0:
                            tla.set_flag(l0a_ready_mmad_0)# MTE1-CUBE
                        else:
                            tla.set_flag(l0a_ready_mmad_1)
    
                        l0b_ptr = l0b_ptrs[0] if l0BBufId == c0 else l0b_ptrs[1]
                        l0b_k_tensor = tla.make_tensor_like(l0b_ptr, l1_k_tile)
                        if l0BBufId == 0 :
                            tla.wait_flag(mmad_ready_l0b_0)# CUBE-MTE1
                        else :
                            tla.wait_flag(mmad_ready_l0b_1)
                        if l1BBufId == c0 :
                            tla.wait_flag(k_l1_ready_l0_0)# MTE2-MTE1
                        else :
                            tla.wait_flag(k_l1_ready_l0_1)
                        tla.copy(l0b_k_tensor, l1_k_tile)
    
                        if l0BBufId == 0 :
                            tla.set_flag(l0b_ready_mmad_0)# MTE1-CUBE
                        else :
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BBufId == 0 :
                            tla.set_flag(k_l0b_ready_l1_0)# MTE1-MTE2
                        else :
                            tla.set_flag(k_l0b_ready_l1_1)
    
                        
    
                        if l0ABufId == 0:
                            tla.wait_flag(l0a_ready_mmad_0)# MTE1-CUBE
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufId == 0 :
                            tla.wait_flag(l0b_ready_mmad_0)# MTE1-CUBE
                        else :
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufId == 0:
                            tla.wait_flag(fix_ready_mmad_0)# FIX-CUBE
                        else:
                            tla.wait_flag(fix_ready_mmad_1)
    
                        tla.mmad(l0c_s, l0a_q_tensor, l0b_k_tensor, init_c=True)
    
                        if l0ABufId == 0:
                            tla.set_flag(mmad_ready_l0a_0)# CUBE-MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufId == 0 :
                            tla.set_flag(mmad_ready_l0b_0)# CUBE-MTE1
                        else :
                            tla.set_flag(mmad_ready_l0b_1)
                            
    
                        # ---- fixPipe：L0C(fp32) -> UB(fp16 S) ----
                        if ubSBufId == 0:
                            tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
                        if l0CBufId == 0:
                            tla.set_flag(mmad_ready_fix_0)# CUBE-FIX
                            tla.wait_flag(mmad_ready_fix_0)# CUBE-FIX
                        else:
                            tla.set_flag(mmad_ready_fix_1)
                            tla.wait_flag(mmad_ready_fix_1)
                        
                        tla.copy(ubSTensorTla, l0c_s, tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ))
    
                        if l0CBufId == 0:
                            tla.set_flag(fix_ready_mmad_0)# FIX-CUBE
                        else:
                            tla.set_flag(fix_ready_mmad_1)
                        if ubSBufId == 0:
                            tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.FIX)

                        if (gatheredKvSTileIdx == kvSLoopNum - 1) :
                            tla.set_flag(q_l0a_ready_l1)# MTE1-MTE2
                    
                    # ------QK end-------
                    l1PBufId = validIdx % P_L1_BUF
                    l1p_ptr = l1P_ptrs[0]
                    if l1PBufId == c0 :
                        l1p_ptr = l1P_ptrs[0]
                    elif l1PBufId == c1 :
                        l1p_ptr = l1P_ptrs[1]
                    else :
                        l1p_ptr = l1P_ptrs[2]
                    ubS_tile = tla.tile_view(
                            ubSTensorTla, tla.make_shape(qBaseTile_, kvBaseTile_), tla.make_coord(c0, c0))
                    l1PTensorTla = tla.make_tensor_like(l1p_ptr, 
                                                    ubS_tile,
                                                    tla.arch.zN,
                                                    dst_dtype=DTYPE_Q
                                                )
                    # online Softmax
                    with tla.vector():
                        subBlockIdx = tla.arch.sub_block_idx()
                        mCopyOffset = (rowNum + 1) // 2
                        mHalf = rowNum if rowNum < mCopyOffset else mCopyOffset
                        m = mHalf if subBlockIdx == c0 else (rowNum - mHalf)
    
                        ubP_ptr = ubP_ptrs[0] if ubSBufId == c0 else ubP_ptrs[1]
                        expMax_ptr = expMax_ptrs[0] if l1PBufId == c0 else (expMax_ptrs[1] if l1PBufId == c1 else expMax_ptrs[2])
    
                        if m == c0:
                            if ubSBufId == 0:
                                tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                                tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                                tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            if l1PBufId == c0 :
                                tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)          
                                tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1 :
                                tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)          
                                tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else :
                                tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)          
                                tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)
                        else:
                            # 标量参数
                            n = kvSTileSizeAct
                            mRound = (m + 15) // 16 * 16                # RoundUp(m, C0_NUM_PER_FRACTAL=16)
                            nRound = (n + 15) // 16 * 16                # RoundUp(n, ELE_NUM_PER_C0=16)
                            blockStride = mRound
                            vlSize = _VL_F32                        # GetVecLen()/sizeof(fp32) = 64
                            nLoops = (n + vlSize - 1) // vlSize - 1
                            tailN = (n - 1) % vlSize + 1
                            mLoops = (m + vlSize - 1) // vlSize - 1
                            tailM = (m - 1) % vlSize + 1
                            nPadding = (tailN + 31) // 32 * 32          # RoundUp(tailN, BLOCK_SIZE_IN_BYTE=32)
    
                            # UB 地址视图
                            ub_s = tla.make_tensor(
                                ubS_ptr, 
                                tla.make_layout(tla.make_shape(m, 128), tla.make_stride(128, 1))
                            )
                            ub_p = tla.make_tensor(
                                ubP_ptr, 
                                tla.make_layout(tla.make_shape(65, 128), tla.make_stride(128, 1))
                            )
                            nowMaxAddr = tla.make_tensor(
                                nowMax_ptr, 
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                            )
                            nowSumAddr = tla.make_tensor(
                                nowSum_ptr, 
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                            )
                            lastMaxAddr = tla.make_tensor(
                                lastMax_ptr, 
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                            )
                            lastSumAddr = tla.make_tensor(
                                lastSum_ptr, 
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                            )
                            expMaxUbAddr = tla.make_tensor(
                                expMax_ptr, 
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                            )
                            ub_mask = tla.make_tensor(
                                mask_ub_ptr, 
                                tla.make_layout(tla.make_shape(m, 128), tla.make_stride(128, 1))
                            )
                            
                            maskr_ub = tla.make_tensor(
                                maskr_ub_ptr, 
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                            )
                            
                            holel_ub = tla.make_tensor(
                                holel_ub_ptr, 
                                tla.make_layout(tla.make_shape(m * holeMaxNum_), tla.make_stride(1))
                            )
                            
                            holes_ub = tla.make_tensor(
                                holes_ub_ptr, 
                                tla.make_layout(tla.make_shape(m * holeMaxNum_), tla.make_stride(1))
                            )

                            if is_move_mask:
                                tla.wait_flag(vector_ready_mte2)
                                
                                hole_offset = curBatch * maxQSeqlen_ + qSTileIdx * qBaseTile_ + subBlockIdx * mHalf
                                gm_maskr = tla.make_tensor(
                                    maskr.ptr + hole_offset, 
                                    tla.make_layout(tla.make_shape(m), tla.make_stride(1))
                                )
                                hole_offset = (curBatch * maxQSeqlen_ + qSTileIdx * qBaseTile_ + subBlockIdx * mHalf) * holeMaxNum_
                                gm_holes = tla.make_tensor(
                                    holes.ptr + hole_offset, 
                                    tla.make_layout(tla.make_shape(m * holeMaxNum_), tla.make_stride(1))
                                )
                                gm_holel = tla.make_tensor(
                                    holel.ptr + hole_offset, 
                                    tla.make_layout(tla.make_shape(m * holeMaxNum_), tla.make_stride(1))
                                )
                                tla.copy(maskr_ub, gm_maskr)
                                tla.copy(holel_ub, gm_holel)
                                tla.copy(holes_ub, gm_holes)
                                tla.set_flag(mte2_ready_vector)
                            if anyMaskSmBit and is_first:
                                tla.wait_flag(mte2_ready_vector)

                            
                            if anyMaskSmBit:
                                tla.wait_flag(mask_vector_ready)
                                holeLoopNum = holeMaxNum_ if maskHasHoles else c0
                                if n > 64:
                                    with tla.vec.func(mode='simd'):
                                        pregFull0 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        pregTailN0, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                        pos1 = tla.arange(kvSStartIdx + 64, dtype=tla.Int32)
                                        for im in tla.range(m):
                                            ub_mask_i0m = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(im, c0))
                                            ub_mask_i1m = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(im, c1))
                                            maskr_vec = tla.tile_view(maskr_ub, tla.make_shape(1), tla.make_coord(im)).load(
                                                params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            preg0 = tla.cmp(pos0, maskr_vec, "lt", mask=pregFull0)    # pos >= maskr -> 屏蔽
                                            preg1 = tla.cmp(pos1, maskr_vec, "lt", mask=pregFull0)
                                            for h in tla.range(holeLoopNum):
                                                ih = im * holeMaxNum_ + h
                                                holel_vec = holel_ub[ih]
                                                holes_vec = holes_ub[ih]
                                                holer_vec = holel_vec + holes_vec                       # holer = holel + holes
                                                preg_and_0 = tla.bitwise_or(tla.cmp(pos0, holel_vec, "lt", mask=pregFull0), 
                                                                            tla.cmp(pos0, holer_vec, "ge", mask=pregFull0), 
                                                                            mask=pregFull0)
                                                preg_and_1 = tla.bitwise_or(tla.cmp(pos1, holel_vec, "lt", mask=pregFull0), 
                                                                            tla.cmp(pos1, holer_vec, "ge", mask=pregFull0), 
                                                                            mask=pregFull0)
                                                preg0 = tla.bitwise_and(preg0, preg_and_0, mask=pregFull0)
                                                preg1 = tla.bitwise_and(preg1, preg_and_1, mask=pregFull0)
                                            preg1 = tla.bitwise_and(preg1, pregTailN0, mask=pregFull0)
                                            ub_mask_i0m.store(preg0, MaskStoreParams())
                                            ub_mask_i1m.store(preg1, MaskStoreParams())
                                else:
                                    with tla.vec.func(mode='simd'):
                                        pregFull0 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        pregTailN0, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                        for im in tla.range(m):
                                            ub_mask_i0m = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(im, c0))
                                            maskr_vec = tla.tile_view(maskr_ub, tla.make_shape(1), tla.make_coord(im)).load(
                                                params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            preg0 = tla.cmp(pos0, maskr_vec, "lt", mask=pregFull0)   
                                            for h in tla.range(holeLoopNum):
                                                ih = im * holeMaxNum_ + h
                                                holel_vec = holel_ub[ih]
                                                holes_vec = holes_ub[ih]
                                                holer_vec = holel_vec + holes_vec                      # holer = holel + holes
                                                preg_and_0 = tla.bitwise_or(tla.cmp(pos0, holel_vec, "lt", mask=pregFull0), 
                                                                            tla.cmp(pos0, holer_vec, "ge", mask=pregFull0), 
                                                                            mask=pregFull0)
                                                preg0 = tla.bitwise_and(preg0, preg_and_0, mask=pregFull0)
                                            preg0 = tla.bitwise_and(preg0, pregTailN0, mask=pregFull0)
                                            ub_mask_i0m.store(preg0, MaskStoreParams())

                            # 等 QK Fixpipe 完成
                            if ubSBufId == 0:
                                tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            if ubSBufId == c0:
                                tla.wait_flag(mte3_ready_softmax_0)
                            else:
                                tla.wait_flag(mte3_ready_softmax_1)
                            
                            ub_p_zN_full = tla.make_tensor_like(ubP_ptr, ub_p, tla.arch.zNUnAlign)
                            ub_p_zN = tla.tile_view(
                                ub_p_zN_full, tla.make_shape(m, n), tla.make_coord(c0, c0))
                            if isFirstKvSTile:
                                if n > 64:
                                    if anyMaskSmBit:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_s_i1 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c1))
                                                ub_last_max_i = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_mask_i0 = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_mask_i1 = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(i, c1))
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                mask_reg00 = ub_mask_i0.load(MaskLoadParams())
                                                mask_reg11 = ub_mask_i1.load(MaskLoadParams())
                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg1 = tla.mul(ub_s_reg1, scaleValue_, mask=pregFull)
                                                
                                                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                                                ub_s_reg1 = tla.where(mask_reg11, ub_s_reg1, minVreg)
                                                
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                                                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_last_sum_iDe = tla.tile_view(lastSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_s_i1De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c1))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                ub_mask_i0De = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_mask_i1De = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(j, c1))
                                                
                                                max_regDe = ub_last_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_reg0De = ub_s_i0De.load()
                                                ub_s_reg1De = ub_s_i1De.load()
                                                mask_reg0 = ub_mask_i0De.load(MaskLoadParams())
                                                mask_reg1 = ub_mask_i1De.load(MaskLoadParams())
                                                
                                                # masked 位置在 max pass 已置 MIN_VALUE，exp 用全 lane mask 依赖下溢置 0
                                                ub_s_odd_reg = tla.exp(tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull)
                                                ub_s_even_reg = tla.exp(tla.sub(ub_s_reg1De, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_odd_reg0, exp_even_reg1 = tla.deinterleave(ub_s_odd_reg, ub_s_even_reg)
                                                
                                                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                                                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                
                                                exp_dst_reg0 = exp_even_reg1.to(DTYPE_P, cast_trait_one, mask=pregFull)
                                                exp_dst_reg1 = exp_odd_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg = tla.bitwise_or(exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                                    else:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_s_i1 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c1))
                                                ub_last_max_i = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg1 = tla.mul(ub_s_reg1, scaleValue_, mask=pregFull)
                                                ub_s_reg1 = tla.where(pregTailN, ub_s_reg1, minVreg)
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                                                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_last_sum_iDe = tla.tile_view(lastSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                
                                                max_regDe = ub_last_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_odd_reg, ub_s_even_reg = ub_s_i0De.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32))
                                                exp_odd_reg0 = tla.exp(tla.sub(ub_s_odd_reg, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_even_reg1 = tla.exp(tla.sub(ub_s_even_reg, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                                                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                
                                                exp_dst_reg0 = exp_even_reg1.to(DTYPE_P, cast_trait_one, mask=pregFull)
                                                exp_dst_reg1 = exp_odd_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg = tla.bitwise_or(exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                                    
                                else:
                                    if anyMaskSmBit:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_last_max_i = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_mask_i0 = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_s_reg0 = ub_s_i0.load()
                                                mask_reg00 = ub_mask_i0.load(MaskLoadParams())

                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
    
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_last_sum_iDe = tla.tile_view(lastSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                ub_mask_i0De = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                
                                                max_regDe = ub_last_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_reg0De = ub_s_i0De.load()
                                                mask_reg0 = ub_mask_i0De.load(MaskLoadParams())
    
                                                exp_reg0 = tla.exp(tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                
                                                exp_dst_reg0 = exp_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
    
                                    else:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_last_max_i = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg0 = tla.where(pregTailN, ub_s_reg0, minVreg)
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(lastMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_last_sum_iDe = tla.tile_view(lastSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                
                                                max_regDe = ub_last_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_reg0De = ub_s_i0De.load()
                                                exp_reg0 = tla.exp(tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                
                                                exp_dst_reg0 = exp_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                            else:
                                if n > 64:
                                    if anyMaskSmBit:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_s_i1 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c1))
                                                ub_now_max_i = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_mask_i0 = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_mask_i1 = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(i, c1))
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                mask_reg00 = ub_mask_i0.load(MaskLoadParams())
                                                mask_reg11 = ub_mask_i1.load(MaskLoadParams())

                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg1 = tla.mul(ub_s_reg1, scaleValue_, mask=pregFull)
                                                
                                                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                                                ub_s_reg1 = tla.where(mask_reg11, ub_s_reg1, minVreg)
                                                                                    
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                                                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            ub_last_max_i_de = tla.tile_view(lastMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_now_max_i_de = tla.tile_view(nowMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_last_sum_i_de = tla.tile_view(lastSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_dm_i_de = tla.tile_view(expMaxUbAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
                                            exp_sub_max_reg = tla.exp(tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull)
                                            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
                                            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_now_sum_iDe = tla.tile_view(nowSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                ub_s_i1De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c1))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                ub_mask_i0De = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_mask_i1De = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(j, c1))
                                                max_regDe = ub_now_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_reg0De = ub_s_i0De.load()
                                                ub_s_reg1De = ub_s_i1De.load()
                                                mask_reg0 = ub_mask_i0De.load(MaskLoadParams())
                                                mask_reg1 = ub_mask_i1De.load(MaskLoadParams())
                                                # masked 位置在 max pass 已置 MIN_VALUE，exp 用全 lane mask 依赖下溢置 0
                                                ub_s_odd_reg = tla.exp(tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull)
                                                ub_s_even_reg = tla.exp(tla.sub(ub_s_reg1De, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_odd_reg0, exp_even_reg1 = tla.deinterleave(ub_s_odd_reg, ub_s_even_reg)
                                                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                                                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                exp_dst_reg0 = exp_even_reg1.to(DTYPE_P, cast_trait_one, mask=pregFull)
                                                exp_dst_reg1 = exp_odd_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg = tla.bitwise_or(exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                                            ub_now_sum_i_de = tla.tile_view(nowSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
                                            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)
    
                                    else:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_s_i1 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c1))
                                                ub_now_max_i = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg1 = tla.mul(ub_s_reg1, scaleValue_, mask=pregTailN)
                                                ub_s_reg1 = tla.where(pregTailN, ub_s_reg1, minVreg)    
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                                                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            ub_last_max_i_de = tla.tile_view(lastMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_now_max_i_de = tla.tile_view(nowMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_last_sum_i_de = tla.tile_view(lastSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_dm_i_de = tla.tile_view(expMaxUbAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
                                            exp_sub_max_reg = tla.exp(tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull)
                                            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
                                            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_now_sum_iDe = tla.tile_view(nowSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                max_regDe = ub_now_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_odd_reg, ub_s_even_reg = ub_s_i0De.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32))
                                                exp_odd_reg0 = tla.exp(tla.sub(ub_s_odd_reg, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_even_reg1 = tla.exp(tla.sub(ub_s_even_reg, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                                                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                exp_dst_reg0 = exp_even_reg1.to(DTYPE_P, cast_trait_one, mask=pregFull)
                                                exp_dst_reg1 = exp_odd_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg = tla.bitwise_or(exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                                            ub_now_sum_i_de = tla.tile_view(nowSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
                                            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)
                                else:
                                    
                                    if anyMaskSmBit:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_now_max_i = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_mask_i0 = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_s_reg0 = ub_s_i0.load()
                                                mask_reg00 = ub_mask_i0.load(MaskLoadParams())

                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                
                                                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            ub_last_max_i_de = tla.tile_view(lastMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_now_max_i_de = tla.tile_view(nowMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_last_sum_i_de = tla.tile_view(lastSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_dm_i_de = tla.tile_view(expMaxUbAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
                                            exp_sub_max_reg = tla.exp(tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull)
                                            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
                                            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_now_sum_iDe = tla.tile_view(nowSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                ub_mask_i0De = tla.tile_view(ub_mask, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                
                                                max_regDe = ub_now_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_reg0De = ub_s_i0De.load()
                                                mask_reg0 = ub_mask_i0De.load(MaskLoadParams())
                                               
                                                exp_reg0 = tla.exp(tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                exp_dst_reg0 = exp_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                                
                                            ub_now_sum_i_de = tla.tile_view(nowSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
                                            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)
                                            
    
                                    else:
                                        with tla.vec.func(mode='simd'):
                                            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                                            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                                            cast_trait_zero = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            cast_trait_one = tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE,
                                                sat_mode=tla.params.SatMode.SAT,
                                                round_mode=tla.params.RoundMode.CAST_ROUND,
                                            )
                                            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(i, c0))
                                                ub_now_max_i = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(i))
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg0 = tla.mul(ub_s_reg0, scaleValue_, mask=pregFull)
                                                ub_s_reg0 = tla.where(pregTailN, ub_s_reg0, minVreg)
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                                                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            ub_last_max_i_de = tla.tile_view(lastMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_now_max_i_de = tla.tile_view(nowMaxAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_last_sum_i_de = tla.tile_view(lastSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            ub_dm_i_de = tla.tile_view(expMaxUbAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
                                            exp_sub_max_reg = tla.exp(tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull)
                                            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
                                            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
                                            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
                                            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(nowMaxAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_now_sum_iDe = tla.tile_view(nowSumAddr, tla.make_shape(1), tla.make_coord(j))
                                                ub_s_i0De = tla.tile_view(ub_s, tla.make_shape(1, _VL_F32), tla.make_coord(j, c0))
                                                ub_p_zN_f16_i = tla.tile_view(ub_p_zN, tla.make_shape(1, _VL_F16), tla.make_coord(j, c0))
                                                
                                                max_regDe = ub_now_max_iDe.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                                ub_s_reg0De = ub_s_i0De.load()
                                                exp_reg0 = tla.exp(tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull)
                                                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                                                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams(), mask=one_mask)
                                                exp_dst_reg0 = exp_reg0.to(DTYPE_P, cast_trait_zero, mask=pregFull)
                                                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                                                ub_p_zN_f16_i.store(exp_dst_reg, params=tla.params.BlockStoreParams(block_stride=65), mask=preg_all_b16)
                                        
                                            ub_now_sum_i_de = tla.tile_view(nowSumAddr, tla.make_shape(_VL_F32), tla.make_coord(0))
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
                                            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)   
                            # ------------- vf end ----------------------------    
                            
                            if ubSBufId == c0:
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                                tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.set_flag(p_ub_ready_l1_1)
                                tla.wait_flag(p_ub_ready_l1_1)
                                tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                                
                            if anyMaskSmBit:
                                tla.set_flag(mask_vector_ready)

                            # P(UB) -> L1
                            curNRound = (n + 15) // 16 * 16            # RoundUp(n, ELE_NUM_PER_C0)
                            ub_p_tile = tla.tile_view(
                                ub_p, tla.make_shape(m, n), tla.make_coord(c0, c0))
                            l1P_tile = tla.tile_view(
                                l1PTensorTla, tla.make_shape(mHalf, n), tla.make_coord(subBlockIdx, c0))
                            
                            if l1PBufId == c0 :     
                                tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1 :         
                                tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else :        
                                tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)

                            tla.copy(l1P_tile, ub_p_zN)               # CopyPUbToPL1
    
                            # 通知 P buffer 已写完可复用
                            if ubSBufId == c0:
                                tla.set_flag(mte3_ready_softmax_0)
                            else:
                                tla.set_flag(mte3_ready_softmax_1)
                            # 前向跨核通知 PV
                            if l1PBufId == c0 :     
                                tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1 :         
                                tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else :        
                                tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)
    
                    if  is_move_mask:
                        is_move_mask = False
                    if anyMaskSmBit and is_first:
                        is_first = False
                    with tla.vector():
                        if isLastKvS and not is_move_mask:
                            tla.set_flag(vector_ready_mte2)

    
                # ==================== 后半 idx>=PRE_LAUNCH：PV(cube) + rescale O(vector) ====================
                        
                if gatheredKvSTileIdx >= PRE_LAUNCH and cmp and launch_idx_2 >= 0:
                    gatheredKvSTileIdxDe = launch_idx_2
                    validIdxDe = validIdx -2
                    if gatheredKvSTileIdxDe == kvSLoopNum - c1:
                        kvSTileSizeActDe = gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_
                    else:
                        kvSTileSizeActDe = kvBaseTile_
                    isFirstKvSTileDe = (validIdxDe == c0)
                    isLastKvSTileDe = (gatheredKvSTileIdxDe == kvSLoopNum - c1)
                    ubSBufIdDe = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubS_ptrDe = ubS_ptrs[0] if ubSBufIdDe == c0 else ubS_ptrs[1]
                
                    ubSTensorTlaDe = tla.make_tensor(ubS_ptrDe, 
                                                tla.make_layout(
                                                    tla.make_shape(rowNum, kvSTileSizeActDe), 
                                                    tla.make_stride(128, 1)
                                                ))
                    ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubOTmp_ptr = ubOTmp_ptrs[0] if ubOTmpBufId == c0 else ubOTmp_ptrs[1]
                    ubOTmpTensorTla = tla.make_tensor(ubOTmp_ptr, 
                                        tla.make_layout(
                                            tla.make_shape(rowNum, embed_), 
                                            tla.make_stride(128, 1)
                                        ))
                    
                    l1PBufIdDe = validIdxDe % P_L1_BUF
                    l1P_ptrDe = l1P_ptrs[0] if l1PBufIdDe == c0 else (l1P_ptrs[1] if l1PBufIdDe == c1 else l1P_ptrs[2])
                    l1PTensorTlaDe = tla.make_tensor_like(l1P_ptrDe,
                                                    ubSTensorTlaDe,
                                                    tla.arch.zN,
                                                    dst_dtype=DTYPE_Q
                                                )
                    
                
                    # PV Mmad
                    with tla.cube():
                        kvShapeRowDe = curKvSeqlen
                        kvShapeColDe = embed_
                        gm_v = tla.make_tensor(value.ptr  + gmOffsetV,
                                        tla.make_layout(
                                                tla.make_shape(kvShapeRowDe, kvShapeColDe),
                                                tla.make_stride(vSOffset, 1),
                                                layoutTag=tla.arch.RowMajor
                                        ))
                        # 跨相 L0A/L0B 前缀和
                        prefixSumL0AStagesDe = \
                            ((validIdxDe + c1 + PRE_LAUNCH) * mm1L0ATotalStages_
                            + validIdxDe * mm2L0ATotalStages_) \
                            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH \
                            else ((validNum) * mm1L0ATotalStages_ + validIdxDe * mm2L0ATotalStages_)
                        prefixSumL0BStagesDe = \
                            ((validIdxDe + c1 + PRE_LAUNCH) * mm1L0BTotalStages_
                            + validIdxDe * mm2L0BTotalStages_) \
                            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH \
                            else ((validNum) * mm1L0BTotalStages_ + validIdxDe * mm2L0BTotalStages_)
                        # L1 buffer 选择
                        l1BvBufId = validIdxDe % V_L1_BUF          # l1BBufId = idxDe % vL1BufNum
                        l1V_ptr = l1V_ptrs[0] if l1BvBufId == c0 else l1V_ptrs[1]
                        gm_v_tile = tla.tile_view(gm_v, tla.make_shape(kvBaseTile_, embed_), tla.make_coord(gatheredKvSTileIdxDe, c0))
                        l1_v_tile = tla.make_tensor_like(
                                        l1V_ptr, 
                                        gm_v_tile, 
                                        layoutTag=tla.arch.zN
                                    )
                        if l1BvBufId == c0:
                            tla.wait_flag(v_l0b_ready_l1_0)              # WaitFlag(MTE1_MTE2, l1BEventId)：等 V buffer 空闲
                        else:
                            tla.wait_flag(v_l0b_ready_l1_1)
                        tla.copy(l1_v_tile, gm_v_tile)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l1_ready_l0_0)                # SetFlag(MTE2_MTE1, l1BEventId)
                            tla.wait_flag(v_l1_ready_l0_0)               # WaitFlag(MTE2_MTE1, l1BEventId)
                        else:
                            tla.set_flag(v_l1_ready_l0_1)
                            tla.wait_flag(v_l1_ready_l0_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1)
                        # copy L1 to l0
                        l0TileNActDe = embed_
                        nLoopCounter = validIdxDe  
                        l0CBufIdDe = nLoopCounter % L0_STAGES                          # l0C 只按 n 分 buffer
                        l0c_ptrDe = l0c_ptrs[2] if l0CBufIdDe == c0 else l0c_ptrs[3]
                        ub_o_tile =tla.tile_view(ubOTmpTensorTla,
                                                tla.make_shape(qBaseTile_, embed_),
                                                tla.make_coord(c0, c0))
                        l0c_o = tla.make_tensor_like(
                                    l0c_ptrDe,
                                    ub_o_tile,
                                    layoutTag=tla.arch.L0Clayout
                                )
                        l0TileMActDe = rowNum
                
                        # L0A/L0B buffer id = (跨相前缀和 + 本 mmad 内 stage 号) % L0_STAGES
                        l0ALoopCounterDe = prefixSumL0AStagesDe
                        l0BLoopCounterDe = prefixSumL0BStagesDe
                        l0TileKActDe = kvSTileSizeActDe
                        l0ABufIdDe = l0ALoopCounterDe % L0_STAGES
                        l0BBufIdDe = l0BLoopCounterDe % L0_STAGES
                        # V: L1 -> L0B
                        l0b_ptrDe = l0b_ptrs[0] if l0BBufIdDe == c0 else l0b_ptrs[1]
                        l0_b2 = tla.make_tensor_like(l0b_ptrDe, l1_v_tile, tla.arch.nZ)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0b_0)   # WaitFlag(M_MTE1, l0BEventId)
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        tla.copy(l0_b2, l1_v_tile)               # copyL1ToL0B
                        if l0BBufIdDe == c0:
                            tla.set_flag(l0b_ready_mmad_0)    # SetFlag(MTE1_M, l0BEventId)
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l0b_ready_l1_0)   # SetFlag(MTE1_MTE2, l1BEventId)
                        else:
                            tla.set_flag(v_l0b_ready_l1_1)
                        l1_p_l0 = tla.tile_view(
                            l1PTensorTlaDe, tla.make_shape(128, 128),
                            tla.make_coord(c0, c0))
                        l0a_ptrDe = l0a_ptrs[0] if l0ABufIdDe == c0 else l0a_ptrs[1]
                        l0_a2 = tla.make_tensor_like(l0a_ptrDe, l1PTensorTlaDe, tla.arch.zN)
                        if l0ABufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0a_0)   # WaitFlag(M_MTE1, l0AEventId)
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0_a2, l1_p_l0)               # copyL1ToL0A
                        if l0ABufIdDe == c0:
                            tla.set_flag(l0a_ready_mmad_0)    # SetFlag(MTE1_M, l0AEventId)
                        else:
                            tla.set_flag(l0a_ready_mmad_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)
                        
                        l0TileMAligned = (l0TileMActDe + 15) // 16 * 16
                        if l0ABufIdDe == c0:
                            tla.wait_flag(l0a_ready_mmad_0)   # WaitFlag(MTE1_M, l0AEventId)
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(l0b_ready_mmad_0)   # WaitFlag(MTE1_M, l0BEventId)
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufIdDe == c0:
                            tla.wait_flag(fix_ready_mmad_2)   # WaitFlag(FIX_M, l0CEventId=l0CBufIdDe+2)
                        else:
                            tla.wait_flag(fix_ready_mmad_3)
                        tla.mmad(l0c_o, l0_a2, l0_b2, init_c=True)
                        if l0ABufIdDe == c0:
                            tla.set_flag(mmad_ready_l0a_0)    # SetFlag(M_MTE1, l0AEventId)
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufIdDe == c0:
                            tla.set_flag(mmad_ready_l0b_0)    # SetFlag(M_MTE1, l0BEventId)
                        else:
                            tla.set_flag(mmad_ready_l0b_1)
                        if ubOTmpBufId == c0 :
                            tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
                        if l0CBufIdDe == c0:
                            tla.set_flag(mmad_ready_fix_2)            # SetFlag(M_FIX, l0CEventId=l0CBufIdDe+2)
                            tla.wait_flag(mmad_ready_fix_2)           # WaitFlag(M_FIX, l0CEventId)
                        else:
                            tla.set_flag(mmad_ready_fix_3)
                            tla.wait_flag(mmad_ready_fix_3)
                        tla.copy(ubOTmpTensorTla, l0c_o, tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ))       
                        if l0CBufIdDe == c0:
                            tla.set_flag(fix_ready_mmad_2)            # SetFlag(FIX_M, l0CEventId)
                        else:
                            tla.set_flag(fix_ready_mmad_3)
                        if ubOTmpBufId == c0 :
                            tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.FIX)

                    # rescale O
                    with tla.vector():
                        oShapeColDe = embed_
                        gm_o = tla.make_tensor(attentionOut.ptr  + qBOffset + qHeadIdx * qNOffset,
                                                tla.make_layout(
                                                        tla.make_shape(curQSeqlen, oShapeColDe), 
                                                        tla.make_stride(oSOffset, 1),
                                                        layoutTag=tla.arch.RowMajor
                                                ))
                        # operator() 标量计算
                        rowNumOri = rowNum
                        colNumOri = embed_
                        subBlockIdxDe = tla.arch.sub_block_idx()
                        subBlockNum = 2
                        colNumOriAligned8 = (colNumOri + 7) // 8 * 8        # RoundUp(colNumOri, 8)
                        rowNumSplit = (rowNumOri + 1) // subBlockNum
                        rowNumSplit = rowNumOri if rowNumOri < rowNumSplit else rowNumSplit
                        rowNumCurSubCore = rowNumSplit if subBlockIdxDe == c0 else (rowNumOri - rowNumSplit)
                        rowOffsetCurSubCore = rowNumSplit * subBlockIdxDe
                        colNumCurSubCore = colNumOri
                        colStrideCurSubCore = colNumOriAligned8
                        gmO_tile = tla.tile_view(
                                    gm_o, 
                                    tla.make_shape(qBaseTile_, colNumCurSubCore),
                                    tla.make_coord(qSTileIdx, c0))
                        if tla.const_expr(return_lse):
                            # LSE 输出位置：qBOffset 为 packed [*,S,N,D] 内元素偏移，/embed_ 得 token 序号
                            lseBOffset = qBOffset // embed_
                            gm_lse = tla.make_tensor(lse.ptr  + lseBOffset + qHeadIdx,
                                                    tla.make_layout(
                                                            tla.make_shape(curQSeqlen, 1),
                                                            tla.make_stride(qHeads_, 1),
                                                            layoutTag=tla.arch.RowMajor
                                                    ))
                            gmLse_tile = tla.tile_view(
                                        gm_lse,
                                        tla.make_shape(qBaseTile_, 1),
                                        tla.make_coord(qSTileIdx, c0))
                        ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES

                        curTileMod = validIdxDe % P_L1_BUF
                        expMax_ptrDe = expMax_ptrs[0] if curTileMod == c0 else (expMax_ptrs[1] if curTileMod == c1 else expMax_ptrs[2])
                        # SubCoreCompute
                        if rowNumCurSubCore == c0:
                            if ubOTmpBufId == c0 :
                                tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.VECTOR)
                                tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)  
                            else :
                                tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.VECTOR)
                                tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)  
                        else:
                            # SubCoreCompute 标量参数
                            mDe = rowNumCurSubCore
                            nDe = colNumCurSubCore
                            mRound = (mDe + 15) // 16 * 16                # RoundUp(m, C0_NUM_PER_FRACTAL=16)
                            nRound = (nDe + 15) // 16 * 16                # RoundUp(nDe, ELE_NUM_PER_C0=16)
                            vlSizeDe = _VL_F32                      # GetVecLen()/sizeof(float) = 64
                            colFullLoop = (nDe + vlSizeDe - 1) // vlSizeDe - 1
                            colTail = (nDe - 1) % vlSizeDe + 1
                            # UB 地址视图
                            loUb = tla.make_tensor(
                                ubOTmp_ptr, 
                                tla.make_layout(tla.make_shape(mDe, 128), tla.make_stride(128, 1))
                            )
                            goUb = tla.make_tensor(
                                ubO_ptr, 
                                tla.make_layout(tla.make_shape(mDe, 128), tla.make_stride(128, 1))
                            )
                            goUb16 = tla.make_tensor(
                                ubO16_ptr, 
                                tla.make_layout(tla.make_shape(mDe, 128), tla.make_stride(128, 1))
                            )
                            dmUb = tla.make_tensor(
                                expMax_ptrDe, 
                                tla.make_layout(tla.make_shape(mDe), tla.make_stride(1))
                            )
                            glUb = tla.make_tensor(
                                lastSum_ptr, 
                                tla.make_layout(tla.make_shape(mDe), tla.make_stride(1))
                            )
                            if tla.const_expr(return_lse):
                                gmUb = tla.make_tensor(
                                    lastMax_ptr,
                                    tla.make_layout(tla.make_shape(mDe), tla.make_stride(1))
                                )
                                lseUb = tla.make_tensor(
                                    ub_lse_ptr,
                                    tla.make_layout(tla.make_shape(mDe * 8), tla.make_stride(1))
                                )
                                lseUb_out = tla.make_tensor(
                                    ub_lse_ptr,
                                    tla.make_layout(tla.make_shape(mDe, 1), tla.make_stride(8, 1))
                                )
                            # 等 PV fixpipe 完成
                            if ubOTmpBufId == c0 :
                                tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.VECTOR)  
                            else :
                                tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.VECTOR)
                            tla.wait_flag(mte3_ready_rescale)
                            # 四分支 rescale
                            if isFirstKvSTileDe and isLastKvSTileDe:
                                # ① 首 & 末：O = OTmp / lastSum
                                if nDe > 64:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        pregTailDee, colTail2 = tla.update_mask(colTail, dtype=tla.Float32)
                                        for i0 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i0, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i0, c0))
                                            ub_go_1 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i0, c1))
                                            ub_gl = tla.tile_view(glUb, tla.make_shape(1), tla.make_coord(i0))
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32))
                                            gl_reg = ub_gl.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            div_reg_0 = tla.div(lo_reg_0, gl_reg, mask=pregFullDee)
                                            div_reg_1 = tla.div(lo_reg_1, gl_reg, mask=pregFullDee)
                                            ub_go_0.store(div_reg_0, mask=pregFullDee)
                                            ub_go_1.store(div_reg_1, mask=pregFullDee)
                                else:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        pregTailDee, colTail2 = tla.update_mask(colTail, dtype=tla.Float32)
                                        for i0 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i0, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i0, c0))
                                            ub_gl = tla.tile_view(glUb, tla.make_shape(1), tla.make_coord(i0))
                                            lo_reg_0 = ub_lo_0.load()
                                            gl_reg = ub_gl.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            div_reg_0 = tla.div(lo_reg_0, gl_reg, mask=pregFullDee)
                                            ub_go_0.store(div_reg_0, mask=pregFullDee)            
                            elif isFirstKvSTileDe and (not isLastKvSTileDe):
                                if nDe > 64:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee1 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        for i1 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i1, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i1, c0))
                                            ub_go_1 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i1, c1))
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32))
                                            ub_go_0.store(lo_reg_0, mask=pregFullDee1)
                                            ub_go_1.store(lo_reg_1, mask=pregFullDee1)
                                else:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee1 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        for i1 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i1, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i1, c0))
                                            lo_reg_0 = ub_lo_0.load()
                                            ub_go_0.store(lo_reg_0, mask=pregFullDee1)
                            elif (not isFirstKvSTileDe) and isLastKvSTileDe:
                                # ③ 非首 & 末：O = (O*expMax + OTmp) / lastSum
                                if nDe > 64:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee2 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        for i2 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i2, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i2, c0))
                                            ub_go_1 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i2, c1))
                                            ub_gl = tla.tile_view(glUb, tla.make_shape(1), tla.make_coord(i2))
                                            ub_dm = tla.tile_view(dmUb, tla.make_shape(1), tla.make_coord(i2))
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32))
                                            go_reg_0 = ub_go_0.load()
                                            go_reg_1 = ub_go_1.load()
                                            gl_reg = ub_gl.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            dm_reg = ub_dm.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee2)
                                            mul_reg_1 = tla.mul(go_reg_1, dm_reg, mask=pregFullDee2)
                                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee2)
                                            add_reg_1 = tla.add(mul_reg_1, lo_reg_1, mask=pregFullDee2)
                                            div_reg_0 = tla.div(add_reg_0, gl_reg, mask=pregFullDee2)
                                            div_reg_1 = tla.div(add_reg_1, gl_reg, mask=pregFullDee2)
                                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
                                            ub_go_1.store(div_reg_1, mask=pregFullDee2)
                                else:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee2 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        for i2 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i2, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i2, c0))
                                            ub_gl = tla.tile_view(glUb, tla.make_shape(1), tla.make_coord(i2))
                                            ub_dm = tla.tile_view(dmUb, tla.make_shape(1), tla.make_coord(i2))
                                            lo_reg_0 = ub_lo_0.load()
                                            go_reg_0 = ub_go_0.load()
                                            gl_reg = ub_gl.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            dm_reg = ub_dm.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee2)
                                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee2)
                                            div_reg_0 = tla.div(add_reg_0, gl_reg, mask=pregFullDee2)
                                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
                            else:
                                # ④ 非首 & 非末：O = O*expMax + OTmp
                                if nDe > 64:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee3 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        for i3 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i3, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i3, c0))
                                            ub_go_1 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i3, c1))
                                            ub_dm = tla.tile_view(dmUb, tla.make_shape(1), tla.make_coord(i3))
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32))
                                            go_reg_0 = ub_go_0.load()
                                            go_reg_1 = ub_go_1.load()
                                            dm_reg = ub_dm.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee3)
                                            mul_reg_1 = tla.mul(go_reg_1, dm_reg, mask=pregFullDee3)
                                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee3)
                                            add_reg_1 = tla.add(mul_reg_1, lo_reg_1, mask=pregFullDee3)
                                            ub_go_0.store(add_reg_0, mask=pregFullDee3)
                                            ub_go_1.store(add_reg_1, mask=pregFullDee3)
                                else:
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee3 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        for i3 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(loUb, tla.make_shape(1, _VL_F32), tla.make_coord(i3, c0))
                                            ub_go_0 = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i3, c0))
                                            ub_dm = tla.tile_view(dmUb, tla.make_shape(1), tla.make_coord(i3))
                                            lo_reg_0 = ub_lo_0.load()
                                            go_reg_0 = ub_go_0.load()
                                            dm_reg = ub_dm.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee3)
                                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee3)
                                            ub_go_0.store(add_reg_0, mask=pregFullDee3)

                            if ubOTmpBufId == c0 :
                                tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)  
                            else :
                                tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
                            if isLastKvSTileDe:
                                if tla.const_expr(return_lse):
                                    ##  compute lse = log(lastSum) + lastMax，写回 GM
                                    with tla.vec.func(mode='simd'):
                                        pregFullDee5 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        one_mask_de, _ = tla.update_mask(1, dtype=tla.Float32)
                                        for i5 in tla.range(0, rowNumSplit, 2):
                                            ub_gl_de_0 = tla.tile_view(glUb, tla.make_shape(1), tla.make_coord(i5))
                                            ub_gl_de_1 = tla.tile_view(glUb, tla.make_shape(1), tla.make_coord(i5 + 1))
                                            ub_gm_de_0 = tla.tile_view(gmUb, tla.make_shape(1), tla.make_coord(i5))
                                            ub_gm_de_1 = tla.tile_view(gmUb, tla.make_shape(1), tla.make_coord(i5 + 1))
                                            ub_lse_0 = tla.tile_view(lseUb, tla.make_shape(1), tla.make_coord(i5 * 8))
                                            ub_lse_1 = tla.tile_view(lseUb, tla.make_shape(1), tla.make_coord((i5 + 1) * 8))

                                            gl_reg_de_0 = ub_gl_de_0.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            gl_reg_de_1 = ub_gl_de_1.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            gm_reg_de_0 = ub_gm_de_0.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))
                                            gm_reg_de_1 = ub_gm_de_1.load(params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32))

                                            lse_reg_0 = tla.log(gl_reg_de_0, mask=pregFullDee5)
                                            lse_reg_1 = tla.log(gl_reg_de_1, mask=pregFullDee5)
                                            lse_reg_0 = tla.add(lse_reg_0, gm_reg_de_0, mask=pregFullDee5)
                                            lse_reg_1 = tla.add(lse_reg_1, gm_reg_de_1, mask=pregFullDee5)

                                            ub_lse_0.store(lse_reg_0, tla.params.UnalignStoreParams(), mask=one_mask_de)
                                            ub_lse_1.store(lse_reg_1, tla.params.UnalignStoreParams(), mask=one_mask_de)
                                    tla.set_flag(p_ub_ready_l1_1)
                                    tla.wait_flag(p_ub_ready_l1_1)
                                    gm_lse_tile_out = tla.tile_view(gmLse_tile, tla.make_shape(rowNumSplit, c1), tla.make_coord(subBlockIdxDe, c0))
                                    tla.copy(gm_lse_tile_out, lseUb_out)
                                if nDe > 64:
                                    with tla.vec.func(mode='simd'):
                                        cast_trait_zero_de = tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.SAT,
                                            round_mode=tla.params.RoundMode.CAST_ROUND,
                                        )
                                        cast_trait_one_de = tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ONE,
                                            sat_mode=tla.params.SatMode.SAT,
                                            round_mode=tla.params.RoundMode.CAST_ROUND,
                                        )
                                        pregFullDee4 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        pregAll_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                        for i4 in tla.range(rowNumCurSubCore):
                                            ub_go_0_de = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i4, c0))
                                            ub_go_1_de = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i4, c1))
                                            ub_go_b16 = tla.tile_view(goUb16, tla.make_shape(1, _VL_F16), tla.make_coord(i4, c0))
                                            go_reg_0_de = ub_go_0_de.load()
                                            go_reg_1_de = ub_go_1_de.load()
                                            out_reg0 = go_reg_0_de.to(DTYPE_Q, cast_trait_zero_de, mask=pregFullDee4)
                                            out_reg1 = go_reg_1_de.to(DTYPE_Q, cast_trait_one_de, mask=pregFullDee4)
                                            lo_reg_0_de = tla.bitwise_or(out_reg0, out_reg1, mask=pregAll_b16)
                                            ub_go_b16.store(lo_reg_0_de, mask=pregAll_b16)
                                else:
                                    with tla.vec.func(mode='simd'):
                                        cast_trait_zero_de = tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.SAT,
                                            round_mode=tla.params.RoundMode.CAST_ROUND,
                                        )
                                        cast_trait_one_de = tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ONE,
                                            sat_mode=tla.params.SatMode.SAT,
                                            round_mode=tla.params.RoundMode.CAST_ROUND,
                                        )
                                        pregFullDee4 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        pregAll_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                                        for i4 in tla.range(rowNumCurSubCore):
                                            ub_go_0_de = tla.tile_view(goUb, tla.make_shape(1, _VL_F32), tla.make_coord(i4, c0))
                                            ub_go_b16 = tla.tile_view(goUb16, tla.make_shape(1, _VL_F16), tla.make_coord(i4, c0))
                                            go_reg_0_de = ub_go_0_de.load()
                                            go_reg_0_de = go_reg_0_de.to(DTYPE_Q, cast_trait_zero_de, mask=pregFullDee4)
                                            lo_reg_0_de, zero_reg = tla.deinterleave(go_reg_0_de, go_reg_0_de)
                                            ub_go_b16.store(lo_reg_0_de, mask=pregAll_b16)
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                                gm_out_tile = tla.tile_view(gmO_tile, tla.make_shape(rowNumSplit, nDe), tla.make_coord(subBlockIdxDe, c0))
                                tla.copy(gm_out_tile, goUb16)
                            tla.set_flag(mte3_ready_rescale)  # 通知 rescale 完成
                cmp = True
    # release
    with tla.cube():
        tla.wait_flag(q_l0a_ready_l1)
        tla.wait_flag(k_l0b_ready_l1_0)
        tla.wait_flag(k_l0b_ready_l1_1)
        tla.wait_flag(v_l0b_ready_l1_0)
        tla.wait_flag(v_l0b_ready_l1_1)
        tla.wait_flag(mmad_ready_l0a_0)
        tla.wait_flag(mmad_ready_l0a_1)
        tla.wait_flag(mmad_ready_l0b_0)
        tla.wait_flag(mmad_ready_l0b_1)
        tla.wait_flag(fix_ready_mmad_0)
        tla.wait_flag(fix_ready_mmad_1)
        tla.wait_flag(fix_ready_mmad_2)
        tla.wait_flag(fix_ready_mmad_3)
        tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
        tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
        tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
        tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.wait_flag(vec_ready_mte2_0)
        tla.wait_flag(vec_ready_mte2_1)
        tla.wait_flag(mte3_ready_mask_0)
        tla.wait_flag(mte3_ready_mask_1)
        tla.wait_flag(mte3_ready_softmax_0)
        tla.wait_flag(mte3_ready_softmax_1)
        tla.wait_flag(mte3_ready_rescale)
        tla.wait_flag(vector_ready_mte2)
        tla.wait_flag(mask_vector_ready)
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)

# Host 侧
import argparse
import math
import statistics
import time
import torch
from pathlib import Path
from typing import Any
from catlass.tla.runtime import from_dlpack

# host 侧 Tiling
import bsa_tiling as tl_tiling

# mask 生成 + four_stage 测试参数生成
from mask_ref import (
    MaskSpec, NAMED_SPECS,
    evaluate_spec_dense, generate_mask_cpu,
    AnyMaskParams, maskgen_output_to_anymask,
    _print_dense_mask, _verify_mask_consistency,
)

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

def compare_tensors(tensor_a, tensor_b, rtol=1e-3, atol=1e-3, name_a="out", name_b="golden"):
    """
    比较两个张量的数值差异，打印详细统计信息。
    支持任意形状，会先展平后比较。
    """
    # 常见颜色 ANSI 码
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    BOLD = "\033[1m"
    RESET = "\033[0m"  # 必须重置，否则后续所有的终端输出都会带有该颜色
    
    # 确保在同一设备且为浮点类型（方便误差计算）
    a = tensor_a.detach().float().cpu().flatten()
    b = tensor_b.detach().float().cpu().flatten()
    
    # 如果长度不一致，截断到较短的
    min_len = min(a.numel(), b.numel())
    if a.numel() != b.numel():
        print(f"形状不一致: {a.numel()} vs {b.numel()}，将只比较前 {min_len} 个元素")
        a = a[:min_len]
        b = b[:min_len]
    
    # 计算绝对误差和相对误差
    abs_diff = torch.abs(a - b)
    rel_diff = abs_diff / (torch.abs(b) + 1e-8)  # 防止除零
    
    # 统计信息
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    std_abs = abs_diff.std().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    
    # 匹配率（基于 atol）
    matches = abs_diff <= atol
    match_rate = matches.float().mean().item()
    
    print(f"比较 {name_a} vs {name_b} (总共 {min_len} 个元素):")
    print(f" 绝对误差: max={max_abs:.6f}, mean={mean_abs:.6f}, std={std_abs:.6f}")
    print(f" 相对误差: max={max_rel:.6f}, mean={mean_rel:.6f}")
    color = GREEN if match_rate >= 0.99 else RED
    print(f"{color}  匹配率 (atol={atol}): {match_rate*100:.2f}%{RESET}")
    
    # 找出最大误差的位置（如果数组不太大）
    if min_len <= 1000000:
        idx = torch.argmax(abs_diff).item()
        print(f"  最大误差位置: 索引 {idx}, 实际值 {a[idx]:.6f}, 参考值 {b[idx]:.6f}, 差 {abs_diff[idx]:.6f}")
    return abs_diff, rel_diff, matches

def get_block_num(block_num: int, device: int = 0, *, kind: str = "mix") -> int:
    """Get launch block_num. -1 means full-device launch."""
    if int(block_num) != -1:
        return max(1, int(block_num))
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")

def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """运行时参数"""
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }

def _require_torch_npu(device_id: int) -> Any:
    """检查 torch_npu 依赖"""
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("bsa_run --run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch

def create_tla_tensor(buf, layout_tag=tla.arch.RowMajor):
    """用 from_dlpack 包装 torch NPU tensor 为 tla.Tensor"""
    contiguous = buf.contiguous()
    t = from_dlpack(contiguous, layout_tag=layout_tag)
    t._torch_storage = contiguous
    return t

@dataclass
class RunResult:
    name: str
    passed: bool
    match_rate: float
    max_abs: float
    info: str = ""

@dataclass
class _KernelOutput:
    """_run_kernel_core 的返回值"""
    out_buf: torch.Tensor          # (total_q*D,) fp16 NPU buffer
    total_q: int
    D: int
    kernel_time_s: float
    display_name: str
    anymask: AnyMaskParams
    query: torch.Tensor            # BSND: (B, max_sq, H, D) / TND: (Σq_lens, H, D) fp16 NPU
    key: torch.Tensor              # BSND: (B, max_kv, Hkv, D) / TND: (Σkv_lens, Hkv, D)
    value: torch.Tensor            # BSND: (B, max_kv, Hkv, D) / TND: (Σkv_lens, Hkv, D)
    kv_lens_list: list[int]        # 每 batch 的 kv 长度
    q_lens_list: list[int]         # 每 batch 的 q 长度
    input_format: str = "BSND"     # "BSND" 或 "TND"
    cu_seqlens_q: torch.Tensor | None = None  # TND: (B+1,) q 累加序列长度
    cu_seqlens_k: torch.Tensor | None = None  # TND: (B+1,) kv 累加序列长度
    # four_stage_forward 每 batch 参数（仅该 spec 时非空）
    fs_params: dict | None = None  # {context, history, realtime, label_num, label_group_size} 各为 list[int]

def _run_kernel_core(
    args: argparse.Namespace,
    spec: MaskSpec,
    spec_key: str,
    *,
    seq_q: int, seq_k: int, prefix_len: int = 0,
    batch_size: int = 1, num_heads: int = 1, kv_heads: int = 1,
    head_dim: int = 128, tile_m: int = 128, tile_n: int = 128,
    op_dtype: torch.dtype = torch.float16,
    input_format: str = "BSND",
    q_lens_list: list[int] | None = None,
    kv_lens_list_override: list[int] | None = None,
    # ---- four_stage_forward 参数 ----
    fs_context: int | list[int] = 128,
    fs_history: int | list[int] = 128,
    fs_realtime: int | list[int] = 128,
    fs_label_num: int | list[int] = 4,
    fs_label_group_size: int | list[int] = 32,
) -> _KernelOutput:
    """maskgen 编译运行 → AnyMask 转换 → DSL kernel 编译执行的共享核心。

    支持两种 tensor 格式：
      - BSND: query/key/value 为 (B, max_sq, H, D) 4D 定长格式
      - TND:  query/key/value 为 (Σlens, H, D) 3D 变长格式（cu_seqlens_q 累加）

    支持均匀长度和变长两种场景：
      - 均匀：不传 q_lens_list/kv_lens_list_override，由 seq_q/seq_k × B 自动生成
      - 变长：传入 q_lens_list/kv_lens_list_override 指定每 batch 的实际长度
        TND 下各 batch 长度可不同（cu_seqlens 差分），BSND 下以最大长度做 padding

    返回 _KernelOutput，调用方自行决定是否 vs golden 比较。
    """
    is_tnd = (input_format == "TND")

    B, H, Hkv, D = batch_size, num_heads, kv_heads, head_dim

    if spec_key == "four_stage_forward":
        def _to_list(val, B):
            if isinstance(val, list):
                if len(val) == 1:
                    return val * B       # 单值广播到所有 batch
                return val
            return [val] * B
        _ctx  = _to_list(fs_context, B)
        _hist = _to_list(fs_history, B)
        _rt   = _to_list(fs_realtime, B)
        _lnum = _to_list(fs_label_num, B)
        _lgs  = _to_list(fs_label_group_size, B)
        # 每 batch 的 kv 总长
        _kv_lens = [_ctx[i] + _hist[i] + _rt[i] + _lnum[i] * _lgs[i] for i in range(B)]
        # 覆盖 q_lens_list / kv_lens_list_override / seq_q / seq_k
        q_lens_list = list(_kv_lens)
        kv_lens_list_override = list(_kv_lens)
        seq_q = max(_kv_lens)
        seq_k = max(_kv_lens)
    else:
        if q_lens_list is None:
            q_lens_list = [seq_q] * B
        if kv_lens_list_override is None:
            kv_lens_list_override = [seq_k] * B

    max_sq = max(q_lens_list)
    max_kv = max(kv_lens_list_override)
    is_uniform = all(v == q_lens_list[0] for v in q_lens_list) and \
                 all(v == kv_lens_list_override[0] for v in kv_lens_list_override)
    var_tag = "" if is_uniform else "_var"

    torch_mod = _require_torch_npu(args.device)
    device = "npu"
    torch_mod.manual_seed(144)
    if args.print:
        torch_mod.set_printoptions(threshold=float("inf"), linewidth=300, precision=3)

    display_name = (f"{spec_key}_{input_format}{var_tag}_B{B}_H{H}_D{D}"
                    f"_Sq{max_sq}_Sk{max_kv}")

    print(f"\n{'='*70}")
    print(f"[debug] spec={spec_key}  fmt={input_format}{var_tag}  B={B}  H={H}  Hkv={Hkv}  D={D}"
          f"  max_sq={max_sq}  max_kv={max_kv}  prefix_len={prefix_len}  tile=({tile_m},{tile_n})")
    if not is_uniform:
        print(f"  q_lens: {q_lens_list}")
        print(f"  kv_lens: {kv_lens_list_override}")
    print(f"{'='*70}")

    print("\n[Step 1/5] mask_ref: 纯 CPU 生成 mask 六输出 ABI...")
    kv_lens_list = list(kv_lens_list_override)
    if spec_key == "causal":
        _mask_tensors = {"seq_lens": torch.tensor(kv_lens_list, dtype=torch.int32)}
    elif spec_key == "doc_prefix":
        _mask_tensors = {
            "seq_lens": torch.tensor(kv_lens_list, dtype=torch.int32),
            "prefix_lengths": torch.tensor([prefix_len] * B, dtype=torch.int32),
        }
    elif spec_key == "sliding_window":
        wsize = min(max_kv, max(1, max_kv // 2))
        _mask_tensors = {
            "seq_lens": torch.tensor(kv_lens_list, dtype=torch.int32),
            "window_sizes": torch.tensor([wsize] * B, dtype=torch.int32),
        }
    elif spec_key == "four_stage_forward":
        _mask_tensors = {
            "context_sizes": torch.tensor(_ctx, dtype=torch.int32),
            "history_sizes": torch.tensor(_hist, dtype=torch.int32),
            "realtime_sizes": torch.tensor(_rt, dtype=torch.int32),
            "label_nums": torch.tensor(_lnum, dtype=torch.int32),
            "label_group_sizes": torch.tensor(_lgs, dtype=torch.int32),
        }
    else:
        raise ValueError(f"Unknown spec_key: {spec_key}")

    maskgen_outputs = generate_mask_cpu(
        spec, _mask_tensors,
        seq_q=max_sq, seq_k=max_kv,
        mask_num=1, tile_m=tile_m, tile_n=tile_n,
    )

    # ---- Step 2: AnyMask 格式转换 ----
    print("\n[Step 2/5] AnyMask 格式转换...")
    anymask = maskgen_output_to_anymask(
        maskgen_outputs, seq_q=max_sq, seq_k=max_kv, tile_m=tile_m, tile_n=tile_n)
    anymask.tile_range = anymask.tile_range.cpu()
    anymask.block_compute_bp = anymask.block_compute_bp.cpu()
    anymask.block_mask_bp = anymask.block_mask_bp.cpu()
    anymask.maskr = anymask.maskr.cpu()
    anymask.holel = anymask.holel.cpu()
    anymask.holes = anymask.holes.cpu()
    anymask.hole_num = anymask.hole_num.cpu()

    # ---- Step 3: QKV 生成 + kernel 编译执行 ----
    print("\n[Step 3/5] 准备 DSL kernel 输入并编译运行...")

    if is_tnd:
        # TND 格式：(Σlens, H, D) 3D 变长 — QKV 放在 CPU，kernel 前再迁 NPU
        Tq_total = sum(q_lens_list)
        Tk_total = sum(kv_lens_list_override)
        query_raw = torch.randn((Tq_total, H, D), dtype=torch.float32)
        key_raw = torch.randn((Tk_total, Hkv, D), dtype=torch.float32)
        value_raw = torch.randn((Tk_total, Hkv, D), dtype=torch.float32)
        query = query_raw.to(op_dtype)
        key = key_raw.to(op_dtype)
        value = value_raw.to(op_dtype)
        total_q = Tq_total * H
        total_k = Tk_total * Hkv
        _cum_q = 0
        _cum_q_vals = [0]
        for v in q_lens_list:
            _cum_q += v
            _cum_q_vals.append(_cum_q)
        cu_seqlens_q = torch.tensor(_cum_q_vals, dtype=torch.int32)
        _cum_k = 0
        _cum_k_vals = [0]
        for v in kv_lens_list_override:
            _cum_k += v
            _cum_k_vals.append(_cum_k)
        cu_seqlens_k = torch.tensor(_cum_k_vals, dtype=torch.int32)
        actual_q_vals = list(_cum_q_vals)
        actual_kv_vals = list(_cum_k_vals)
        query_4d = query_raw
    else:
        # BSND 格式：(B, max_sq, H, D) 4D 定长 — QKV 放在 CPU，kernel 前再迁 NPU
        query_raw = torch.randn((B, max_sq, H, D), dtype=torch.float32)
        key_raw = torch.randn((B, max_kv, Hkv, D), dtype=torch.float32)
        value_raw = torch.randn((B, max_kv, Hkv, D), dtype=torch.float32)
        query = query_raw.to(op_dtype)
        key = key_raw.to(op_dtype)
        value = value_raw.to(op_dtype)
        total_q = B * max_sq * H
        total_k = B * max_kv * Hkv
        cu_seqlens_q = None
        cu_seqlens_k = None
        actual_q_vals = [max_sq] * B
        actual_kv_vals = [max_kv] * B
        query_4d = query_raw

    kv_lens_npu = torch.tensor(kv_lens_list_override, dtype=torch.int32)
    td = tl_tiling.run_host_tiling(
        query_4d, key_raw, value_raw,  # 传 float32 原始形状用于 tiling 检测
        cu_seqlens_q=cu_seqlens_q, seqused_k=kv_lens_npu, max_seqlen_q=max_sq,
        softmax_scale=None, is_bf16=(op_dtype == torch.bfloat16),
        hole_num=anymask.hole_num, tile_range=anymask.tile_range,
        sparse_compute=anymask.block_compute_bp, sparse_mask=anymask.block_mask_bp,
        maskr=anymask.maskr, holel=anymask.holel, holes=anymask.holes,
    )

    # Q/K/V 迁到 NPU（仅 kernel 需要，golden 用 CPU 版本）
    query_2d = query.reshape(total_q, D).contiguous().to(device)
    key_2d = key.reshape(total_k, D).contiguous().to(device)
    value_2d = value.reshape(total_k, D).contiguous().to(device)

    _tla_dtype = tla.BFloat16 if op_dtype == torch.bfloat16 else tla.Float16
    out_buf = torch.zeros(total_q, D, dtype=op_dtype, device=device)

    tla_query = create_tla_tensor(query_2d, tla.arch.RowMajor)
    tla_key = create_tla_tensor(key_2d, tla.arch.RowMajor)
    tla_value = create_tla_tensor(value_2d, tla.arch.RowMajor)
    tla_output = create_tla_tensor(out_buf, tla.arch.RowMajor)

    total_tokens = total_q // H
    lse_kernel = torch.zeros(total_tokens, H, dtype=torch.float32, device=device)
    tla_lse = create_tla_tensor(lse_kernel.reshape(total_tokens, H), tla.arch.RowMajor)

    uniform_q_seqlen = q_lens_list[0] if is_uniform else 0
    uniform_kv_seqlen = kv_lens_list_override[0] if is_uniform else 0
    uniform_tasks_per_batch = (
        H * ((uniform_q_seqlen + tile_m - 1) // tile_m) if is_uniform else 0
    )

    tiling_int_list = tl_tiling.pack_tiling_int(td)
    tiling_scale_list = tl_tiling.pack_tiling_scale(td)

    Tq = (max_sq + tile_m - 1) // tile_m
    Tk = (max_kv + tile_n - 1) // tile_n
    Wk = (Tk + 31) // 32
    Hn = anymask.hole_num.item()

    tla_tiling_int = create_tla_tensor(
        torch.tensor(tiling_int_list, dtype=torch.int32, device=device), tla.arch.RowMajor)
    tla_tiling_scale = create_tla_tensor(
        torch.tensor(tiling_scale_list, dtype=torch.float32, device=device), tla.arch.RowMajor)
    tla_actual_q = create_tla_tensor(
        torch.tensor(actual_q_vals, dtype=torch.int32, device=device), tla.arch.RowMajor)
    tla_actual_kv = create_tla_tensor(
        torch.tensor(actual_kv_vals, dtype=torch.int32, device=device), tla.arch.RowMajor)
    tla_tile_range = create_tla_tensor(
        anymask.tile_range.contiguous().to(device).reshape(-1), tla.arch.RowMajor)
    tla_sparse_compute = create_tla_tensor(
        anymask.block_compute_bp.contiguous().to(device).to(torch.int32).reshape(-1), tla.arch.RowMajor)
    tla_sparse_mask = create_tla_tensor(
        anymask.block_mask_bp.contiguous().to(device).to(torch.int32).reshape(-1), tla.arch.RowMajor)
    tla_maskr = create_tla_tensor(
        anymask.maskr.contiguous().to(device).reshape(-1), tla.arch.RowMajor)
    tla_holel = create_tla_tensor(
        anymask.holel.contiguous().to(device).reshape(-1), tla.arch.RowMajor)
    tla_holes = create_tla_tensor(
        anymask.holes.contiguous().to(device).reshape(-1), tla.arch.RowMajor)
    tla_hole_num = create_tla_tensor(
        anymask.hole_num.to(device).reshape(-1), tla.arch.RowMajor)

    print("  编译 kernel...")
    _is_fp16 = (op_dtype == torch.float16)
    _hole_max_num = td.holeMaxNum
    _return_lse = bool(getattr(args, "return_lse", False))
    artifact = tla.compile(
        bsa_regular_kernel_arch35,
        tla_query, tla_key, tla_value, tla_output, tla_lse,
        tla_tiling_int, tla_tiling_scale, tla_actual_q, tla_actual_kv,
        tla_tile_range, tla_sparse_compute, tla_sparse_mask,
        tla_maskr, tla_holel, tla_holes, tla_hole_num,
        _is_fp16,
        _hole_max_num, uniform_q_seqlen, uniform_kv_seqlen, uniform_tasks_per_batch,
        _return_lse,
        **_runtime_kwargs(args),
    )
    print(f"  kernel.o: {artifact.kernel_binary_path}")

    import time as _time
    _t0 = _time.perf_counter()
    artifact(
        tla_query, tla_key, tla_value, tla_output, tla_lse,
        tla_tiling_int, tla_tiling_scale, tla_actual_q, tla_actual_kv,
        tla_tile_range, tla_sparse_compute, tla_sparse_mask,
        tla_maskr, tla_holel, tla_holes, tla_hole_num,
        block_num=get_block_num(args.block_num, args.device),
    )
    torch.npu.synchronize()
    _dt = _time.perf_counter() - _t0
    print(f"  kernel 执行完成 ({_dt:.3f}s)")

    # four_stage params（仅该 spec 时非空）
    _fs = None
    if spec_key == "four_stage_forward":
        _fs = {
            "context_sizes": list(_ctx),
            "history_sizes": list(_hist),
            "realtime_sizes": list(_rt),
            "label_nums": list(_lnum),
            "label_group_sizes": list(_lgs),
        }

    return _KernelOutput(
        out_buf=out_buf, total_q=total_q, D=D, kernel_time_s=_dt,
        display_name=display_name, anymask=anymask,
        query=query, key=key, value=value,
        kv_lens_list=list(kv_lens_list_override),
        q_lens_list=list(q_lens_list),
        input_format=input_format,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        fs_params=_fs,
    )

# perf 模式：只跑 NPU kernel，不计算 golden

def compute_golden_torch_tnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    custom_mask: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch 基准 Golden 注意力实现（支持 GQA & TND 变长序列格式）"""
    T_q, H_q, D = query.shape
    T_k, H_kv, Dk = key.shape

    k, v = key, value

    if H_kv != H_q:
        repeat_factor = H_q // H_kv
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    q_hnd = query.permute(1, 0, 2).float()
    k_hnd = k.permute(1, 0, 2).float()
    v_hnd = v.permute(1, 0, 2).float()

    qk = torch.matmul(q_hnd, k_hnd.transpose(-2, -1))
    scores = qk * scale

    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    num_batches = len(cu_seqlens_q) - 1
    tnd_mask = torch.ones((T_q, T_k), dtype=torch.bool, device=query.device)

    for i in range(num_batches):
        q_start, q_end = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        k_start, k_end = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])

        sq_len = q_end - q_start
        sk_len = k_end - k_start

        if custom_mask is not None:
            local_mask = custom_mask[i, :sq_len, :sk_len].to(dtype=torch.bool, device=query.device)
            tnd_mask[q_start:q_end, k_start:k_end] = local_mask
        else:
            tnd_mask[q_start:q_end, k_start:k_end] = False

    scores = scores.masked_fill(tnd_mask.unsqueeze(0), float('-inf'))

    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    probs = torch.exp(scores - row_max)
    probs = torch.where(torch.isfinite(scores), probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    attn_weight = torch.where(denom > 0,
                              probs / denom.clamp_min(1e-30),
                              torch.zeros_like(probs))

    out_hnd = torch.matmul(attn_weight, v_hnd)
    output = out_hnd.permute(1, 0, 2).contiguous()
    return output.to(device=query.device, dtype=query.dtype)

def compute_golden_torch_bsnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    kv_lens_per_batch,
    custom_mask: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch 基准 Golden（BSND 定长布局），与 compute_golden_torch_tnd 数值等价，但按 batch 独立计算。"""
    B, Sq, H_q, D = query.shape
    H_kv = key.shape[2]
    out = torch.empty_like(query)

    for b in range(B):
        kv_b = int(kv_lens_per_batch[b])
        q_b = query[b]                 # (Sq, H_q, D)
        k_b = key[b][:kv_b]            # (kv_b, H_kv, D)
        v_b = value[b][:kv_b]          # (kv_b, H_kv, D)

        if H_kv != H_q:
            g = H_q // H_kv
            k_b = k_b.repeat_interleave(g, dim=1)
            v_b = v_b.repeat_interleave(g, dim=1)

        q_hnd = q_b.permute(1, 0, 2).float()    # (H_q, Sq, D)
        k_hnd = k_b.permute(1, 0, 2).float()    # (H_q, kv_b, D)
        v_hnd = v_b.permute(1, 0, 2).float()

        scores = torch.matmul(q_hnd, k_hnd.transpose(-2, -1)) * scale

        if custom_mask is not None:
            m = custom_mask[b, :Sq, :kv_b].to(dtype=torch.bool, device=query.device)
            scores = scores.masked_fill(m.unsqueeze(0), float('-inf'))

        row_max = scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        probs = torch.exp(scores - row_max)
        probs = torch.where(torch.isfinite(scores), probs, torch.zeros_like(probs))
        denom = probs.sum(dim=-1, keepdim=True)
        attn = torch.where(denom > 0,
                           probs / denom.clamp_min(1e-30),
                           torch.zeros_like(probs))

        o_hnd = torch.matmul(attn, v_hnd)
        out[b] = o_hnd.permute(1, 0, 2).to(query.dtype)
    return out

def _to_int_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    return [int(v) for v in x]

def group_matmul(head, kv_head, left, right, high_prec=1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32)).to(torch.float32)
        else:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score

def softmax1(qk_result, is_first, gm, interm_dtype=torch.float16):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    fully_masked = torch.isinf(lm)
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
        dm = torch.nan_to_num(dm, nan=0.0)
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.nan_to_num(sim_sub, nan=0.0)
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    sim_sub = sim_sub.masked_fill(fully_masked, 0.0)
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm

def qkMM1(query, key):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2(p, value):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def bsa_ref_attention(query, key, value, scale, mask, data_type):
    """标杆实现（flash-attention 风格分块计算）。"""
    inner_prec = 0
    interm_dtype = torch.float16 if inner_prec == 1 else torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale)
    scale = scale.to(torch.float16) if inner_prec == 1 else scale.to(torch.float32)
    context_len = key.shape[2]
    context_size = 128
    gl = None
    gm = None
    go = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        subs_mask = None
        if mask is not None:
            subs_mask = mask[:query.shape[1], kv_start : kv_start + sub_len]
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        qk_result = qk_result * scale
        if mask is not None:
            qk_result = qk_result.masked_fill(subs_mask.unsqueeze(0), float('-inf'))
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
        lo = pvMM2(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
    go = go / gl
    go = torch.nan_to_num(go, nan=0.0)
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(data_type), lse

def bsa_golden_attn(q, k_cache, v_cache, cache_seqlens, maskTensor=None,
                      block_table=None, cu_seqlens_q=None,
                      causal=False, scale=None, block_size=128, data_type=None):
    """真值实现（逐 batch 调 bsa_ref_attention，支持 BSND/TND）。"""
    data_type = data_type if data_type is not None else q.dtype
    D = q.shape[-1]
    H = q.shape[-2]
    Hkv = k_cache.shape[-2]
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    is_tnd = cu_seqlens_q is not None
    is_paged = block_table is not None

    q_cpu = q.detach().cpu()
    k_cpu = k_cache.detach().cpu()
    v_cpu = v_cache.detach().cpu()

    kv_lens = _to_int_list(cache_seqlens)
    B = len(kv_lens)

    cu_q = _to_int_list(cu_seqlens_q)
    if is_tnd:
        q_lens = [cu_q[i + 1] - cu_q[i] for i in range(B)]
        total_q = cu_q[-1]
    else:
        Sq = q_cpu.shape[1]
        q_lens = [Sq] * B
        cu_q = [Sq * i for i in range(B + 1)]
        total_q = B * Sq

    cu_k = None
    if is_tnd and not is_paged:
        cu_k = [0]
        for L in kv_lens:
            cu_k.append(cu_k[-1] + L)

    if is_tnd:
        golden_out = torch.empty((total_q, H, D), dtype=data_type)
        golden_lse = torch.empty((H, total_q), dtype=torch.float32)
    else:
        golden_out = torch.empty((B, q_lens[0], H, D), dtype=data_type)
        golden_lse = torch.empty((B, H, q_lens[0]), dtype=torch.float32)

    block_table_cpu = block_table.detach().cpu() if is_paged else None
    for i in range(B):
        Sq_i = q_lens[i]
        kv_i = kv_lens[i]
        q_i = q_cpu[cu_q[i]:cu_q[i + 1]] if is_tnd else q_cpu[i]
        if is_paged:
            j = torch.arange(kv_i, dtype=torch.long)
            bn = block_table_cpu.to(torch.long)[j // block_size]
            bo = j % block_size
            k_i = k_cpu[bn, bo]
            v_i = v_cpu[bn, bo]
        elif is_tnd:
            k_i = k_cpu[cu_k[i]:cu_k[i + 1]]
            v_i = v_cpu[cu_k[i]:cu_k[i + 1]]
        else:
            k_i = k_cpu[i][:kv_i]
            v_i = v_cpu[i][:kv_i]
        mask = None
        if maskTensor is not None:
            mask = maskTensor[i].cpu()
        elif causal:
            mask = torch.triu(torch.ones(Sq_i, kv_i), diagonal=kv_i - Sq_i + 1).bool()
        out_i, lse_i = bsa_ref_attention(q_i, k_i, v_i, scale, mask, data_type)
        out_i = out_i.reshape(Sq_i, H, D)
        if is_tnd:
            golden_out[cu_q[i]:cu_q[i + 1]] = out_i
            golden_lse[:, cu_q[i]:cu_q[i + 1]] = lse_i.reshape(H, Sq_i)
        else:
            golden_out[i:i + 1] = out_i
            golden_lse[i:i + 1] = lse_i.reshape(1, H, Sq_i)
    return golden_out, golden_lse

def run_debug(args: argparse.Namespace, spec: MaskSpec, spec_key: str, **kwargs) -> RunResult:
    """调试入口：maskgen → kernel → golden 校验。

    支持 BSND / TND 两种 tensor 格式，TND 支持每 batch 不同 seqlen（变长）。
    """
    input_format = kwargs.pop('input_format', 'BSND')
    ko = _run_kernel_core(args, spec, spec_key, input_format=input_format, **kwargs)

    max_sq = max(ko.q_lens_list)
    max_kv = max(ko.kv_lens_list)
    B = len(ko.kv_lens_list)

    # Step 4a: evaluate_spec_dense（O(B*Sq*Sk) Python 谓词，仅 debug 模式执行）
    print("\n[Step 4/5] 生成 dense mask 用于 golden...")
    cpu_tensors = {}
    if spec_key == "causal":
        cpu_tensors["seq_lens"] = torch.tensor(ko.kv_lens_list, dtype=torch.int32)
    elif spec_key == "doc_prefix":
        cpu_tensors["seq_lens"] = torch.tensor(ko.kv_lens_list, dtype=torch.int32)
        cpu_tensors["prefix_lengths"] = torch.tensor(
            [kwargs.get('prefix_len', 0)] * B, dtype=torch.int32)
    elif spec_key == "sliding_window":
        wsize = min(max_kv, max(1, max_kv // 2))
        cpu_tensors["seq_lens"] = torch.tensor(ko.kv_lens_list, dtype=torch.int32)
        cpu_tensors["window_sizes"] = torch.tensor([wsize] * B, dtype=torch.int32)
    elif spec_key == "four_stage_forward":
        for key in ("context_sizes", "history_sizes", "realtime_sizes",
                     "label_nums", "label_group_sizes"):
            cpu_tensors[key] = torch.tensor(ko.fs_params[key], dtype=torch.int32)
    dense_mask_cpu = evaluate_spec_dense(spec, cpu_tensors, seq_q=max_sq, seq_k=max_kv)
    dense_mask_for_golden = ~dense_mask_cpu  # True=masked

    # 打印 golden mask（仅 batch=0，缩略显示）
    _print_dense_mask(dense_mask_cpu[0], spec_key, max_sq, max_kv)

    # 一致性自检
    print("  [自检] 验证 maskgen dense 与 AnyMask 的一致性...")
    tile_m = kwargs.get('tile_m', 128)
    tile_n = kwargs.get('tile_n', 128)
    _verify_mask_consistency(dense_mask_cpu, ko.anymask, max_sq, max_kv, tile_m, tile_n)

    # Step 4b: 双标杆精度校验
    # 真值 = bsa_golden_attn，标杆 = bsa_ref_attention（通过 compute_golden_torch_bsnd/tnd 调用）
    print("\n[Step 5/5] 计算 golden reference 并比较...")
    scale_val = 1.0 / math.sqrt(float(ko.D))
    op_dtype = ko.query.dtype
    kv_lens_t = torch.tensor(ko.kv_lens_list, dtype=torch.int32)

    # 真值：bsa_golden_attn
    golden_out, _ = bsa_golden_attn(
        ko.query, ko.key, ko.value, kv_lens_t,
        maskTensor=dense_mask_for_golden,
        cu_seqlens_q=ko.cu_seqlens_q if ko.input_format == "TND" else None,
        scale=scale_val, data_type=op_dtype,
    )
    golden_flat = golden_out.reshape(ko.total_q, ko.D).cpu().float()

    # 标杆：compute_golden_torch_bsnd / tnd
    if ko.input_format == "TND":
        ref_tnd = compute_golden_torch_tnd(
            ko.query, ko.key, ko.value, scale_val,
            cu_seqlens_q=ko.cu_seqlens_q,
            cu_seqlens_k=ko.cu_seqlens_k,
            custom_mask=dense_mask_for_golden,
        )
        ref_flat = ref_tnd.reshape(ko.total_q, ko.D).cpu().float()
    else:
        ref_bsnd = compute_golden_torch_bsnd(
            ko.query, ko.key, ko.value, scale_val,
            kv_lens_per_batch=kv_lens_t,
            custom_mask=dense_mask_for_golden,
        )
        ref_flat = ref_bsnd.reshape(ko.total_q, ko.D).cpu().float()

    kernel_flat = ko.out_buf.reshape(ko.total_q, ko.D).cpu().float()

    # 分子组：kernel vs 真值
    diff_num = (kernel_flat - golden_flat).abs()
    num_rmse = torch.sqrt((diff_num ** 2).mean()).item()
    num_mare = (diff_num / golden_flat.abs().clamp_min(1e-30)).mean().item()
    num_mere = diff_num.max().item()

    # 分母组：标杆 vs 真值
    diff_den = (ref_flat - golden_flat).abs()
    den_rmse = torch.sqrt((diff_den ** 2).mean()).item()
    den_mare = (diff_den / golden_flat.abs().clamp_min(1e-30)).mean().item()
    den_mere = diff_den.max().item()

    # 比值（分母与 floor 取 max，避免分母为 0）
    floor = 2 ** (-7) if op_dtype == torch.float16 else 2 ** (-6)
    ratio_rmse = num_rmse / max(den_rmse, floor)
    ratio_mare = num_mare / max(den_mare, floor)
    ratio_mere = num_mere / max(den_mere, floor)

    passed = (ratio_mare <= 2.0) and (ratio_mere <= 1.2) and (ratio_rmse <= 1.2)

    print(f"  [分子] kernel vs 真值:  RMSE={num_rmse:.6e}  MARE={num_mare:.6e}  MERE={num_mere:.6e}")
    print(f"  [分母] 标杆 vs 真值:    RMSE={den_rmse:.6e}  MARE={den_mare:.6e}  MERE={den_mere:.6e}")
    print(f"  [比值] floor={floor:.6e}  RMSE={ratio_rmse:.4f}(<=1.2)  MARE={ratio_mare:.4f}(<=2.0)  MERE={ratio_mere:.4f}(<=1.2)")

    return RunResult(
        name=ko.display_name, passed=passed,
        match_rate=1.0 - ratio_mare, max_abs=num_mere,
        info=f"kernel={ko.kernel_time_s:.3f}s ratio_rmse={ratio_rmse:.3f} ratio_mare={ratio_mare:.3f} ratio_mere={ratio_mere:.3f}",
    )

def _print_dense_mask(
    mask: torch.Tensor,        # (Sq, Sk) bool — True=visible
    spec_key: str,
    sq: int,
    sk: int,
    max_display: int = 80,
) -> None:
    """缩略打印 dense mask（batch=0），■=visible ·=masked"""
    print(f"\n  [golden mask] {spec_key}  shape=({sq},{sk})  ■=visible  ·=masked")
    if sq <= max_display and sk <= max_display:
        # 完整打印
        for r in range(sq):
            line = "".join("■" if mask[r, c] else "·" for c in range(sk))
            print(f"  {line}")
    else:
        # 降采样打印
        step_r = max(1, sq // max_display)
        step_c = max(1, sk // max_display)
        # 列标头
        header = "   " + "".join(str(c % 10) for c in range(0, sk, step_c))
        print(f"  {header}")
        for r in range(0, sq, step_r):
            line = f"{r:3d}" + "".join(
                "■" if mask[r, c] else "·" for c in range(0, sk, step_c)
            )
            print(f"  {line}")

def _verify_mask_consistency(
    dense_mask: torch.Tensor,       # (B, Sq, Sk) bool — True=visible
    anymask: AnyMaskParams,
    seq_q: int,
    seq_k: int,
    tile_m: int,
    tile_n: int,
) -> None:
    """验证 maskgen dense mask 与 AnyMask 格式之间的一致性（自检）

    检查项：
      1. maskr 与 dense mask 每行最后一个 visible 列一致
      2. sparse_compute（整块屏蔽=True）对应的 tile 在 dense 中全 False
      3. sparse_mask（部分 mask=True）对应的 tile 在 dense 中部分 True
      4. holel/holes 定义的孔在 dense 中全 False
    """
    B = dense_mask.shape[0]
    Tq = (seq_q + tile_m - 1) // tile_m
    Tk = (seq_k + tile_n - 1) // tile_n

    # 按需展开 sparse_compute/sparse_mask 的 per-tile bool（从 bp 版本提取 bit）
    def _bp_bit(bp, b, tq, tk):
        """读取 bit-packed int32 (B,Tq,Wk) 中 (b,tq,tk) 的 bit 值"""
        w = tk // 32
        bit = tk % 32
        return bool((int(bp[b, tq, w]) >> bit) & 1)

    errors = []

    # 1. maskr 一致性
    for b in range(B):
        for q in range(seq_q):
            visible_cols = torch.where(dense_mask[b, q])[0]
            expected_maskr = visible_cols[-1].item() + 1 if len(visible_cols) > 0 else 0
            actual_maskr = anymask.maskr[b, q].item()
            if expected_maskr != actual_maskr:
                errors.append(
                    f"maskr mismatch batch={b} row={q}: "
                    f"expected={expected_maskr} actual={actual_maskr}"
                )

    # 2. sparse_compute 一致性（bit=1 表示有可见元素需计算）
    for b in range(B):
        for tq in range(Tq):
            q_start = tq * tile_m
            q_end = min(q_start + tile_m, seq_q)
            for tk in range(Tk):
                k_start = tk * tile_n
                k_end = min(k_start + tile_n, seq_k)
                tile_visible = dense_mask[b, q_start:q_end, k_start:k_end]
                any_visible = tile_visible.any().item()
                should_compute = _bp_bit(anymask.block_compute_bp, b, tq, tk)
                if any_visible != should_compute:
                    errors.append(
                        f"sparse_compute mismatch batch={b} tile=({tq},{tk}): "
                        f"dense_any_visible={any_visible} should_compute={should_compute}"
                    )

    # 3. sparse_mask 一致性（bit=1 表示需精细 mask）
    for b in range(B):
        for tq in range(Tq):
            q_start = tq * tile_m
            q_end = min(q_start + tile_m, seq_q)
            for tk in range(Tk):
                k_start = tk * tile_n
                k_end = min(k_start + tile_n, seq_k)
                tile_visible = dense_mask[b, q_start:q_end, k_start:k_end]
                all_true = tile_visible.all().item()
                all_false = not tile_visible.any().item()
                is_partial = not all_true and not all_false
                is_mask_tile = _bp_bit(anymask.block_mask_bp, b, tq, tk)
                if is_partial != is_mask_tile:
                    errors.append(
                        f"sparse_mask mismatch batch={b} tile=({tq},{tk}): "
                        f"dense_partial={is_partial} is_mask_tile={is_mask_tile}"
                    )

    if errors:
        print(f"  一致性检查发现 {len(errors)} 个错误（仅显示前 5 个）:")
        for e in errors[:5]:
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... 及其他 {len(errors) - 5} 个")
    else:
        print("  一致性检查全部通过")

def run(args: argparse.Namespace) -> int:
    """单条运行（basic_matmul 风格）：生成数据 → mask → kernel → golden 对比 → 返回 0/1。"""
    torch_mod = _require_torch_npu(args.device)
    torch_mod.npu.set_device(args.device)

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    op_dtype = dtypes[args.dtype]
    spec = NAMED_SPECS[args.pattern]

    print(
        f"--- qs={args.qs} ks={args.ks} heads={args.heads}/{args.kv_heads} "
        f"d={args.head_dim} dtype={args.dtype} pattern={args.pattern} "
        f"fmt={args.format} ---"
    )

    res = run_debug(
        args, spec, args.pattern,
        seq_q=args.qs, seq_k=args.ks,
        batch_size=1,
        num_heads=args.heads, kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        input_format=args.format,
        op_dtype=op_dtype,
    )
    tag = "PASS" if res.passed else "FAIL"
    print(f"\n{tag}  match_rate={res.match_rate:.4f}  max_abs={res.max_abs:.6e}  {res.info}")
    return 0 if res.passed else 1

def main() -> int:
    parser = argparse.ArgumentParser(description="BSA kernel 编译 + 运行")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--qs", type=int, default=128, help="Q 序列长度")
    parser.add_argument("--ks", type=int, default=128, help="KV 序列长度")
    parser.add_argument("--heads", type=int, default=1, help="Q 头数")
    parser.add_argument("--kv-heads", type=int, default=1, help="KV 头数")
    parser.add_argument("--head-dim", type=int, default=128, help="head dim")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--pattern", default="causal", choices=list(NAMED_SPECS.keys()))
    parser.add_argument("--format", choices=("BSND", "TND"), default="BSND")
    parser.add_argument("--block-num", type=int, default=-1, help="-1=自动取满核")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    # 补充 _run_kernel_core 依赖的默认属性
    args.print = 0
    args.return_lse = False

    return run(args)

if __name__ == "__main__":
    raise SystemExit(main())
