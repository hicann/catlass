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


_VL_F32 = 64
_VL_F16 = 128
FRACTAL_ALIGN = 16


@tla.kernel
def flash_attention_infer_kernel(
    query: tla.Tensor,  # gQ  [S1, D]  fp16（BSND/TND：packed [*,S,N,D] 的单 head 切片）
    key: tla.Tensor,  # gK  [D, S2]  fp16（ColumnMajor：直接给出 K^T）
    value: tla.Tensor,  # gV  [S2, D]  fp16
    attentionOut: tla.Tensor,  # gO  [S1, D]  fp16
    tilingInt: tla.Tensor,  # Tiling 整型包(Int32 1D): FA 21字段 + qFormat/kvFormat
    tilingScale: tla.Tensor,  # Tiling 浮点包（fp32 1D，[scaleValue]）
    actualQseqlen: tla.Tensor,  # gActualQseqlen：TND 为累加 [B+1]，BSND 占位 [B]
    actualKvseqlen: tla.Tensor,  # gActualKvseqlen：TND 非 paged 累加 [B+1]，BSND 逐 batch [B]
    tileRange: tla.Tensor,  # TileMask：[batch, Tq] int32（收缩 KV 循环）
    tileCompute: tla.Tensor,  # TileMask：[batch, Tq, Wk] int32（bit=1 表示该 tile 有可见元素需计算）
    fineMask: tla.Tensor,  # TileMask：[batch, Tq, Wk] int32（bit=1 需精细 mask）
    maskr: tla.Tensor,  # TileMask：[batch, Sq] int32
    blockTable: tla.Tensor,  # PA: [B*maxBlocksPerBatch] int32 块表(物理页号)
    is_fp16: tla.Constexpr[
        bool
    ],  # True=fp16 输入，False=bf16 输入（编译期决定 DTYPE_Q/K/V/P/O）
    uniform_q_seqlen: tla.Constexpr[
        int
    ],  # 定长 Q 每 batch 序列长度（0=非 uniform，走运行时路径）
    uniform_kv_seqlen: tla.Constexpr[int],  # 定长 KV 每 batch 序列长度（0=非 uniform）
    uniform_tasks_per_batch: tla.Constexpr[int],  # 每 batch 的 Q 任务数（0=非 uniform）
    paged: tla.Constexpr[bool],  # PA: kernel 侧块表寻址(cache [numBlocks,Hkv,bs,D])
    max_blocks_per_batch: tla.Constexpr[int],  # PA: 每 batch 块表长度(= ceil(kv/bs))
    head_dim: tla.Constexpr[int],  # head 维度(编译期, 替代 tiling embed 字段)
) -> None:
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

    # TilingData: FA 21字段前缀 + qFormat/kvFormat(2字段)
    _TI_BATCH = 0
    _TI_NUMHEADS = 1
    _TI_KVHEADS = 2
    _TI_MAXQ = 3
    _TI_MAXKV = 4
    _TI_FIRSTTASK = 5
    _TI_TOTALTASK = 6
    _TI_QBASE = 7
    _TI_KVBASE = 8
    _TI_QFORMAT = 21
    _TI_KVFORMAT = 22

    embed_ = head_dim  # 编译期常量(替代 tiling 字段)
    embedV_ = head_dim
    qBaseTile_ = 128
    kvBaseTile_ = 128
    batch_ = tilingInt[_TI_BATCH]
    qHeads_ = tilingInt[_TI_NUMHEADS]
    kvHeads_ = tilingInt[_TI_KVHEADS]
    maxQSeqlen_ = tilingInt[_TI_MAXQ]
    maxKvSeqlen_ = tilingInt[_TI_MAXKV]
    qSeqlen_ = tilingInt[_TI_MAXQ]
    kvSeqlen_ = tilingInt[_TI_MAXKV]
    qFormat_ = tilingInt[_TI_QFORMAT]  # (0=TND, 1=BSND)
    kvFormat_ = tilingInt[_TI_KVFORMAT]
    scaleValue_ = tilingScale[0]

    mm1L1TileN_ = 128
    mm2L1TileN_ = 128

    mm1L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (128 // L0_TILE_K)
    mm1L0BTotalStages_ = (kvBaseTile_ // L0_TILE_N) * (128 // L0_TILE_K)
    mm2L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (kvBaseTile_ // L0_TILE_K)
    mm2L0BTotalStages_ = (kvBaseTile_ // L0_TILE_K) * (128 // L0_TILE_N)

    qSTileNum_ = _ceil_div(maxQSeqlen_, qBaseTile_)  # 首 batch qStile 数
    firstBatchTaskNum_ = tilingInt[_TI_FIRSTTASK]  # qSTileNum_ * qHeads_
    totalTaskNum_ = tilingInt[_TI_TOTALTASK]  # batch_ * firstBatchTaskNum_

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
    q_l1_ready_l0 = tla.flag("q_l1_ready_l0", tla.arch.MTE2, tla.arch.MTE1)
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
    mte3_ready_softmax_0 = tla.flag(
        "mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR
    )
    mte3_ready_softmax_1 = tla.flag(
        "mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR
    )
    # --- rescale
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)

    # --- tilemask
    mutex_maskr = tla.mutex(resource="maskr_ub", id=16)

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

        tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)

    # 片上内存分配
    l1Q_ptrs = [
        tla.allocate(qBaseTile_ * 128, DTYPE_Q, tla.AddressSpace.l1, 512),
    ]
    l1K_ptrs = [
        tla.allocate(128 * kvBaseTile_, DTYPE_K, tla.AddressSpace.l1, 512),
        tla.allocate(128 * kvBaseTile_, DTYPE_K, tla.AddressSpace.l1, 512),
    ]
    l1P_ptrs = [
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
    ]
    l1V_ptrs = [
        tla.allocate(kvBaseTile_ * 128, DTYPE_V, tla.AddressSpace.l1, 512),
        tla.allocate(kvBaseTile_ * 128, DTYPE_V, tla.AddressSpace.l1, 512),
    ]

    l0a_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_K, DTYPE_Q, tla.AddressSpace.l0a, 512),
        tla.allocate(L0_TILE_M * L0_TILE_K, DTYPE_Q, tla.AddressSpace.l0a, 512),
    ]
    l0b_ptrs = [
        tla.allocate(L0_TILE_K * L0_TILE_N, DTYPE_K, tla.AddressSpace.l0b, 512),
        tla.allocate(L0_TILE_K * L0_TILE_N, DTYPE_K, tla.AddressSpace.l0b, 512),
    ]
    l0c_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
    ]

    ubS_ptrs = [
        tla.allocate(qBaseTile_ // 2 * kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2 * kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256),
    ]
    ubP_ptrs = [
        tla.allocate(
            (qBaseTile_ // 2 + 1) * kvBaseTile_, DTYPE_P, tla.AddressSpace.ub, 256
        ),
        tla.allocate(
            (qBaseTile_ // 2 + 1) * kvBaseTile_, DTYPE_P, tla.AddressSpace.ub, 256
        ),
    ]
    ubOTmp_ptrs = [
        tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256),
    ]

    # O 累加器 + 行统计标量
    ubO_ptr = tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256)
    ubO16_ptr = tla.recast_ptr(ubO_ptr, dtype=DTYPE_Q)
    nowMax_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    expMax_ptrs = [
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
    ]
    nowSum_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    lastMax_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    lastSum_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)

    mask_ub_ptr = tla.allocate(
        qBaseTile_ // 2 * 128, tla.Int32, tla.AddressSpace.ub, 256
    )
    maskr_ub_ptr = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)

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

    embedVRound = (embedV_ + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
    groupSize = qHeads_ // kvHeads_

    qBOffset = tla.as_numeric(0)
    kBOffset = tla.as_numeric(0)
    vBOffset = tla.as_numeric(0)
    oBOffset = tla.as_numeric(0)
    preTotalTaskNum = tla.as_numeric(0)
    curBatch = tla.as_numeric(0)

    curQSeqlen = tla.as_numeric(maxQSeqlen_)
    curKvSeqlen = tla.as_numeric(maxKvSeqlen_)
    curTotalTaskNum = firstBatchTaskNum_
    if tla.const_expr(uniform_tasks_per_batch != 0):
        # 定长
        curQSeqlen = tla.as_numeric(uniform_q_seqlen)
        curKvSeqlen = tla.as_numeric(uniform_kv_seqlen)
    else:
        curKvSeqlen = actualKvseqlen[curBatch]
        if qFormat_ == TND:
            curQSeqlen = actualQseqlen[curBatch + 1] - actualQseqlen[curBatch]
        if kvFormat_ == TND:
            curKvSeqlen = actualKvseqlen[curBatch + 1] - actualKvseqlen[curBatch]
    maxQsBlockNum = (maxQSeqlen_ + qBaseTile_ - 1) // qBaseTile_

    task_range = tla.range(tla.arch.block_idx(), totalTaskNum_, tla.arch.block_num())
    for taskSlotIdx in task_range:
        if tla.const_expr(uniform_tasks_per_batch != 0):
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
                curQSeqlen = tla.as_numeric(maxQSeqlen_)
                curKvSeqlen = actualKvseqlen[curBatch]
                if qFormat_ == TND:
                    curQSeqlen = curQSeqlen = (
                        actualQseqlen[curBatch + 1] - actualQseqlen[curBatch]
                    )
                if kvFormat_ == TND:
                    curKvSeqlen = (
                        actualKvseqlen[curBatch + 1] - actualKvseqlen[curBatch]
                    )
                curTotalTaskNum = (
                    curTotalTaskNum + _ceil_div(curQSeqlen, qBaseTile_) * qHeads_
                )
            taskIdxCurBatch = taskSlotIdx - preTotalTaskNum
        logicalQSTileIdx = taskIdxCurBatch // qHeads_
        qHeadIdx = taskIdxCurBatch - logicalQSTileIdx * qHeads_
        qsBlockNum = _ceil_div(curQSeqlen, qBaseTile_)
        evenTileCount = (qsBlockNum + 1) // 2
        qSTileIdx = (
            (logicalQSTileIdx * 2)
            if logicalQSTileIdx < evenTileCount
            else ((qsBlockNum - 1 - logicalQSTileIdx) * 2 + 1)
        )
        kvHeadIdx = qHeadIdx // groupSize
        qSIdex = qSTileIdx * qBaseTile_
        gmOffsetQ = qBOffset + qHeadIdx * qNOffset
        gmOffsetO = oBOffset + qHeadIdx * oNOffset
        gmOffsetK = kBOffset + kvHeadIdx * kNOffset
        gmOffsetV = vBOffset + kvHeadIdx * vNOffset
        rowNum = (
            (curQSeqlen - (qsBlockNum - 1) * qBaseTile_)
            if qSTileIdx == qsBlockNum - 1
            else qBaseTile_
        )
        rowNumRound = (rowNum + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
        kvSTileSizeAct = tla.as_numeric(kvBaseTile_)
        noSkipKvS = curKvSeqlen

        trIdx = curBatch * maxQsBlockNum + qSTileIdx
        tileRangeVal = tileRange[trIdx]
        tileRangeCount = kvBaseTile_ * tileRangeVal
        noSkipKvS = min(tileRangeCount, noSkipKvS)
        kvSLoopNum = (noSkipKvS + kvBaseTile_ - 1) // kvBaseTile_
        if kvSLoopNum > 0:
            gatheredKvSeqlen = noSkipKvS
            kvSTileSizeActDe = tla.as_numeric(kvBaseTile_)

            with tla.cube():
                # loadQGM：整块 Q 常驻 L1，task 内只加载一次。
                gm_q = tla.make_tensor(
                    query.ptr + qBOffset + qHeadIdx * qNOffset,
                    tla.make_layout(
                        tla.make_shape(curQSeqlen, embed_), tla.make_stride(qSOffset, 1)
                    ),
                )
                gmQTensorTla = tla.tile_view(
                    gm_q,
                    tla.make_shape(qBaseTile_, embed_),
                    tla.make_coord(qSTileIdx, c0),
                )
                l1_q = tla.make_tensor_like(l1Q_ptrs[0], gmQTensorTla, tla.arch.zN)
                tla.wait_flag(q_l0a_ready_l1)  # MTE1_MTE2
                tla.copy(l1_q, gmQTensorTla)
                tla.set_flag(q_l1_ready_l0)  # MTE2_MTE1
                tla.wait_flag(q_l1_ready_l0)  # MTE2_MTE1

            # 循环两阶段：idx < kvSLoopNum 做 QK Mmad(cube) + online Softmax(vector)；
            #            idx >= PRE_LAUNCH 做 PV Mmad(cube) + rescale O(vector)
            KvS_range = tla.range(c0, kvSLoopNum + PRE_LAUNCH, c1)
            launch_idx_0 = tla.as_numeric(-1)
            launch_idx_1 = tla.as_numeric(-2)
            launch_idx_2 = tla.as_numeric(-3)
            validIdx = tla.as_numeric(-1)
            validNum = tla.as_numeric(0)
            cmp = tla.as_numeric(True)
            Tk = (kvSeqlen_ + kvBaseTile_ - 1) // kvBaseTile_
            Wk = _ceil_div(Tk, 32)
            fineMaskWordBase = (curBatch * maxQsBlockNum + qSTileIdx) * Wk
            maskHasFine = fineMask[fineMaskWordBase] != c0
            for fineMaskWordIdx in tla.range(c1, Wk, c1):
                if fineMask[fineMaskWordBase + fineMaskWordIdx] != c0:
                    maskHasFine = tla.as_numeric(True)
            is_move_mask = maskHasFine
            is_first = tla.as_numeric(True)
            Tq = (maxQSeqlen_ + qBaseTile_ - 1) // qBaseTile_
            for gatheredKvSTileIdx in KvS_range:
                elemIdx = (
                    curBatch * maxQsBlockNum + qSTileIdx
                ) * Tk + gatheredKvSTileIdx
                if gatheredKvSTileIdx < kvSLoopNum:
                    computeWordIdx = (
                        curBatch * maxQsBlockNum + qSTileIdx
                    ) * Wk + gatheredKvSTileIdx // 32
                    computeBitPos = gatheredKvSTileIdx % 32
                    cmp = ((tileCompute[computeWordIdx] >> computeBitPos) & 1) == 1
                if cmp:
                    launch_idx_2 = launch_idx_1
                    launch_idx_1 = launch_idx_0
                    launch_idx_0 = gatheredKvSTileIdx
                    validIdx = validIdx + 1
                # ==================== 前半 idx<kvSLoopNum：QK(cube) + Softmax(vector) ====================
                if gatheredKvSTileIdx < kvSLoopNum and cmp:
                    validNum = validNum + 1
                    if gatheredKvSTileIdx == kvSLoopNum - c1:
                        kvSTileSizeAct = (
                            gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_
                        )
                    else:
                        kvSTileSizeAct = tla.as_numeric(kvBaseTile_)
                    isFirstKvSTile = validIdx == c0
                    isLastKvS = gatheredKvSTileIdx == kvSLoopNum - c1
                    ubSBufId = validIdx % UB_S_OTMP_BUF_STAGES
                    ubS_ptr = ubS_ptrs[0] if ubSBufId == c0 else ubS_ptrs[1]

                    kvSStartIdx = gatheredKvSTileIdx * kvBaseTile_
                    # TileMask 位图三分支判定
                    wordIdx = (
                        curBatch * maxQsBlockNum + qSTileIdx
                    ) * Wk + gatheredKvSTileIdx // 32
                    bitPos = gatheredKvSTileIdx % 32
                    fineMaskBit = (fineMask[wordIdx] >> bitPos) & 1 == 1

                    colNumRound = (
                        (kvSTileSizeAct + FRACTAL_ALIGN - 1)
                        // FRACTAL_ALIGN
                        * FRACTAL_ALIGN
                    )
                    ubSTensorTla = tla.make_tensor(
                        ubS_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeAct),
                            tla.make_stride(128, 1),
                        ),
                    )
                    # QK Mmad
                    with tla.cube():
                        gm_q = tla.make_tensor(
                            query.ptr + qBOffset + qHeadIdx * qNOffset,
                            tla.make_layout(
                                tla.make_shape(curQSeqlen, embed_),
                                tla.make_stride(qSOffset, 1),
                            ),
                        )

                        gmQTensorTla = tla.tile_view(
                            gm_q,
                            tla.make_shape(qBaseTile_, embed_),
                            tla.make_coord(qSTileIdx, c0),
                        )
                        l1_q = tla.make_tensor_like(
                            l1Q_ptrs[0], gmQTensorTla, tla.arch.zN
                        )

                        if tla.const_expr(paged):
                            # PA: 每 tile 读块表页号, 直接以物理页为基址
                            phys_k = blockTable[
                                curBatch * max_blocks_per_batch + gatheredKvSTileIdx
                            ]
                            k_tile_base = (
                                key.ptr
                                + (
                                    phys_k * kvHeads_ * kvBaseTile_
                                    + kvHeadIdx * kvBaseTile_
                                )
                                * embed_
                            )
                        else:
                            gm_k = tla.make_tensor(
                                key.ptr + (kBOffset + kvHeadIdx * kNOffset),
                                tla.make_layout(
                                    tla.make_shape(embed_, curKvSeqlen),
                                    tla.make_stride(1, kSOffset),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )

                        prefixSumL0AStages = (
                            (validIdx * mm1L0ATotalStages_)
                            if validIdx <= PRE_LAUNCH
                            else (
                                validIdx * mm1L0ATotalStages_
                                + (validIdx - PRE_LAUNCH) * mm2L0ATotalStages_
                            )
                        )
                        prefixSumL0BStages = (
                            (validIdx * mm1L0BTotalStages_)
                            if validIdx <= PRE_LAUNCH
                            else (
                                validIdx * mm1L0BTotalStages_
                                + (validIdx - PRE_LAUNCH) * mm2L0BTotalStages_
                            )
                        )
                        # -----------------QK-----------------
                        l1TileNAct = kvSTileSizeAct
                        nLoopCounterL1 = validIdx

                        # copy gm_k to L1
                        l1BBufId = nLoopCounterL1 % K_L1_BUF
                        l1K_ptr = l1K_ptrs[0] if l1BBufId == c0 else l1K_ptrs[1]
                        if tla.const_expr(paged):
                            gm_k_tile = tla.make_tensor(
                                k_tile_base,
                                tla.make_layout(
                                    tla.make_shape(embed_, kvBaseTile_),
                                    tla.make_stride(1, embed_),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )
                        else:
                            gm_k_tile = tla.tile_view(
                                gm_k,
                                tla.make_shape(embed_, kvBaseTile_),
                                tla.make_coord(c0, gatheredKvSTileIdx),
                            )
                        l1_k_tile = tla.make_tensor_like(
                            l1K_ptr, gm_k_tile, layoutTag=tla.arch.nZ
                        )
                        if l1BBufId == c0:
                            tla.wait_flag(k_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.wait_flag(k_l0b_ready_l1_1)
                        tla.copy(l1_k_tile, gm_k_tile)
                        if l1BBufId == c0:
                            tla.set_flag(k_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.set_flag(k_l1_ready_l0_1)

                        # copy L1 to l0
                        l0TileNAct = l1TileNAct
                        l0CBufId = (nLoopCounterL1) % L0_STAGES
                        l0c_ptr = l0c_ptrs[0] if l0CBufId == c0 else l0c_ptrs[1]
                        ub_s_tile = tla.tile_view(
                            ubSTensorTla,
                            tla.make_shape(qBaseTile_, kvBaseTile_),
                            tla.make_coord(c0, c0),
                        )
                        l0c_s = tla.make_tensor_like(
                            l0c_ptr, ub_s_tile, layoutTag=tla.arch.L0Clayout
                        )

                        l0ALoopCounter = prefixSumL0AStages
                        l0BLoopCounter = prefixSumL0BStages
                        l0ABufId = l0ALoopCounter % L0_STAGES
                        l0BBufId = l0BLoopCounter % L0_STAGES

                        l0a_ptr = l0a_ptrs[0] if l0ABufId == c0 else l0a_ptrs[1]
                        l0a_q_tensor = tla.make_tensor_like(l0a_ptr, l1_q, tla.arch.zN)

                        if l0ABufId == 0:
                            tla.wait_flag(mmad_ready_l0a_0)  # CUBE_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0a_q_tensor, l1_q)
                        if l0ABufId == 0:
                            tla.set_flag(l0a_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.set_flag(l0a_ready_mmad_1)

                        l0b_ptr = l0b_ptrs[0] if l0BBufId == c0 else l0b_ptrs[1]
                        l0b_k_tensor = tla.make_tensor_like(l0b_ptr, l1_k_tile)
                        if l0BBufId == 0:
                            tla.wait_flag(mmad_ready_l0b_0)  # CUBE_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        if l1BBufId == c0:
                            tla.wait_flag(k_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.wait_flag(k_l1_ready_l0_1)
                        tla.copy(l0b_k_tensor, l1_k_tile)

                        if l0BBufId == 0:
                            tla.set_flag(l0b_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BBufId == 0:
                            tla.set_flag(k_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.set_flag(k_l0b_ready_l1_1)

                        if l0ABufId == 0:
                            tla.wait_flag(l0a_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufId == 0:
                            tla.wait_flag(l0b_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufId == 0:
                            tla.wait_flag(fix_ready_mmad_0)  # FIX_CUBE
                        else:
                            tla.wait_flag(fix_ready_mmad_1)

                        tla.mmad(l0c_s, l0a_q_tensor, l0b_k_tensor, init_c=True)

                        if l0ABufId == 0:
                            tla.set_flag(mmad_ready_l0a_0)  # CUBE_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufId == 0:
                            tla.set_flag(mmad_ready_l0b_0)  # CUBE_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0b_1)

                        # ---- fixPipe：L0C(fp32) -> UB(fp16 S) ----
                        if ubSBufId == 0:
                            tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
                        if l0CBufId == 0:
                            tla.set_flag(mmad_ready_fix_0)  # CUBE-FIX
                            tla.wait_flag(mmad_ready_fix_0)  # CUBE-FIX
                        else:
                            tla.set_flag(mmad_ready_fix_1)
                            tla.wait_flag(mmad_ready_fix_1)

                        tla.copy(
                            ubSTensorTla,
                            l0c_s,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )

                        if l0CBufId == 0:
                            tla.set_flag(fix_ready_mmad_0)  # FIX_CUBE
                        else:
                            tla.set_flag(fix_ready_mmad_1)
                        if ubSBufId == 0:
                            tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.FIX)

                        if gatheredKvSTileIdx == kvSLoopNum - 1:
                            tla.set_flag(q_l0a_ready_l1)  # MTE1_MTE2

                    # ------QK end-------
                    l1PBufId = validIdx % P_L1_BUF
                    l1p_ptr = l1P_ptrs[0]
                    if l1PBufId == c0:
                        l1p_ptr = l1P_ptrs[0]
                    elif l1PBufId == c1:
                        l1p_ptr = l1P_ptrs[1]
                    else:
                        l1p_ptr = l1P_ptrs[2]
                    ubS_tile = tla.tile_view(
                        ubSTensorTla,
                        tla.make_shape(qBaseTile_, kvBaseTile_),
                        tla.make_coord(c0, c0),
                    )
                    l1PTensorTla = tla.make_tensor_like(
                        l1p_ptr,
                        ubS_tile,
                        tla.arch.zN,
                    )
                    # online Softmax
                    with tla.vector():
                        subBlockIdx = tla.arch.sub_block_idx()
                        subIdxEff = subBlockIdx if rowNum > 1 else c0
                        mCopyOffset = (rowNum + 1) // 2
                        mHalf = rowNum if rowNum < mCopyOffset else mCopyOffset
                        m = mHalf if subIdxEff == c0 else (rowNum - mHalf)

                        ubP_ptr = ubP_ptrs[0] if ubSBufId == c0 else ubP_ptrs[1]
                        expMax_ptr = (
                            expMax_ptrs[0]
                            if l1PBufId == c0
                            else (expMax_ptrs[1] if l1PBufId == c1 else expMax_ptrs[2])
                        )

                        if m == c0:
                            if ubSBufId == 0:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_0, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_1, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            if l1PBufId == c0:
                                tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                                tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1:
                                tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                                tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else:
                                tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
                                tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)
                        else:
                            # 标量参数
                            n = kvSTileSizeAct
                            mRound = (
                                (m + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
                            )
                            nRound = (
                                (n + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
                            )
                            blockStride = mRound
                            vlSize = _VL_F32  # GetVecLen()/sizeof(fp32) = 64
                            nLoops = (n + vlSize - 1) // vlSize - 1
                            tailN = (n - 1) % vlSize + 1
                            mLoops = (m + vlSize - 1) // vlSize - 1
                            tailM = (m - 1) % vlSize + 1
                            nPadding = (
                                (tailN + 31) // 32 * 32
                            )  # RoundUp(tailN, BLOCK_SIZE_IN_BYTE=32)

                            # UB 地址视图
                            ub_s = tla.make_tensor(
                                ubS_ptr,
                                tla.make_layout(
                                    tla.make_shape(m, 128), tla.make_stride(128, 1)
                                ),
                            )
                            ub_p = tla.make_tensor(
                                ubP_ptr,
                                tla.make_layout(
                                    tla.make_shape(65, 128), tla.make_stride(128, 1)
                                ),
                            )
                            nowMaxAddr = tla.make_tensor(
                                nowMax_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            nowSumAddr = tla.make_tensor(
                                nowSum_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            lastMaxAddr = tla.make_tensor(
                                lastMax_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            lastSumAddr = tla.make_tensor(
                                lastSum_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            expMaxUbAddr = tla.make_tensor(
                                expMax_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            ub_mask = tla.make_tensor(
                                mask_ub_ptr,
                                tla.make_layout(
                                    tla.make_shape(m, 128), tla.make_stride(128, 1)
                                ),
                            )

                            maskr_ub = tla.make_tensor(
                                maskr_ub_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )

                            if is_move_mask and m != c0:
                                maskr_offset = (
                                    curBatch * maxQSeqlen_
                                    + qSTileIdx * qBaseTile_
                                    + subIdxEff * mHalf
                                )
                                gm_maskr = tla.make_tensor(
                                    maskr.ptr + maskr_offset,
                                    tla.make_layout(
                                        tla.make_shape(m), tla.make_stride(1)
                                    ),
                                )
                                mutex_maskr.lock(pipe=tla.arch.MTE2)
                                tla.copy(maskr_ub, gm_maskr)
                                mutex_maskr.unlock(pipe=tla.arch.MTE2)

                            if fineMaskBit:
                                mutex_maskr.lock(pipe=tla.arch.VECTOR)
                                if n > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFull0 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregTailN0, _ = tla.update_mask(
                                            tailN, dtype=tla.Float32
                                        )
                                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                        pos1 = tla.arange(
                                            kvSStartIdx + 64, dtype=tla.Int32
                                        )
                                        for im in tla.range(m):
                                            ub_mask_i0m = tla.tile_view(
                                                ub_mask,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(im, c0),
                                            )
                                            ub_mask_i1m = tla.tile_view(
                                                ub_mask,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(im, c1),
                                            )
                                            maskr_vec = tla.tile_view(
                                                maskr_ub,
                                                tla.make_shape(1),
                                                tla.make_coord(im),
                                            ).load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            preg0 = tla.cmp(
                                                pos0, maskr_vec, "lt", mask=pregFull0
                                            )  # pos >= maskr -> 屏蔽
                                            preg1 = tla.cmp(
                                                pos1, maskr_vec, "lt", mask=pregFull0
                                            )
                                            preg1 = tla.bitwise_and(
                                                preg1, pregTailN0, mask=pregFull0
                                            )
                                            ub_mask_i0m.store(preg0, MaskStoreParams())
                                            ub_mask_i1m.store(preg1, MaskStoreParams())
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFull0 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregTailN0, _ = tla.update_mask(
                                            tailN, dtype=tla.Float32
                                        )
                                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                        for im in tla.range(m):
                                            ub_mask_i0m = tla.tile_view(
                                                ub_mask,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(im, c0),
                                            )
                                            maskr_vec = tla.tile_view(
                                                maskr_ub,
                                                tla.make_shape(1),
                                                tla.make_coord(im),
                                            ).load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            preg0 = tla.cmp(
                                                pos0, maskr_vec, "lt", mask=pregFull0
                                            )
                                            preg0 = tla.bitwise_and(
                                                preg0, pregTailN0, mask=pregFull0
                                            )
                                            ub_mask_i0m.store(preg0, MaskStoreParams())

                            # 等 QK Fixpipe 完成
                            if ubSBufId == 0:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_0, tla.arch.VECTOR
                                )
                            else:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_1, tla.arch.VECTOR
                                )
                            if ubSBufId == c0:
                                tla.wait_flag(mte3_ready_softmax_0)
                            else:
                                tla.wait_flag(mte3_ready_softmax_1)

                            ub_p_zN_full = tla.make_tensor_like(
                                ubP_ptr, ub_p, tla.arch.zNUnAlign
                            )
                            ub_p_zN = tla.tile_view(
                                ub_p_zN_full,
                                tla.make_shape(m, n),
                                tla.make_coord(c0, c0),
                            )
                            if isFirstKvSTile:
                                if n > 64:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_mask_i1 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg11 = ub_mask_i1.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_reg1 = tla.where(
                                                    mask_reg11, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_s_i1De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i1De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                ub_s_reg1De = ub_s_i1De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg1 = ub_mask_i1De.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_odd_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                ub_s_even_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg1De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_odd_reg0, exp_even_reg1 = (
                                                    tla.deinterleave(
                                                        ub_s_odd_reg, ub_s_even_reg
                                                    )
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.where(
                                                    pregTailN, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )

                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_odd_reg, ub_s_even_reg = (
                                                    ub_s_i0De.load(
                                                        params=tla.params.NormalLoadParams(
                                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                        )
                                                    )
                                                )
                                                exp_odd_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_odd_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_even_reg1 = tla.exp(
                                                    tla.sub(
                                                        ub_s_even_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )

                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                else:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    pregTailN, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                            else:
                                if n > 64:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_mask_i1 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg11 = ub_mask_i1.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_reg1 = tla.where(
                                                    mask_reg11, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_s_i1De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i1De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                ub_s_reg1De = ub_s_i1De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg1 = ub_mask_i1De.load(
                                                    MaskLoadParams()
                                                )
                                                # masked 位置在 max pass 已置 MIN_VALUE，exp 用全 lane mask 依赖下溢置 0
                                                ub_s_odd_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                ub_s_even_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg1De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_odd_reg0, exp_even_reg1 = (
                                                    tla.deinterleave(
                                                        ub_s_odd_reg, ub_s_even_reg
                                                    )
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )
                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregTailN,
                                                )
                                                ub_s_reg1 = tla.where(
                                                    pregTailN, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_odd_reg, ub_s_even_reg = (
                                                    ub_s_i0De.load(
                                                        params=tla.params.NormalLoadParams(
                                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                        )
                                                    )
                                                )
                                                exp_odd_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_odd_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_even_reg1 = tla.exp(
                                                    tla.sub(
                                                        ub_s_even_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )
                                else:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            pos0 = tla.arange(
                                                kvSStartIdx, dtype=tla.Int32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )

                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )

                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )

                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )

                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    pregTailN, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )

                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )

                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )
                            # ------------- vf end ----------------------------
                            if ubSBufId == c0:
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                                tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.set_flag(p_ub_ready_l1_1)
                                tla.wait_flag(p_ub_ready_l1_1)
                                tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            if fineMaskBit:
                                mutex_maskr.unlock(pipe=tla.arch.VECTOR)

                            # P(UB) -> L1
                            curNRound = (
                                (n + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
                            )
                            ub_p_tile = tla.tile_view(
                                ub_p, tla.make_shape(m, n), tla.make_coord(c0, c0)
                            )
                            l1P_tile = tla.tile_view(
                                l1PTensorTla,
                                tla.make_shape(mHalf, n),
                                tla.make_coord(subIdxEff, c0),
                            )

                            if l1PBufId == c0:
                                tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1:
                                tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else:
                                tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)

                            tla.copy(l1P_tile, ub_p_zN)  # CopyPUbToPL1

                            # 通知 P buffer 已写完可复用
                            if ubSBufId == c0:
                                tla.set_flag(mte3_ready_softmax_0)
                            else:
                                tla.set_flag(mte3_ready_softmax_1)
                            # 前向跨核通知 PV
                            if l1PBufId == c0:
                                tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1:
                                tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else:
                                tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)

                    if is_move_mask:
                        is_move_mask = tla.as_numeric(False)
                    if fineMaskBit and is_first:
                        is_first = tla.as_numeric(False)

                # ==================== 后半 idx>=PRE_LAUNCH：PV(cube) + rescale O(vector) ====================
                if gatheredKvSTileIdx >= PRE_LAUNCH and cmp and launch_idx_2 >= 0:
                    gatheredKvSTileIdxDe = launch_idx_2
                    validIdxDe = validIdx - 2
                    if gatheredKvSTileIdxDe == kvSLoopNum - c1:
                        kvSTileSizeActDe = (
                            gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_
                        )
                    else:
                        kvSTileSizeActDe = tla.as_numeric(kvBaseTile_)
                    isFirstKvSTileDe = validIdxDe == c0
                    isLastKvSTileDe = gatheredKvSTileIdxDe == kvSLoopNum - c1
                    ubSBufIdDe = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubS_ptrDe = ubS_ptrs[0] if ubSBufIdDe == c0 else ubS_ptrs[1]

                    ubSTensorTlaDe = tla.make_tensor(
                        ubS_ptrDe,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeActDe),
                            tla.make_stride(128, 1),
                        ),
                    )
                    ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubOTmp_ptr = ubOTmp_ptrs[0] if ubOTmpBufId == c0 else ubOTmp_ptrs[1]
                    ubOTmpTensorTla = tla.make_tensor(
                        ubOTmp_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, embed_), tla.make_stride(128, 1)
                        ),
                    )
                    l1PBufIdDe = validIdxDe % P_L1_BUF
                    l1P_ptrDe = (
                        l1P_ptrs[0]
                        if l1PBufIdDe == c0
                        else (l1P_ptrs[1] if l1PBufIdDe == c1 else l1P_ptrs[2])
                    )
                    l1PTensorTlaDe = tla.make_tensor_like(
                        l1P_ptrDe,
                        ubSTensorTlaDe,
                        tla.arch.zN,
                    )

                    # PV Mmad
                    with tla.cube():
                        kvShapeRowDe = curKvSeqlen
                        kvShapeColDe = tla.as_numeric(embed_)
                        if tla.const_expr(paged):
                            phys_v = blockTable[
                                curBatch * max_blocks_per_batch + gatheredKvSTileIdxDe
                            ]
                            v_tile_base = (
                                value.ptr
                                + (
                                    phys_v * kvHeads_ * kvBaseTile_
                                    + kvHeadIdx * kvBaseTile_
                                )
                                * embed_
                            )
                        else:
                            gm_v = tla.make_tensor(
                                value.ptr + gmOffsetV,
                                tla.make_layout(
                                    tla.make_shape(kvShapeRowDe, kvShapeColDe),
                                    tla.make_stride(vSOffset, 1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                        # 跨相 L0A/L0B 前缀和
                        prefixSumL0AStagesDe = (
                            (
                                (validIdxDe + c1 + PRE_LAUNCH) * mm1L0ATotalStages_
                                + validIdxDe * mm2L0ATotalStages_
                            )
                            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH
                            else (
                                (validNum) * mm1L0ATotalStages_
                                + validIdxDe * mm2L0ATotalStages_
                            )
                        )
                        prefixSumL0BStagesDe = (
                            (
                                (validIdxDe + c1 + PRE_LAUNCH) * mm1L0BTotalStages_
                                + validIdxDe * mm2L0BTotalStages_
                            )
                            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH
                            else (
                                (validNum) * mm1L0BTotalStages_
                                + validIdxDe * mm2L0BTotalStages_
                            )
                        )
                        # L1 buffer 选择
                        l1BvBufId = (
                            validIdxDe % V_L1_BUF
                        )  # l1BBufId = idxDe % vL1BufNum
                        l1V_ptr = l1V_ptrs[0] if l1BvBufId == c0 else l1V_ptrs[1]
                        if tla.const_expr(paged):
                            gm_v_tile = tla.make_tensor(
                                v_tile_base,
                                tla.make_layout(
                                    tla.make_shape(kvBaseTile_, embed_),
                                    tla.make_stride(embed_, 1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                        else:
                            gm_v_tile = tla.tile_view(
                                gm_v,
                                tla.make_shape(kvBaseTile_, embed_),
                                tla.make_coord(gatheredKvSTileIdxDe, c0),
                            )
                        l1_v_tile = tla.make_tensor_like(
                            l1V_ptr, gm_v_tile, layoutTag=tla.arch.zN
                        )
                        if l1BvBufId == c0:
                            tla.wait_flag(v_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.wait_flag(v_l0b_ready_l1_1)
                        tla.copy(l1_v_tile, gm_v_tile)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l1_ready_l0_0)  # MTE2_MTE1
                            tla.wait_flag(v_l1_ready_l0_0)  # MTE2_MTE1
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
                        l0TileNActDe = tla.as_numeric(embed_)
                        nLoopCounter = validIdxDe
                        l0CBufIdDe = nLoopCounter % L0_STAGES  # l0C 只按 n 分 buffer
                        l0c_ptrDe = l0c_ptrs[2] if l0CBufIdDe == c0 else l0c_ptrs[3]
                        ub_o_tile = tla.tile_view(
                            ubOTmpTensorTla,
                            tla.make_shape(qBaseTile_, embed_),
                            tla.make_coord(c0, c0),
                        )
                        l0c_o = tla.make_tensor_like(
                            l0c_ptrDe, ub_o_tile, layoutTag=tla.arch.L0Clayout
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
                            tla.wait_flag(mmad_ready_l0b_0)  # M_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        tla.copy(l0_b2, l1_v_tile)  # copyL1ToL0B
                        if l0BBufIdDe == c0:
                            tla.set_flag(l0b_ready_mmad_0)  # MTE1_M
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.set_flag(v_l0b_ready_l1_1)
                        l1_p_l0 = tla.tile_view(
                            l1PTensorTlaDe,
                            tla.make_shape(128, 128),
                            tla.make_coord(c0, c0),
                        )
                        l0a_ptrDe = l0a_ptrs[0] if l0ABufIdDe == c0 else l0a_ptrs[1]
                        l0_a2 = tla.make_tensor_like(
                            l0a_ptrDe, l1PTensorTlaDe, tla.arch.zN
                        )
                        if l0ABufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0a_0)  # M_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0_a2, l1_p_l0)  # copyL1ToL0A
                        if l0ABufIdDe == c0:
                            tla.set_flag(l0a_ready_mmad_0)  # MTE1_M
                        else:
                            tla.set_flag(l0a_ready_mmad_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

                        l0TileMAligned = (
                            (l0TileMActDe + FRACTAL_ALIGN - 1)
                            // FRACTAL_ALIGN
                            * FRACTAL_ALIGN
                        )
                        if l0ABufIdDe == c0:
                            tla.wait_flag(l0a_ready_mmad_0)  # MTE1_M
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(l0b_ready_mmad_0)  # MTE1_M
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufIdDe == c0:
                            tla.wait_flag(
                                fix_ready_mmad_2
                            )  # WaitFlag(FIX_M, l0CEventId=l0CBufIdDe+2)
                        else:
                            tla.wait_flag(fix_ready_mmad_3)
                        tla.mmad(l0c_o, l0_a2, l0_b2, init_c=True)
                        if l0ABufIdDe == c0:
                            tla.set_flag(mmad_ready_l0a_0)  # M_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufIdDe == c0:
                            tla.set_flag(mmad_ready_l0b_0)  # M_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0b_1)
                        if ubOTmpBufId == c0:
                            tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
                        if l0CBufIdDe == c0:
                            tla.set_flag(
                                mmad_ready_fix_2
                            )  # SetFlag(M_FIX, l0CEventId=l0CBufIdDe+2)
                            tla.wait_flag(mmad_ready_fix_2)  # M_FIX
                        else:
                            tla.set_flag(mmad_ready_fix_3)
                            tla.wait_flag(mmad_ready_fix_3)
                        tla.copy(
                            ubOTmpTensorTla,
                            l0c_o,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        if l0CBufIdDe == c0:
                            tla.set_flag(fix_ready_mmad_2)  # FIX_M
                        else:
                            tla.set_flag(fix_ready_mmad_3)
                        if ubOTmpBufId == c0:
                            tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.FIX)

                    # rescale O
                    with tla.vector():
                        oShapeColDe = tla.as_numeric(embed_)
                        gm_o = tla.make_tensor(
                            attentionOut.ptr + qBOffset + qHeadIdx * qNOffset,
                            tla.make_layout(
                                tla.make_shape(curQSeqlen, oShapeColDe),
                                tla.make_stride(oSOffset, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        # operator() 标量计算
                        rowNumOri = rowNum
                        colNumOri = tla.as_numeric(embed_)
                        subBlockIdxDe = tla.arch.sub_block_idx()
                        subIdxEffDe = subBlockIdxDe if rowNumOri > 1 else c0
                        subBlockNum = tla.as_numeric(2)
                        colNumOriAligned8 = (
                            (colNumOri + 7) // 8 * 8
                        )  # RoundUp(colNumOri, 8)
                        rowNumSplit = (rowNumOri + 1) // subBlockNum
                        rowNumSplit = (
                            rowNumOri if rowNumOri < rowNumSplit else rowNumSplit
                        )
                        rowNumCurSubCore = (
                            rowNumSplit
                            if subIdxEffDe == c0
                            else (rowNumOri - rowNumSplit)
                        )
                        rowOffsetCurSubCore = rowNumSplit * subIdxEffDe
                        colNumCurSubCore = colNumOri
                        colStrideCurSubCore = colNumOriAligned8
                        gmO_tile = tla.tile_view(
                            gm_o,
                            tla.make_shape(qBaseTile_, colNumCurSubCore),
                            tla.make_coord(qSTileIdx, c0),
                        )
                        ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES

                        curTileMod = validIdxDe % P_L1_BUF
                        expMax_ptrDe = (
                            expMax_ptrs[0]
                            if curTileMod == c0
                            else (
                                expMax_ptrs[1] if curTileMod == c1 else expMax_ptrs[2]
                            )
                        )
                        # SubCoreCompute
                        if rowNumCurSubCore == c0:
                            if ubOTmpBufId == c0:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_0, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_1, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
                        else:
                            # SubCoreCompute 标量参数
                            mDe = rowNumCurSubCore
                            nDe = colNumCurSubCore
                            mRound = (
                                (mDe + FRACTAL_ALIGN - 1)
                                // FRACTAL_ALIGN
                                * FRACTAL_ALIGN
                            )
                            nRound = (
                                (nDe + FRACTAL_ALIGN - 1)
                                // FRACTAL_ALIGN
                                * FRACTAL_ALIGN
                            )
                            vlSizeDe = _VL_F32
                            colFullLoop = (nDe + vlSizeDe - 1) // vlSizeDe - 1
                            colTail = (nDe - 1) % vlSizeDe + 1
                            # UB 地址视图
                            loUb = tla.make_tensor(
                                ubOTmp_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe, 128), tla.make_stride(128, 1)
                                ),
                            )
                            goUb = tla.make_tensor(
                                ubO_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe, 128), tla.make_stride(128, 1)
                                ),
                            )
                            goUb16 = tla.make_tensor(
                                ubO16_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe, 128), tla.make_stride(128, 1)
                                ),
                            )
                            dmUb = tla.make_tensor(
                                expMax_ptrDe,
                                tla.make_layout(
                                    tla.make_shape(mDe), tla.make_stride(1)
                                ),
                            )
                            glUb = tla.make_tensor(
                                lastSum_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe), tla.make_stride(1)
                                ),
                            )
                            # 等 PV fixpipe 完成
                            if ubOTmpBufId == c0:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_0, tla.arch.VECTOR
                                )
                            else:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_1, tla.arch.VECTOR
                                )
                            tla.wait_flag(mte3_ready_rescale)
                            # 四分支 rescale
                            if isFirstKvSTileDe and isLastKvSTileDe:
                                # 首 & 末：O = OTmp / lastSum
                                if nDe > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregTailDee, colTail2 = tla.update_mask(
                                            colTail, dtype=tla.Float32
                                        )
                                        for i0 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c1),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i0),
                                            )
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                )
                                            )
                                            gl_reg = ub_gl.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            div_reg_0 = tla.div(
                                                lo_reg_0, gl_reg, mask=pregFullDee
                                            )
                                            div_reg_1 = tla.div(
                                                lo_reg_1, gl_reg, mask=pregFullDee
                                            )
                                            ub_go_0.store(div_reg_0, mask=pregFullDee)
                                            ub_go_1.store(div_reg_1, mask=pregFullDee)
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregTailDee, colTail2 = tla.update_mask(
                                            colTail, dtype=tla.Float32
                                        )
                                        for i0 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i0),
                                            )
                                            lo_reg_0 = ub_lo_0.load()
                                            gl_reg = ub_gl.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            div_reg_0 = tla.div(
                                                lo_reg_0, gl_reg, mask=pregFullDee
                                            )
                                            ub_go_0.store(div_reg_0, mask=pregFullDee)
                            elif isFirstKvSTileDe and (not isLastKvSTileDe):
                                if nDe > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee1 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i1 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c1),
                                            )
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                )
                                            )
                                            ub_go_0.store(lo_reg_0, mask=pregFullDee1)
                                            ub_go_1.store(lo_reg_1, mask=pregFullDee1)
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee1 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i1 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            lo_reg_0 = ub_lo_0.load()
                                            ub_go_0.store(lo_reg_0, mask=pregFullDee1)
                            elif (not isFirstKvSTileDe) and isLastKvSTileDe:
                                # 非首 & 末：O = (O*expMax + OTmp) / lastSum
                                if nDe > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee2 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i2 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c1),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
                                            )
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                )
                                            )
                                            go_reg_0 = ub_go_0.load()
                                            go_reg_1 = ub_go_1.load()
                                            gl_reg = ub_gl.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            dm_reg = ub_dm.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee2
                                            )
                                            mul_reg_1 = tla.mul(
                                                go_reg_1, dm_reg, mask=pregFullDee2
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee2
                                            )
                                            add_reg_1 = tla.add(
                                                mul_reg_1, lo_reg_1, mask=pregFullDee2
                                            )
                                            div_reg_0 = tla.div(
                                                add_reg_0, gl_reg, mask=pregFullDee2
                                            )
                                            div_reg_1 = tla.div(
                                                add_reg_1, gl_reg, mask=pregFullDee2
                                            )
                                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
                                            ub_go_1.store(div_reg_1, mask=pregFullDee2)
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee2 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i2 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
                                            )
                                            lo_reg_0 = ub_lo_0.load()
                                            go_reg_0 = ub_go_0.load()
                                            gl_reg = ub_gl.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            dm_reg = ub_dm.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee2
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee2
                                            )
                                            div_reg_0 = tla.div(
                                                add_reg_0, gl_reg, mask=pregFullDee2
                                            )
                                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
                            else:
                                # 非首 & 非末：O = O*expMax + OTmp
                                if nDe > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee3 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i3 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c1),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i3),
                                            )
                                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                )
                                            )
                                            go_reg_0 = ub_go_0.load()
                                            go_reg_1 = ub_go_1.load()
                                            dm_reg = ub_dm.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee3
                                            )
                                            mul_reg_1 = tla.mul(
                                                go_reg_1, dm_reg, mask=pregFullDee3
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee3
                                            )
                                            add_reg_1 = tla.add(
                                                mul_reg_1, lo_reg_1, mask=pregFullDee3
                                            )
                                            ub_go_0.store(add_reg_0, mask=pregFullDee3)
                                            ub_go_1.store(add_reg_1, mask=pregFullDee3)
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee3 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i3 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i3),
                                            )
                                            lo_reg_0 = ub_lo_0.load()
                                            go_reg_0 = ub_go_0.load()
                                            dm_reg = ub_dm.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee3
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee3
                                            )
                                            ub_go_0.store(add_reg_0, mask=pregFullDee3)

                            if ubOTmpBufId == c0:
                                tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
                            if isLastKvSTileDe:
                                if nDe > 64:
                                    with tla.vec.func(mode="simd"):
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
                                        pregFullDee4 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregAll_b16 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float16
                                        )
                                        for i4 in tla.range(rowNumCurSubCore):
                                            ub_go_0_de = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i4, c0),
                                            )
                                            ub_go_1_de = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i4, c1),
                                            )
                                            ub_go_b16 = tla.tile_view(
                                                goUb16,
                                                tla.make_shape(1, _VL_F16),
                                                tla.make_coord(i4, c0),
                                            )
                                            go_reg_0_de = ub_go_0_de.load()
                                            go_reg_1_de = ub_go_1_de.load()
                                            out_reg0 = go_reg_0_de.to(
                                                DTYPE_Q,
                                                cast_trait_zero_de,
                                                mask=pregFullDee4,
                                            )
                                            out_reg1 = go_reg_1_de.to(
                                                DTYPE_Q,
                                                cast_trait_one_de,
                                                mask=pregFullDee4,
                                            )
                                            lo_reg_0_de = tla.bitwise_or(
                                                out_reg0, out_reg1, mask=pregAll_b16
                                            )
                                            ub_go_b16.store(
                                                lo_reg_0_de, mask=pregAll_b16
                                            )
                                else:
                                    with tla.vec.func(mode="simd"):
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
                                        pregFullDee4 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregAll_b16 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float16
                                        )
                                        for i4 in tla.range(rowNumCurSubCore):
                                            ub_go_0_de = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i4, c0),
                                            )
                                            ub_go_b16 = tla.tile_view(
                                                goUb16,
                                                tla.make_shape(1, _VL_F16),
                                                tla.make_coord(i4, c0),
                                            )
                                            go_reg_0_de = ub_go_0_de.load()
                                            go_reg_0_de = go_reg_0_de.to(
                                                DTYPE_Q,
                                                cast_trait_zero_de,
                                                mask=pregFullDee4,
                                            )
                                            lo_reg_0_de, zero_reg = tla.deinterleave(
                                                go_reg_0_de, go_reg_0_de
                                            )
                                            ub_go_b16.store(
                                                lo_reg_0_de, mask=pregAll_b16
                                            )
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                                gm_out_tile = tla.tile_view(
                                    gmO_tile,
                                    tla.make_shape(rowNumSplit, nDe),
                                    tla.make_coord(subBlockIdxDe, c0),
                                )
                                tla.copy(gm_out_tile, goUb16)
                            tla.set_flag(mte3_ready_rescale)
                cmp = tla.as_numeric(True)
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
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


# Host 侧
import argparse
import time
import torch
from catlass.tla.runtime import from_dlpack


@dataclass
class TileMaskParams:
    """FA kernel 所需的 tile mask 输入格式"""

    tile_range: torch.Tensor  # (B, Tq) int32 — 每 Q tile 的 KV 循环上界
    tile_compute_bp: torch.Tensor  # (B, Tq, Wk) int32 — bit=1: 该 tile 有可见元素
    fine_mask_bp: torch.Tensor  # (B, Tq, Wk) int32 — bit=1: 需行级右边界 mask
    maskr: torch.Tensor  # (B, Sq) int32 — 行右边界(k >= maskr 被 mask)


def generate_fa_mask(mask_mode, batch, seq_q, seq_k, tile_m=128, tile_n=128):
    Tq = (seq_q + tile_m - 1) // tile_m
    Tk = (seq_k + tile_n - 1) // tile_n
    Wk = (Tk + 31) // 32

    if mask_mode == "none":
        maskr_1d = torch.full((seq_q,), seq_k, dtype=torch.int32)
        tile_range = torch.full((batch, Tq), Tk, dtype=torch.int32)
        tile_compute_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        fine_mask_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        for kt in range(Tk):
            tile_compute_bp[:, :, kt // 32] |= 1 << (kt % 32)
        dense = torch.ones((batch, seq_q, seq_k), dtype=torch.bool)
    else:  # causal
        prefix = seq_k - seq_q
        # 行右边界(不含): 列 >= maskr 的被遮蔽
        maskr_1d = (torch.arange(seq_q) + prefix + 1).clamp(max=seq_k).to(torch.int32)
        dense_2d = torch.arange(seq_k).unsqueeze(0) < maskr_1d.unsqueeze(1)  # (Sq, Sk)
        dense = dense_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()

        tile_range = torch.zeros((batch, Tq), dtype=torch.int32)
        tile_compute_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        fine_mask_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        for qt in range(Tq):
            q_start = qt * tile_m
            q_end = min(q_start + tile_m, seq_q)
            # 该 Q tile 的最大可见列(最宽行的右边界)
            max_vis = min(int(maskr_1d[q_end - 1]), seq_k)
            tr = (max_vis + tile_n - 1) // tile_n  # 最后一个有可见元素的 tile
            tile_range[:, qt] = tr
            for kt in range(tr):
                tile_compute_bp[:, qt, kt // 32] |= 1 << (kt % 32)
                # 精细 mask: tile 跨越因果边界(最窄行看不全此 tile)
                k_end = (kt + 1) * tile_n
                min_vis = int(maskr_1d[q_start])  # 最窄行右边界
                if k_end > min_vis:
                    fine_mask_bp[:, qt, kt // 32] |= 1 << (kt % 32)

    maskr = maskr_1d.unsqueeze(0).expand(batch, seq_q).contiguous()
    tm = TileMaskParams(
        tile_range=tile_range,
        tile_compute_bp=tile_compute_bp,
        fine_mask_bp=fine_mask_bp,
        maskr=maskr,
    )
    return tm, dense


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


def _require_torch_npu(device_id: int) -> Any:
    """检查 torch_npu 依赖"""
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("FA run requires torch_npu.") from exc
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
    info: str = ""
    sentinel_info: str = ""


@dataclass
class _KernelOutput:
    """_run_kernel_core 的返回值"""

    out_buf: torch.Tensor  # (total_q*D,) fp16 NPU buffer
    total_q: int
    D: int
    kernel_time_s: float
    display_name: str
    tilemask: TileMaskParams
    query: torch.Tensor  # BSND: (B, max_sq, H, D) / TND: (Σq_lens, H, D) fp16 NPU
    key: torch.Tensor  # BSND: (B, max_kv, Hkv, D) / TND: (Σkv_lens, Hkv, D)
    value: torch.Tensor  # BSND: (B, max_kv, Hkv, D) / TND: (Σkv_lens, Hkv, D)
    kv_lens_list: list[int]  # 每 batch 的 kv 长度
    q_lens_list: list[int]  # 每 batch 的 q 长度
    input_format: str = "BSND"  # "BSND" 或 "TND"
    cu_seqlens_q: torch.Tensor | None = None  # TND: (B+1,) q 累加序列长度
    cu_seqlens_k: torch.Tensor | None = None  # TND: (B+1,) kv 累加序列长度


def _run_kernel_core(
    args: argparse.Namespace,
    mask_mode: str,
    mask_key: str,
    *,
    seq_q: int,
    seq_k: int,
    batch_size: int = 1,
    num_heads: int = 1,
    kv_heads: int = 1,
    head_dim: int = 128,
    tile_m: int = 128,
    tile_n: int = 128,
    op_dtype: torch.dtype = torch.float16,
    input_format: str = "BSND",
    q_lens_list: list[int] | None = None,
    kv_lens_list_override: list[int] | None = None,
) -> _KernelOutput:
    """TileMask 生成 → DSL kernel 编译执行"""
    is_tnd = input_format == "TND"

    B, H, Hkv, D = batch_size, num_heads, kv_heads, head_dim

    if q_lens_list is None:
        q_lens_list = [seq_q] * B
    if kv_lens_list_override is None:
        kv_lens_list_override = [seq_k] * B

    max_sq = max(q_lens_list)
    max_kv = max(kv_lens_list_override)
    is_uniform = all(v == q_lens_list[0] for v in q_lens_list) and all(
        v == kv_lens_list_override[0] for v in kv_lens_list_override
    )
    var_tag = "" if is_uniform else "_var"

    torch_mod = _require_torch_npu(args.device)
    device = "npu"
    torch_mod.manual_seed(144)

    display_name = (
        f"{mask_key}_{input_format}{var_tag}_B{B}_H{H}_D{D}_Sq{max_sq}_Sk{max_kv}"
    )

    if not is_uniform:
        print(f"  q_lens: {q_lens_list}")
        print(f"  kv_lens: {kv_lens_list_override}")

    kv_lens_list = list(kv_lens_list_override)
    if mask_key == "causal" or mask_key == "none":
        _mask_tensors = {"seq_lens": torch.tensor(kv_lens_list, dtype=torch.int32)}
    else:
        raise ValueError(f"Unknown mask_key: {mask_key}")

    tilemask, _ = generate_fa_mask(
        "none" if mask_mode == "none" else "causal",
        batch=B,
        seq_q=max_sq,
        seq_k=max_kv,
        tile_m=tile_m,
        tile_n=tile_n,
    )

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
    _scale_value = 1.0 / (head_dim**0.5)

    # Q/K/V 迁到 NPU（仅 kernel 需要，golden 用 CPU 版本）
    query_2d = query.reshape(total_q, D).contiguous().to(device)
    key_2d = key.reshape(total_k, D).contiguous().to(device)
    value_2d = value.reshape(total_k, D).contiguous().to(device)

    _tla_dtype = tla.BFloat16 if op_dtype == torch.bfloat16 else tla.Float16
    out_buf = torch.full(
        (total_q, D), getattr(args, "sentinel", -7.0), dtype=op_dtype, device=device
    )

    tla_query = create_tla_tensor(query_2d, tla.arch.RowMajor)
    tla_key = create_tla_tensor(key_2d, tla.arch.RowMajor)
    tla_value = create_tla_tensor(value_2d, tla.arch.RowMajor)

    _paged = bool(getattr(args, "paged", False))
    _mnb = 0
    if _paged:
        # ===== PagedAttention: cache [numBlocks, Hkv, bs, D] + 打乱块表; kernel 物理页寻址 =====
        _bs = tile_n
        assert int(getattr(args, "blocksize", _bs)) == _bs, (
            "--blocksize 必须等于 kvBaseTile(tile_n)"
        )
        _is_tnd = key.dim() == 3  # TND: (total_k, Hkv, D) vs BSND: (B, max_kv, Hkv, D)
        _hkv = key.shape[1] if _is_tnd else key.shape[2]
        _cu_k = (
            cu_seqlens_k.tolist() if (_is_tnd and cu_seqlens_k is not None) else None
        )
        B_tot = len(kv_lens_list_override)
        _mnb = (max_kv + _bs - 1) // _bs
        num_blocks = B_tot * _mnb
        if getattr(args, "pa_table", "shuffle") == "identity":
            block_table = (
                torch.arange(num_blocks, dtype=torch.int32).reshape(B_tot, _mnb).clone()
            )
        else:
            _gtab = torch.Generator().manual_seed(7)
            block_table = (
                torch.randperm(num_blocks, generator=_gtab)
                .reshape(B_tot, _mnb)
                .to(torch.int32)
            )
        k_cache = torch.zeros(num_blocks, _hkv, _bs, D, dtype=op_dtype)
        v_cache = torch.zeros_like(k_cache)
        for _b in range(B_tot):
            _sk = int(kv_lens_list_override[_b])
            _koff = (
                int(_cu_k[_b]) if _cu_k else 0
            )  # TND: 该 batch 在扁平张量中的起始偏移
            for _i in range(_mnb):
                _lo, _hi = _i * _bs, min((_i + 1) * _bs, _sk)
                if _is_tnd:
                    # TND: key 是 3D (total_k, Hkv, D), 用偏移切片
                    k_cache[block_table[_b, _i], :, : _hi - _lo, :] = key[
                        _koff + _lo : _koff + _hi
                    ].permute(1, 0, 2)
                    v_cache[block_table[_b, _i], :, : _hi - _lo, :] = value[
                        _koff + _lo : _koff + _hi
                    ].permute(1, 0, 2)
                else:
                    # BSND: key 是 4D (B, max_kv, Hkv, D), 按 batch 索引
                    k_cache[block_table[_b, _i], :, : _hi - _lo, :] = key[
                        _b, _lo:_hi, :, :
                    ].permute(1, 0, 2)
                    v_cache[block_table[_b, _i], :, : _hi - _lo, :] = value[
                        _b, _lo:_hi, :, :
                    ].permute(1, 0, 2)
        tla_key = create_tla_tensor(
            k_cache.reshape(-1, D).contiguous().to(device), tla.arch.RowMajor
        )
        tla_value = create_tla_tensor(
            v_cache.reshape(-1, D).contiguous().to(device), tla.arch.RowMajor
        )
        tla_block_table = create_tla_tensor(
            block_table.reshape(-1).contiguous().to(device), tla.arch.RowMajor
        )
    else:
        tla_block_table = create_tla_tensor(
            torch.zeros(1, dtype=torch.int32, device=device), tla.arch.RowMajor
        )
    tla_output = create_tla_tensor(out_buf, tla.arch.RowMajor)

    uniform_q_seqlen = q_lens_list[0] if is_uniform else 0
    uniform_kv_seqlen = kv_lens_list_override[0] if is_uniform else 0
    uniform_tasks_per_batch = (
        H * ((uniform_q_seqlen + tile_m - 1) // tile_m) if is_uniform else 0
    )

    # tiling: FA 21字段(fa_tiling) + qFormat/kvFormat
    import fa_tiling as _fa_tiling

    _fa_td = _fa_tiling.compute_tiling(
        batch=len(kv_lens_list_override),
        num_heads=num_heads,
        kv_heads=kv_heads,
        q_seqlen_list=list(
            q_lens_list if q_lens_list else [max_sq] * len(kv_lens_list_override)
        ),
        kv_seqlen_list=list(kv_lens_list_override),
        head_dim=head_dim,
        q_base_tile=tile_m,
        kv_base_tile=tile_n,
    )
    tiling_int_list = (
        _fa_tiling.pack_tiling_int(_fa_td)
        + [
            1 if input_format == "BSND" else 0,  # qFormat
            1 if input_format == "BSND" else 0,
        ]  # kvFormat
    )
    tiling_scale_list = [_scale_value]

    Tq = (max_sq + tile_m - 1) // tile_m
    Tk = (max_kv + tile_n - 1) // tile_n
    Wk = (Tk + 31) // 32

    tla_tiling_int = create_tla_tensor(
        torch.tensor(tiling_int_list, dtype=torch.int32, device=device),
        tla.arch.RowMajor,
    )
    tla_tiling_scale = create_tla_tensor(
        torch.tensor(tiling_scale_list, dtype=torch.float32, device=device),
        tla.arch.RowMajor,
    )
    tla_actual_q = create_tla_tensor(
        torch.tensor(actual_q_vals, dtype=torch.int32, device=device), tla.arch.RowMajor
    )
    tla_actual_kv = create_tla_tensor(
        torch.tensor(actual_kv_vals, dtype=torch.int32, device=device),
        tla.arch.RowMajor,
    )
    tla_tile_range = create_tla_tensor(
        tilemask.tile_range.contiguous().to(device).reshape(-1), tla.arch.RowMajor
    )
    tla_tile_compute = create_tla_tensor(
        tilemask.tile_compute_bp.contiguous().to(device).to(torch.int32).reshape(-1),
        tla.arch.RowMajor,
    )
    tla_fine_mask = create_tla_tensor(
        tilemask.fine_mask_bp.contiguous().to(device).to(torch.int32).reshape(-1),
        tla.arch.RowMajor,
    )
    tla_maskr = create_tla_tensor(
        tilemask.maskr.contiguous().to(device).reshape(-1), tla.arch.RowMajor
    )

    _is_fp16 = op_dtype == torch.float16

    artifact = tla.compile(
        flash_attention_infer_kernel,
        tla_query,
        tla_key,
        tla_value,
        tla_output,
        tla_tiling_int,
        tla_tiling_scale,
        tla_actual_q,
        tla_actual_kv,
        tla_tile_range,
        tla_tile_compute,
        tla_fine_mask,
        tla_maskr,
        tla_block_table,
        _is_fp16,
        uniform_q_seqlen,
        uniform_kv_seqlen,
        uniform_tasks_per_batch,
        _paged,
        _mnb,
        head_dim,
        options="--npu-arch 3510",
    )

    import time as _time

    _t0 = _time.perf_counter()
    _rep = max(1, int(getattr(args, "launch_repeat", 1)))
    for _ in range(_rep):
        artifact(
            tla_query,
            tla_key,
            tla_value,
            tla_output,
            tla_tiling_int,
            tla_tiling_scale,
            tla_actual_q,
            tla_actual_kv,
            tla_tile_range,
            tla_tile_compute,
            tla_fine_mask,
            tla_maskr,
            tla_block_table,
            block_num=get_block_num(args.block_num, args.device),
        )
    torch.npu.synchronize()
    _dt = _time.perf_counter() - _t0

    return _KernelOutput(
        out_buf=out_buf,
        total_q=total_q,
        D=D,
        kernel_time_s=_dt,
        display_name=display_name,
        tilemask=tilemask,
        query=query,
        key=key,
        value=value,
        kv_lens_list=list(kv_lens_list_override),
        q_lens_list=list(q_lens_list),
        input_format=input_format,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )


def _fa_style_bm(query, key, value, scale, kv_lens_per_batch, custom_mask=None):
    """标杆: CPU模拟FA"""
    B, Sq, Hq, D = query.shape
    Hkv = key.shape[2]
    dt = query.dtype
    FILL = -3e38  # FA 的 MIN_VALUE
    out = torch.empty_like(query)
    for b in range(B):
        kv_b = int(kv_lens_per_batch[b])
        q_b = query[b].permute(1, 0, 2)  # [Hq, Sq, D] f16
        k_b = key[b][:kv_b].permute(1, 0, 2)
        v_b = value[b][:kv_b].permute(1, 0, 2)
        if Hkv != Hq:
            g = Hq // Hkv
            k_b = k_b.repeat_interleave(g, dim=0)
            v_b = v_b.repeat_interleave(g, dim=0)
        gl = gm = go = None
        for kv_start in range(0, kv_b, 128):
            sub_len = min(128, kv_b - kv_start)
            sub_k = k_b[:, kv_start : kv_start + sub_len, :]
            sub_v = v_b[:, kv_start : kv_start + sub_len, :]
            qk = torch.matmul(q_b, sub_k.transpose(-2, -1)).float() * scale
            if custom_mask is not None:
                m = custom_mask[b, :Sq, kv_start : kv_start + sub_len].bool()
                qk = qk.masked_fill(m.unsqueeze(0), FILL)
            lm = torch.max(qk, dim=-1, keepdim=True)[0]
            if kv_start == 0:
                hm = lm
                dm = torch.zeros_like(lm)
            else:
                hm = torch.maximum(gm, lm)
                dm = gm - hm
            gm = hm
            sim = torch.exp(qk - hm)
            row_sum = torch.sum(sim, dim=-1, keepdim=True)
            p = sim.to(dt)
            lo = torch.matmul(p, sub_v).float()
            if kv_start == 0:
                gl = row_sum
                go = lo
            else:
                dm = torch.exp(dm)
                gl = gl * dm + row_sum
                go = go * dm + lo
        out[b] = torch.nan_to_num(go / gl, nan=0.0).permute(1, 0, 2).to(query.dtype)
    return out


def compute_golden_torch_tnd(
    query, key, value, scale, cu_seqlens_q=None, cu_seqlens_k=None, custom_mask=None
):
    """TND 变长格式 golden: 逐 batch 切片, f32 全量 attention."""
    T_q, H_q, D = query.shape
    T_k, H_kv, _ = key.shape
    k, v = key, value
    if H_kv != H_q:
        g = H_q // H_kv
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    q_hnd = query.permute(1, 0, 2).float()
    k_hnd = k.permute(1, 0, 2).float()
    v_hnd = v.permute(1, 0, 2).float()
    scores = torch.matmul(q_hnd, k_hnd.transpose(-2, -1)) * scale
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    nb = len(cu_seqlens_q) - 1
    tnd_mask = torch.ones((T_q, T_k), dtype=torch.bool, device=query.device)
    for i in range(nb):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        if custom_mask is not None:
            local = custom_mask[i, : qe - qs, : ke - ks].to(
                dtype=torch.bool, device=query.device
            )
            tnd_mask[qs:qe, ks:ke] = local
        else:
            tnd_mask[qs:qe, ks:ke] = False
    scores = scores.masked_fill(tnd_mask.unsqueeze(0), float("-inf"))
    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    probs = torch.exp(scores - row_max)
    probs = torch.where(torch.isfinite(scores), probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    attn = torch.where(
        denom > 0, probs / denom.clamp_min(1e-30), torch.zeros_like(probs)
    )
    out = torch.matmul(attn, v_hnd)
    return out.permute(1, 0, 2).contiguous().to(dtype=query.dtype)


def compute_golden_torch_bsnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    kv_lens_per_batch,
    custom_mask: torch.Tensor = None,
) -> torch.Tensor:
    """基准 Golden（BSND 定长布局），与 compute_golden_torch_tnd 数值等价，但按 batch 独立计算。"""
    B, Sq, H_q, D = query.shape
    H_kv = key.shape[2]
    out = torch.empty_like(query)

    for b in range(B):
        kv_b = int(kv_lens_per_batch[b])
        q_b = query[b]  # (Sq, H_q, D)
        k_b = key[b][:kv_b]  # (kv_b, H_kv, D)
        v_b = value[b][:kv_b]  # (kv_b, H_kv, D)

        if H_kv != H_q:
            g = H_q // H_kv
            k_b = k_b.repeat_interleave(g, dim=1)
            v_b = v_b.repeat_interleave(g, dim=1)

        q_hnd = q_b.permute(1, 0, 2).float()  # (H_q, Sq, D)
        k_hnd = k_b.permute(1, 0, 2).float()  # (H_q, kv_b, D)
        v_hnd = v_b.permute(1, 0, 2).float()

        scores = torch.matmul(q_hnd, k_hnd.transpose(-2, -1)) * scale

        if custom_mask is not None:
            m = custom_mask[b, :Sq, :kv_b].to(dtype=torch.bool, device=query.device)
            scores = scores.masked_fill(m.unsqueeze(0), float("-inf"))

        row_max = scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        probs = torch.exp(scores - row_max)
        probs = torch.where(torch.isfinite(scores), probs, torch.zeros_like(probs))
        denom = probs.sum(dim=-1, keepdim=True)
        attn = torch.where(
            denom > 0, probs / denom.clamp_min(1e-30), torch.zeros_like(probs)
        )

        o_hnd = torch.matmul(attn, v_hnd)
        out[b] = o_hnd.permute(1, 0, 2).to(query.dtype)
    return out


def run_debug(
    args: argparse.Namespace, mask_mode: str, mask_key: str, **kwargs
) -> RunResult:
    """调试入口：TileMask 生成 → kernel → golden 校验"""
    input_format = kwargs.pop("input_format", "BSND")
    ko = _run_kernel_core(
        args, mask_mode, mask_key, input_format=input_format, **kwargs
    )

    max_sq = max(ko.q_lens_list)
    max_kv = max(ko.kv_lens_list)
    B = len(ko.kv_lens_list)

    if bool(getattr(args, "perf_only", False)):
        return RunResult(name="perf_only", passed=True, info="verify skipped")
    cpu_tensors = {}
    cpu_tensors["seq_lens"] = torch.tensor(ko.kv_lens_list, dtype=torch.int32)

    # dense mask(仅用于 golden): True=可见
    _prefix = max_kv - max_sq if args.mask == "causal" else max_kv
    _cols = torch.arange(max_kv)
    _rows = torch.arange(max_sq)
    dense_mask_cpu = (
        (_cols.unsqueeze(0) <= (_rows.unsqueeze(1) + _prefix))
        .unsqueeze(0)
        .expand(B, -1, -1)
    )
    dense_mask_for_golden = ~dense_mask_cpu  # True=masked

    scale_val = 1.0 / math.sqrt(float(ko.D))
    op_dtype = ko.query.dtype
    kv_lens_t = torch.tensor(ko.kv_lens_list, dtype=torch.int32)

    # 真值 = f32 全量; 标杆 = FA分块模拟; TND/BSND 分别路由
    if ko.input_format == "TND":
        golden_fa = compute_golden_torch_tnd(
            ko.query,
            ko.key,
            ko.value,
            scale_val,
            cu_seqlens_q=ko.cu_seqlens_q,
            cu_seqlens_k=ko.cu_seqlens_k,
            custom_mask=dense_mask_for_golden,
        )
        golden_flat = golden_fa.reshape(ko.total_q, ko.D).cpu().float()
        # 标杆: 逐 batch 切片为 BSND, 调 _fa_style_bm 后拼接
        ref_parts = []
        cu_q = (
            ko.cu_seqlens_q.tolist() if ko.cu_seqlens_q is not None else [0, ko.total_q]
        )
        cu_k = (
            ko.cu_seqlens_k.tolist()
            if ko.cu_seqlens_k is not None
            else [0, ko.key.shape[0]]
        )
        for bi in range(len(cu_q) - 1):
            qs, qe = int(cu_q[bi]), int(cu_q[bi + 1])
            ks, ke = int(cu_k[bi]), int(cu_k[bi + 1])
            if qe <= qs:
                continue
            q_b = ko.query[qs:qe].unsqueeze(0)  # (1, sq, H, D)
            k_b = ko.key[ks:ke].unsqueeze(0)
            v_b = ko.value[ks:ke].unsqueeze(0)
            m_b = (
                dense_mask_for_golden[bi : bi + 1, : qe - qs, : ke - ks]
                if dense_mask_for_golden is not None
                else None
            )
            ref_b = _fa_style_bm(
                q_b, k_b, v_b, scale_val, kv_lens_per_batch=[ke - ks], custom_mask=m_b
            )
            ref_parts.append(ref_b.reshape(-1, ko.D))
        ref_flat = torch.cat(ref_parts, dim=0).cpu().float()
    else:
        golden_fa = compute_golden_torch_bsnd(
            ko.query,
            ko.key,
            ko.value,
            scale_val,
            kv_lens_per_batch=kv_lens_t,
            custom_mask=dense_mask_for_golden,
        )
        golden_flat = golden_fa.reshape(ko.total_q, ko.D).cpu().float()
        ref_fa = _fa_style_bm(
            ko.query,
            ko.key,
            ko.value,
            scale_val,
            kv_lens_per_batch=kv_lens_t,
            custom_mask=dense_mask_for_golden,
        )
        ref_flat = ref_fa.reshape(ko.total_q, ko.D).cpu().float()

    kernel_flat = ko.out_buf.reshape(ko.total_q, ko.D).cpu().float()

    _eps_rel = 1e-7

    def _met_std(a):
        _d = (a - golden_flat).abs()
        _r = _d / (golden_flat.abs() + _eps_rel)
        return _r.max().item(), _r.mean().item(), torch.sqrt((_d**2).mean()).item()

    num_mare, num_mere, num_rmse = _met_std(kernel_flat)
    den_mare, den_mere, den_rmse = _met_std(ref_flat)

    # 比值（分母与 floor 取 max，避免分母为 0）
    floor = 2 ** (-7) if op_dtype == torch.float16 else 2 ** (-6)
    ratio_rmse = num_rmse / max(den_rmse, floor)
    ratio_mare = num_mare / max(den_mare, floor)
    ratio_mere = num_mere / max(den_mere, floor)

    passed = (ratio_mare <= 2.0) and (ratio_mere <= 1.2) and (ratio_rmse <= 1.2)

    sentinel_t = torch.full_like(kernel_flat, getattr(args, "sentinel", -7.0))
    unchanged = torch.isclose(kernel_flat, sentinel_t, rtol=0.0, atol=1e-2)
    sentinel_info = (
        f"O unchanged (sentinel)? {bool(unchanged.all())} "
        f"changed_count={int((~unchanged).sum().item())} / {kernel_flat.numel()}"
    )

    return RunResult(
        name=ko.display_name,
        passed=passed,
        info=f"kernel={ko.kernel_time_s:.3f}s",
        sentinel_info=sentinel_info,
    )


def run(args: argparse.Namespace) -> int:
    """单条运行"""
    torch_mod = _require_torch_npu(args.device)
    torch_mod.npu.set_device(args.device)

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    op_dtype = dtypes[args.dtype]
    _pattern = args.mask

    print(
        f"--- BATCH=({args.batch},{args.qseqlen},{args.kvseqlen}) "
        f"HEAD=({args.headnum},{args.kvheadnum}) "
        f"HEAD_DIM={args.head_dim} "
        f"dtype={args.dtype} mask={args.mask} sentinel={args.sentinel} ---"
    )

    res = run_debug(
        args,
        args.mask,
        _pattern,
        seq_q=args.qseqlen,
        seq_k=args.kvseqlen,
        batch_size=args.batch,
        num_heads=args.headnum,
        kv_heads=args.kvheadnum,
        head_dim=args.head_dim,
        input_format=args.format,
        op_dtype=op_dtype,
    )
    print(
        f"host=torch_npu BATCH={args.batch} Q_SEQ={args.qseqlen} KV_SEQ={args.kvseqlen} "
        f"HEAD_NUM={args.headnum} KV_HEAD_NUM={args.kvheadnum} HEAD_DIM={args.head_dim}"
    )
    print(res.sentinel_info)
    print(f"passed={res.passed}")
    return 0 if res.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashAttention Infer kernel")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--qseqlen", type=int, default=117, help="Q 序列长度")
    parser.add_argument("--kvseqlen", type=int, default=512, help="KV 序列长度")
    parser.add_argument("--headnum", type=int, default=8, help="Q 头数")
    parser.add_argument("--kvheadnum", type=int, default=1, help="KV 头数")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--mask",
        choices=("none", "causal"),
        default="causal",
        help="mask: none=全可见; causal=chunk-prefill 底对齐因果",
    )
    parser.add_argument("--format", choices=("BSND", "TND"), default="BSND")
    parser.add_argument("--block-num", type=int, default=-1, help="-1=自动取满核")
    parser.add_argument(
        "--paged",
        action="store_true",
        help="PagedAttention: cache+块表, kernel 侧物理页寻址",
    )
    parser.add_argument(
        "--sentinel",
        type=float,
        default=-7.0,
        help="O 的初始值，用于检测 kernel 是否真正写入",
    )
    args = parser.parse_args()
    args.head_dim = 128

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
