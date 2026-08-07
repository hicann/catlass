from __future__ import annotations

import argparse
from typing import Any
import numpy as np

import catlass as tla
from catlass.params import UnalignStoreParams, NormalLoadParams, LoadDist
from catlass.runtime import from_dlpack

from fa_tiling import (
    QK_L1_TILE_M,
    QK_L1_TILE_N,
    QK_L1_TILE_K_LEFT,
    QK_L1_TILE_K_RIGHT,
    PV_L1_TILE_M,
    PV_L1_TILE_N,
    PV_L1_TILE_K_LEFT,
    PV_L1_TILE_K_RIGHT,
    Q_L1_BUF_NUM as Q_L1_BUF,
    K_L1_BUF_NUM as K_L1_BUF,
    V_L1_BUF_NUM as V_L1_BUF,
    P_L1_BUF_NUM as P_L1_BUF,
)
from fa_tiling import compute_tiling, pack_tiling_int, make_actual_seqlen

# tile 大小与硬件参数 (编译期常量)
HEAD_DIM = 128

Q_BLOCK = 128
KV_BLOCK = 128
Q_BLOCK_SUB = Q_BLOCK // 2

# 编译期形状参数
BATCH = 2
HEAD_NUM = 8
KV_HEAD_NUM = 2
Q_SEQ = 117
KV_SEQ = 512
GROUP_SIZE = HEAD_NUM // KV_HEAD_NUM
Q_BLOCK_COUNT = (Q_SEQ + Q_BLOCK - 1) // Q_BLOCK
KV_BLOCK_COUNT = (KV_SEQ + KV_BLOCK - 1) // KV_BLOCK
TOTAL_TASKS = BATCH * HEAD_NUM * Q_BLOCK_COUNT

ALIGNMENT = 512
L0_HALF_SIZE = 32768
VL_FLOAT_ELE = 64
QK_SCALE = 1.0 / (HEAD_DIM ** 0.5)
MIN_VALUE = -3e38

# Prelaunch
PRE_LAUNCH = 2
MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1
UB_S_OTMP_BUF_STAGES = 2
UB_S_BUF_STAGES = 2

L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128
L0_STAGES = 2

mm1L0ATotalStages_ = (Q_BLOCK // L0_TILE_M) * (HEAD_DIM // L0_TILE_K)
mm1L0BTotalStages_ = (KV_BLOCK // L0_TILE_N) * (HEAD_DIM // L0_TILE_K)
mm2L0ATotalStages_ = (Q_BLOCK // L0_TILE_M) * (KV_BLOCK // L0_TILE_K)
mm2L0BTotalStages_ = (KV_BLOCK // L0_TILE_K) * (HEAD_DIM // L0_TILE_N)

QK_ML0_LOOP_NUM = (Q_BLOCK + L0_TILE_M - 1) // L0_TILE_M
QK_NL0_LOOP_NUM = (KV_BLOCK + L0_TILE_N - 1) // L0_TILE_N
QK_KL0_LOOP_NUM = (HEAD_DIM + L0_TILE_K - 1) // L0_TILE_K

PV_ML0_LOOP_NUM = (Q_BLOCK + L0_TILE_M - 1) // L0_TILE_M
PV_NL0_LOOP_NUM = (HEAD_DIM + L0_TILE_N - 1) // L0_TILE_N
PV_KL0_LOOP_NUM = (KV_BLOCK + L0_TILE_K - 1) // L0_TILE_K

THRESHOLD = 0.05
UNCHANGED_THRESHOLD = 0.1


@tla.kernel
def flash_attention_infer_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_o: tla.Tensor,
    mem_mask: tla.Tensor,   # 暂不支持，只能为全0
    tiling_data: tla.Tensor,
    actual_q_seqlen: tla.Tensor,
    actual_kv_seqlen: tla.Tensor,
):
    # QK->softmax (qkReadyFlag, id=0/1, AIC FIX set/wait <-> AIV V wait/set)
    qk_ready_0 = tla.cross_flag("qk_ready_0", mode=4)
    qk_ready_1 = tla.cross_flag("qk_ready_1", mode=4)
    # softmax->PV (softmaxReadyFlag, id=2/3/4, AIV MTE3 set <-> AIC MTE1 wait)
    sm_ready_mm2_0 = tla.cross_flag("sm_ready_mm2_0", mode=4)
    sm_ready_mm2_1 = tla.cross_flag("sm_ready_mm2_1", mode=4)
    sm_ready_mm2_2 = tla.cross_flag("sm_ready_mm2_2", mode=4)
    # PV->rescale (pvReadyFlag, id=5/6, AIC FIX set <-> AIV V wait) — PV fixpipe L0C->UB 后通知 rescale
    pv_ready_0 = tla.cross_flag("pv_ready_0", mode=4)
    pv_ready_1 = tla.cross_flag("pv_ready_1", mode=4)

    q_l0a_ready_l1 = tla.flag("q_l0a_ready_l1", tla.arch.MTE1, tla.arch.MTE2)
    q_l1_ready_l0 = tla.flag("q_l1_ready_l0", tla.arch.MTE2, tla.arch.MTE1)

    k_l0b_ready_l1_0 = tla.flag("k_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_1 = tla.flag("k_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    k_l1_ready_l0_0 = tla.flag("k_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_1 = tla.flag("k_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    mmad_ready_l0a_0 = tla.flag("mmad_ready_l0a_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0a_1 = tla.flag("mmad_ready_l0a_1", tla.arch.CUBE, tla.arch.MTE1)
    l0a_ready_mmad_0 = tla.flag("l0a_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_ready_mmad_1 = tla.flag("l0a_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    mmad_ready_l0b_0 = tla.flag("mmad_ready_l0b_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_1 = tla.flag("mmad_ready_l0b_1", tla.arch.CUBE, tla.arch.MTE1)
    l0b_ready_mmad_0 = tla.flag("l0b_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_1 = tla.flag("l0b_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    fix_ready_mmad_qk_0 = tla.flag("fix_ready_mmad_qk_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_qk_1 = tla.flag("fix_ready_mmad_qk_1", tla.arch.FIX, tla.arch.CUBE)
    mmad_ready_fix_qk_0 = tla.flag("mmad_ready_fix_qk_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_qk_1 = tla.flag("mmad_ready_fix_qk_1", tla.arch.CUBE, tla.arch.FIX)

    fix_ready_mmad_pv_0 = tla.flag("fix_ready_mmad_pv_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_pv_1 = tla.flag("fix_ready_mmad_pv_1", tla.arch.FIX, tla.arch.CUBE)
    mmad_ready_fix_pv_0 = tla.flag("mmad_ready_fix_pv_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_pv_1 = tla.flag("mmad_ready_fix_pv_1", tla.arch.CUBE, tla.arch.FIX)

    v_l0b_ready_l1_0 = tla.flag("v_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_1 = tla.flag("v_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l1_ready_l0_0 = tla.flag("v_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_1 = tla.flag("v_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    #   V_MTE3(ubSBufId)         EVENT_ID0/1 — softmax V 完 -> 触发 P->L1 MTE3
    #   MTE3_V(ubSBufId + 2)     EVENT_ID2/3 — 上一轮 P->L1 MTE3 完成 -> 本轮 softmax 可动
    mte3_ready_softmax_0 = tla.flag("mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR)
    mte3_ready_softmax_1 = tla.flag("mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR)
    v_mte3_0 = tla.flag("v_mte3_0", tla.arch.VECTOR, tla.arch.MTE3)
    v_mte3_1 = tla.flag("v_mte3_1", tla.arch.VECTOR, tla.arch.MTE3)
    # mask copy (MTE2->V)
    mask_copy_end = tla.flag("mask_copy_end", tla.arch.MTE2, tla.arch.VECTOR)
    # rescale MTE3→V : 上一轮 O->GM MTE3 完成后本轮 rescale 可动
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)
    # rescale V→MTE3 : cast 写完 ub_out_f16 后才 O->GM MTE3
    rescale_v_mte3 = tla.flag("rescale_v_mte3", tla.arch.VECTOR, tla.arch.MTE3)

    c0 = 0
    c1 = 1

    mem_allocator = tla.utils.LocalmemAllocator()

    l1_q_ptr = mem_allocator.allocate(Q_BLOCK * HEAD_DIM * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_q_ptr = tla.recast_ptr(l1_q_ptr, dtype=tla.Float16)
    l1_k_ping_ptr = mem_allocator.allocate(HEAD_DIM * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_k_ping_ptr = tla.recast_ptr(l1_k_ping_ptr, dtype=tla.Float16)
    l1_k_pong_ptr = mem_allocator.allocate(HEAD_DIM * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_k_pong_ptr = tla.recast_ptr(l1_k_pong_ptr, dtype=tla.Float16)
    l1_v_ping_ptr = mem_allocator.allocate(KV_BLOCK * HEAD_DIM * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_v_ping_ptr = tla.recast_ptr(l1_v_ping_ptr, dtype=tla.Float16)
    l1_v_pong_ptr = mem_allocator.allocate(KV_BLOCK * HEAD_DIM * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_v_pong_ptr = tla.recast_ptr(l1_v_pong_ptr, dtype=tla.Float16)
    # P L1 buffer: pL1BufNum=3
    l1_p_0_ptr = mem_allocator.allocate(Q_BLOCK * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_p_0_ptr = tla.recast_ptr(l1_p_0_ptr, dtype=tla.Float16)
    l1_p_1_ptr = mem_allocator.allocate(Q_BLOCK * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_p_1_ptr = tla.recast_ptr(l1_p_1_ptr, dtype=tla.Float16)
    l1_p_2_ptr = mem_allocator.allocate(Q_BLOCK * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.l1)
    l1_p_2_ptr = tla.recast_ptr(l1_p_2_ptr, dtype=tla.Float16)

    # L0A/L0B ping/pong 各 2 份, QK/PV 共用，全局递增
    l0a_ping_ptr = mem_allocator.allocate(L0_HALF_SIZE, ALIGNMENT, tla.AddressSpace.l0a)
    l0a_ping_ptr = tla.recast_ptr(l0a_ping_ptr, dtype=tla.Float16)
    l0a_pong_ptr = mem_allocator.allocate(L0_HALF_SIZE, ALIGNMENT, tla.AddressSpace.l0a)
    l0a_pong_ptr = tla.recast_ptr(l0a_pong_ptr, dtype=tla.Float16)
    l0b_ping_ptr = mem_allocator.allocate(L0_HALF_SIZE, ALIGNMENT, tla.AddressSpace.l0b)
    l0b_ping_ptr = tla.recast_ptr(l0b_ping_ptr, dtype=tla.Float16)
    l0b_pong_ptr = mem_allocator.allocate(L0_HALF_SIZE, ALIGNMENT, tla.AddressSpace.l0b)
    l0b_pong_ptr = tla.recast_ptr(l0b_pong_ptr, dtype=tla.Float16)

    l0_s_ping_ptr = mem_allocator.allocate(Q_BLOCK * KV_BLOCK * 4, ALIGNMENT, tla.AddressSpace.l0c)
    l0_s_ping_ptr = tla.recast_ptr(l0_s_ping_ptr, dtype=tla.Float32)
    l0_s_pong_ptr = mem_allocator.allocate(Q_BLOCK * KV_BLOCK * 4, ALIGNMENT, tla.AddressSpace.l0c)
    l0_s_pong_ptr = tla.recast_ptr(l0_s_pong_ptr, dtype=tla.Float32)

    l0_pv_ping_ptr = mem_allocator.allocate(Q_BLOCK * HEAD_DIM * 4, ALIGNMENT, tla.AddressSpace.l0c)
    l0_pv_ping_ptr = tla.recast_ptr(l0_pv_ping_ptr, dtype=tla.Float32)
    l0_pv_pong_ptr = mem_allocator.allocate(Q_BLOCK * HEAD_DIM * 4, ALIGNMENT, tla.AddressSpace.l0c)
    l0_pv_pong_ptr = tla.recast_ptr(l0_pv_pong_ptr, dtype=tla.Float32)

    ub_s_ping_ptr = mem_allocator.allocate(Q_BLOCK_SUB * KV_BLOCK * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_s_ping_ptr = tla.recast_ptr(ub_s_ping_ptr, dtype=tla.Float32)
    ub_s_pong_ptr = mem_allocator.allocate(Q_BLOCK_SUB * KV_BLOCK * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_s_pong_ptr = tla.recast_ptr(ub_s_pong_ptr, dtype=tla.Float32)
    # P buffer (zN布局, Q_BLOCK_SUB+1 padding 用于 BlockStoreParams)
    ub_p_f16_ping_ptr = mem_allocator.allocate((Q_BLOCK_SUB + 1) * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.ub)
    ub_p_f16_ping_ptr = tla.recast_ptr(ub_p_f16_ping_ptr, dtype=tla.Float16)
    ub_p_f16_pong_ptr = mem_allocator.allocate((Q_BLOCK_SUB + 1) * KV_BLOCK * 2, ALIGNMENT, tla.AddressSpace.ub)
    ub_p_f16_pong_ptr = tla.recast_ptr(ub_p_f16_pong_ptr, dtype=tla.Float16)

    ub_pv_ping_ptr = mem_allocator.allocate(Q_BLOCK_SUB * HEAD_DIM * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_pv_ping_ptr = tla.recast_ptr(ub_pv_ping_ptr, dtype=tla.Float32)
    ub_pv_pong_ptr = mem_allocator.allocate(Q_BLOCK_SUB * HEAD_DIM * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_pv_pong_ptr = tla.recast_ptr(ub_pv_pong_ptr, dtype=tla.Float32)

    ub_acc_ptr = mem_allocator.allocate(Q_BLOCK_SUB * HEAD_DIM * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_acc_ptr = tla.recast_ptr(ub_acc_ptr, dtype=tla.Float32)

    ub_out_f16_ptr = mem_allocator.allocate(Q_BLOCK_SUB * HEAD_DIM * 2, ALIGNMENT, tla.AddressSpace.ub)
    ub_out_f16_ptr = tla.recast_ptr(ub_out_f16_ptr, dtype=tla.Float16)

    ub_now_max_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_now_max_ptr = tla.recast_ptr(ub_now_max_ptr, dtype=tla.Float32)
    ub_last_max_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_last_max_ptr = tla.recast_ptr(ub_last_max_ptr, dtype=tla.Float32)
    ub_sum_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_sum_ptr = tla.recast_ptr(ub_sum_ptr, dtype=tla.Float32)
    ub_tmp_ptr = mem_allocator.allocate(2 * VL_FLOAT_ELE * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_tmp_ptr = tla.recast_ptr(ub_tmp_ptr, dtype=tla.Float32)
    # mask占位
    ub_mask_ptr = mem_allocator.allocate(Q_BLOCK_SUB * KV_BLOCK * 1, ALIGNMENT, tla.AddressSpace.ub)
    ub_mask_ptr = tla.recast_ptr(ub_mask_ptr, dtype=tla.Int8)

    ub_exp_sum_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_exp_sum_ptr = tla.recast_ptr(ub_exp_sum_ptr, dtype=tla.Float32)
    ub_exp_max_0_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_exp_max_0_ptr = tla.recast_ptr(ub_exp_max_0_ptr, dtype=tla.Float32)
    ub_exp_max_1_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_exp_max_1_ptr = tla.recast_ptr(ub_exp_max_1_ptr, dtype=tla.Float32)
    ub_exp_max_2_ptr = mem_allocator.allocate(Q_BLOCK_SUB * 4, ALIGNMENT, tla.AddressSpace.ub)
    ub_exp_max_2_ptr = tla.recast_ptr(ub_exp_max_2_ptr, dtype=tla.Float32)

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

    # tilingdata
    q_heads = HEAD_NUM
    kv_heads = KV_HEAD_NUM
    group_size = GROUP_SIZE
    max_q_seqlen = Q_SEQ
    max_kv_seqlen = KV_SEQ
    first_batch_task_num = HEAD_NUM * Q_BLOCK_COUNT
    total_task_num = TOTAL_TASKS

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
        tla.set_flag(fix_ready_mmad_qk_0)
        tla.set_flag(fix_ready_mmad_qk_1)
        tla.set_flag(fix_ready_mmad_pv_0)
        tla.set_flag(fix_ready_mmad_pv_1)

        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1, aiv_id=0)
        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1, aiv_id=1)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1, aiv_id=0)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1, aiv_id=1)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1, aiv_id=0)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1, aiv_id=1)

    with tla.vector():
        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR, aiv_id=1)
        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR, aiv_id=1)

        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR, aiv_id=1)
        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR, aiv_id=1)

        tla.set_flag(mte3_ready_softmax_0)
        tla.set_flag(mte3_ready_softmax_1)

        tla.set_flag(mte3_ready_rescale)

    task_range = tla.range(
        tla.arch.block_idx(),
        TOTAL_TASKS,
        tla.arch.block_dim(),
    )
    cur_batch = c0
    pre_total_task_num = c0
    cur_total_task_num = first_batch_task_num
    q_b_offset = c0
    o_b_offset = c0
    q_seqlen_cur = Q_SEQ
    kv_block_count_cur = KV_BLOCK_COUNT
    for task in task_range:
        if task >= cur_total_task_num:
            cur_batch = cur_batch + c1
            pre_total_task_num = cur_total_task_num
            q_b_offset = q_b_offset + q_seqlen_cur * q_heads * HEAD_DIM
            o_b_offset = o_b_offset + q_seqlen_cur * q_heads * HEAD_DIM
            q_seqlen_cur = Q_SEQ
            cur_total_task_num = cur_total_task_num + ((q_seqlen_cur + Q_BLOCK - c1) // Q_BLOCK) * q_heads

        task_idx_cur_batch = task - pre_total_task_num
        q_block_idx = task_idx_cur_batch // q_heads
        head_idx = task_idx_cur_batch - q_block_idx * q_heads
        kv_head_idx = head_idx // group_size

        q_tile_size = (Q_SEQ - (Q_BLOCK_COUNT - 1) * Q_BLOCK) if q_block_idx == Q_BLOCK_COUNT - 1 else Q_BLOCK

        qo_stride = q_heads * HEAD_DIM
        kv_stride = kv_heads * HEAD_DIM
        kv_b_offset = cur_batch * max_kv_seqlen * kv_heads * HEAD_DIM

        mem_q_block = tla.make_tensor(
            mem_q.ptr + q_b_offset + head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(max_q_seqlen, HEAD_DIM),
                tla.make_stride(qo_stride, 1),
            )
        )
        mem_k_block = tla.make_tensor(
            mem_k.ptr + kv_b_offset + kv_head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(HEAD_DIM, max_kv_seqlen),
                tla.make_stride(1, kv_stride),
                layoutTag=tla.arch.ColumnMajor,
            )
        )
        mem_v_block = tla.make_tensor(
            mem_v.ptr + kv_b_offset + kv_head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(max_kv_seqlen, HEAD_DIM),
                tla.make_stride(kv_stride, 1)
            )
        )
        mem_o_block = tla.make_tensor(
            mem_o.ptr + o_b_offset + head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(max_q_seqlen, HEAD_DIM),
                tla.make_stride(qo_stride, 1),
            )
        )

        with tla.cube():
            q_gm = tla.tile_view(
                mem_q_block, tla.make_shape(Q_BLOCK, HEAD_DIM), tla.make_coord(q_block_idx, c0)
            )
            l1_q = tla.make_tensor_like(l1_q_ptr, q_gm)
            tla.wait_flag(q_l0a_ready_l1)
            tla.copy(l1_q, q_gm)
            tla.set_flag(q_l1_ready_l0)
            tla.wait_flag(q_l1_ready_l0)


        for kv_iter in tla.range(c0, kv_block_count_cur + PRE_LAUNCH, c1):
            if kv_iter < kv_block_count_cur:
                ubSBufId = kv_iter % UB_S_OTMP_BUF_STAGES
                l1PBufId_qk = kv_iter % P_L1_BUF
                l1BBufId = kv_iter % K_L1_BUF
                l0CBufId = kv_iter % L0_STAGES

                kv_tile_size = (KV_SEQ - (KV_BLOCK_COUNT - 1) * KV_BLOCK) if kv_iter == KV_BLOCK_COUNT - 1 else KV_BLOCK

                # L0A/L0B ping/pong flag
                prefixSumL0AStages = (kv_iter * mm1L0ATotalStages_) if (kv_iter <= PRE_LAUNCH) \
                    else (kv_iter * mm1L0ATotalStages_ + (kv_iter - PRE_LAUNCH) * mm2L0ATotalStages_)
                prefixSumL0BStages = (kv_iter * mm1L0BTotalStages_) if (kv_iter <= PRE_LAUNCH) \
                    else (kv_iter * mm1L0BTotalStages_ + (kv_iter - PRE_LAUNCH) * mm2L0BTotalStages_)

                with tla.cube():
                    l1_k_ptr = l1_k_ping_ptr if l1BBufId == c0 else l1_k_pong_ptr
                    l0_s_ptr = l0_s_ping_ptr if l0CBufId == c0 else l0_s_pong_ptr
                    ub_s_ptr = ub_s_ping_ptr if ubSBufId == c0 else ub_s_pong_ptr

                    k_gm = tla.tile_view(
                        mem_k_block, tla.make_shape(HEAD_DIM, KV_BLOCK), tla.make_coord(c0, kv_iter)
                    )
                    q_gm = tla.tile_view(
                        mem_q_block, tla.make_shape(Q_BLOCK, HEAD_DIM), tla.make_coord(q_block_idx, c0)
                    )
                    l1_q = tla.make_tensor_like(l1_q_ptr, q_gm)
                    l1_k = tla.make_tensor_like(l1_k_ptr, k_gm, layoutTag=tla.arch.nZ)

                    for nL0Itr in tla.range(c0, QK_NL0_LOOP_NUM, c1):
                        nLoopCounter = nL0Itr * QK_NL0_LOOP_NUM + nL0Itr
                        l0c_s = tla.make_tensor_like(l0_s_ptr, k_gm, tla.arch.L0Clayout)
                        s_ub = tla.make_tensor(ub_s_ptr,
                                                tla.make_layout(
                                                    tla.make_shape(q_tile_size, kv_tile_size),
                                                    tla.make_stride(KV_BLOCK, 1)
                                                ))
                        ub_s_fix = tla.tile_view(
                            s_ub,
                            tla.make_shape(Q_BLOCK, KV_BLOCK),
                            tla.make_coord(c0, c0),
                        )

                        if l1BBufId == c0:
                            tla.wait_flag(k_l0b_ready_l1_0)
                        else:
                            tla.wait_flag(k_l0b_ready_l1_1)
                        tla.copy(l1_k, k_gm)
                        if l1BBufId == c0:
                            tla.set_flag(k_l1_ready_l0_0)
                        else:
                            tla.set_flag(k_l1_ready_l0_1)

                        for mL0Itr in tla.range(c0, QK_ML0_LOOP_NUM, c1):
                            for kL0Itr in tla.range(c0, QK_KL0_LOOP_NUM, c1):
                                l0ALoopCounter = prefixSumL0AStages + mL0Itr * QK_KL0_LOOP_NUM + kL0Itr
                                l0BLoopCounter = prefixSumL0BStages + nLoopCounter * QK_KL0_LOOP_NUM + kL0Itr
                                l0ABufId = l0ALoopCounter % L0_STAGES
                                l0BBufId_l0 = l0BLoopCounter % L0_STAGES
                                l0_q_ptr = l0a_ping_ptr if l0ABufId == c0 else l0a_pong_ptr
                                l0_k_ptr = l0b_ping_ptr if l0BBufId_l0 == c0 else l0b_pong_ptr
                                l0_q = tla.make_tensor_like(l0_q_ptr, l1_q)
                                l0_k = tla.make_tensor_like(l0_k_ptr, l1_k)

                                if l0ABufId == c0:
                                    tla.wait_flag(mmad_ready_l0a_0)
                                else:
                                    tla.wait_flag(mmad_ready_l0a_1)
                                tla.copy(l0_q, l1_q)
                                if l0ABufId == c0:
                                    tla.set_flag(l0a_ready_mmad_0)
                                else:
                                    tla.set_flag(l0a_ready_mmad_1)

                                if l0BBufId_l0 == c0:
                                    tla.wait_flag(mmad_ready_l0b_0)
                                else:
                                    tla.wait_flag(mmad_ready_l0b_1)
                                # 首次L1->L0B 需等 K L1 搬运完成
                                if mL0Itr == c0 and kL0Itr == c0:
                                    if l1BBufId == c0:
                                        tla.wait_flag(k_l1_ready_l0_0)
                                    else:
                                        tla.wait_flag(k_l1_ready_l0_1)
                                tla.copy(l0_k, l1_k)
                                if l0BBufId_l0 == c0:
                                    tla.set_flag(l0b_ready_mmad_0)
                                else:
                                    tla.set_flag(l0b_ready_mmad_1)
                                # 末次L1->L0B 释放 K L1 给下一base tile
                                if mL0Itr == (QK_ML0_LOOP_NUM - c1) and kL0Itr == (QK_KL0_LOOP_NUM - c1):
                                    if l1BBufId == c0:
                                        tla.set_flag(k_l0b_ready_l1_0)
                                    else:
                                        tla.set_flag(k_l0b_ready_l1_1)

                                if l0ABufId == c0:
                                    tla.wait_flag(l0a_ready_mmad_0)
                                else:
                                    tla.wait_flag(l0a_ready_mmad_1)
                                if l0BBufId_l0 == c0:
                                    tla.wait_flag(l0b_ready_mmad_0)
                                else:
                                    tla.wait_flag(l0b_ready_mmad_1)

                                if mL0Itr == c0 and kL0Itr == c0:
                                    if l0CBufId == c0:
                                        tla.wait_flag(fix_ready_mmad_qk_0)
                                    else:
                                        tla.wait_flag(fix_ready_mmad_qk_1)
                                tla.mmad(l0c_s, l0_q, l0_k, init_c=True)
                                if l0ABufId == c0:
                                    tla.set_flag(mmad_ready_l0a_0)
                                else:
                                    tla.set_flag(mmad_ready_l0a_1)
                                if l0BBufId_l0 == c0:
                                    tla.set_flag(mmad_ready_l0b_0)
                                else:
                                    tla.set_flag(mmad_ready_l0b_1)

                        if nL0Itr == c0:
                            if ubSBufId == c0:
                                tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX, aiv_id=0)
                                tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX, aiv_id=1)
                            else:
                                tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX, aiv_id=0)
                                tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX, aiv_id=1)
                        if l0CBufId == c0:
                            tla.set_flag(mmad_ready_fix_qk_0)
                            tla.wait_flag(mmad_ready_fix_qk_0)
                        else:
                            tla.set_flag(mmad_ready_fix_qk_1)
                            tla.wait_flag(mmad_ready_fix_qk_1)
                        tla.copy(ub_s_fix, l0c_s, tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ))
                        if l0CBufId == c0:
                            tla.set_flag(fix_ready_mmad_qk_0)
                        else:
                            tla.set_flag(fix_ready_mmad_qk_1)
                    if ubSBufId == c0:
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.FIX, aiv_id=0)
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.FIX, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.FIX, aiv_id=0)
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.FIX, aiv_id=1)
                    if kv_iter == (kv_block_count_cur - c1):
                        tla.set_flag(q_l0a_ready_l1)

                with tla.vector():
                    if ubSBufId == c0:
                        tla.cross_core_wait_flag(qk_ready_0, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_wait_flag(qk_ready_0, tla.arch.VECTOR, aiv_id=1)
                    else:
                        tla.cross_core_wait_flag(qk_ready_1, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_wait_flag(qk_ready_1, tla.arch.VECTOR, aiv_id=1)
                    if ubSBufId == c0:
                        tla.wait_flag(mte3_ready_softmax_0)
                    else:
                        tla.wait_flag(mte3_ready_softmax_1)

                    vec_idx = tla.arch.sub_block_idx()
                    update = kv_iter != c0

                    q_tile_size_half = (q_tile_size + 1) // 2
                    q_tile_size_sub = q_tile_size_half if vec_idx == c0 else (q_tile_size - q_tile_size_half)

                    ub_s_ptr = ub_s_ping_ptr if ubSBufId == c0 else ub_s_pong_ptr
                    ub_p_f16_ptr = ub_p_f16_ping_ptr if ubSBufId == c0 else ub_p_f16_pong_ptr
                    l1_p_ptr_qk = l1_p_0_ptr
                    if l1PBufId_qk == c0:
                        l1_p_ptr_qk = l1_p_0_ptr
                    elif l1PBufId_qk == c1:
                        l1_p_ptr_qk = l1_p_1_ptr
                    else:
                        l1_p_ptr_qk = l1_p_2_ptr
                    ub_exp_max_ptr_qk = ub_exp_max_0_ptr
                    if l1PBufId_qk == c0:
                        ub_exp_max_ptr_qk = ub_exp_max_0_ptr
                    elif l1PBufId_qk == c1:
                        ub_exp_max_ptr_qk = ub_exp_max_1_ptr
                    else:
                        ub_exp_max_ptr_qk = ub_exp_max_2_ptr

                    ub_s = tla.make_tensor(
                        ub_s_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub, KV_BLOCK),
                            tla.make_stride(KV_BLOCK, 1)
                        ),
                    )

                    # Mask占位
                    mem_mask_block = tla.make_tensor(
                        mem_mask.ptr
                        + (q_block_idx * Q_BLOCK + vec_idx * q_tile_size_half) * max_kv_seqlen
                        + kv_iter * KV_BLOCK,
                        tla.make_layout(tla.make_shape(q_tile_size_sub, kv_tile_size), tla.make_stride(max_kv_seqlen, 1)),
                    )
                    ub_mask_sub = tla.make_tensor(
                        ub_mask_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub, kv_tile_size), tla.make_stride(KV_BLOCK, 1)),
                    )
                    tla.copy(ub_mask_sub, mem_mask_block)
                    tla.set_flag(mask_copy_end)
                    tla.wait_flag(mask_copy_end)
                    tla.pipe_barrier(tla.pipes.ALL)
                    ub_mask = tla.make_tensor(
                        ub_mask_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub, kv_tile_size), tla.make_stride(KV_BLOCK, 1)),
                    )
                    ub_now_max = tla.make_tensor(
                        ub_now_max_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub), tla.make_stride(1))
                    )
                    ub_last_max = tla.make_tensor(
                        ub_last_max_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub), tla.make_stride(1))
                    )
                    ub_sum = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub), tla.make_stride(1))
                    )
                    ub_exp_sum = tla.make_tensor(
                        ub_exp_sum_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub), tla.make_stride(1))
                    )
                    ub_exp_max_qk = tla.make_tensor(
                        ub_exp_max_ptr_qk,
                        tla.make_layout(tla.make_shape(q_tile_size_sub), tla.make_stride(1))
                    )
                    # P UB layout: RowMajor (65, 128), zN
                    ub_p_f16 = tla.make_tensor(
                        ub_p_f16_ptr,
                        tla.make_layout(
                            tla.make_shape(Q_BLOCK_SUB + 1, KV_BLOCK),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    ub_p_zN_full = tla.make_tensor_like(ub_p_f16_ptr, ub_p_f16, tla.arch.zNUnAlign)
                    ub_p_zN = tla.tile_view(
                        ub_p_zN_full, tla.make_shape(q_tile_size_sub, kv_tile_size), tla.make_coord(c0, c0)
                    )
                    ub_tmp = tla.make_tensor(
                        ub_tmp_ptr,
                        tla.make_layout(tla.make_shape(2 * VL_FLOAT_ELE), tla.make_stride(1)),
                    )

                    remaining = kv_tile_size % VL_FLOAT_ELE
                    # Phase 1: scale*qk + mask, compute block_max
                    with tla.vec.func(mode='simd'):
                        reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        one_mask_p1, _ = tla.update_mask(1, dtype=tla.Float32)
                        mask_tail_n, _ = tla.update_mask(remaining, dtype=tla.Float32)
                        min_reg = tla.full(-65504.0, dtype=tla.Float32)
                        for i0 in tla.range(q_tile_size_sub):
                            if kv_tile_size > VL_FLOAT_ELE:
                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i0, c0))
                                ub_s_i1 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i0, c1))
                                ub_s_reg0_if0 = ub_s_i0.load()
                                ub_s_reg1_if0 = ub_s_i1.load()
                                ub_s_reg0_if0 = tla.mul(ub_s_reg0_if0, QK_SCALE, mask=reduce_mask)
                                ub_s_reg1_if0 = tla.mul(ub_s_reg1_if0, QK_SCALE, mask=reduce_mask)
                                if kv_tile_size < KV_BLOCK:
                                    ub_s_reg1_if0 = tla.where(mask_tail_n, ub_s_reg1_if0, min_reg)
                                ub_s_i0.store(ub_s_reg0_if0, mask=reduce_mask)
                                ub_s_i1.store(ub_s_reg1_if0, mask=reduce_mask)
                                tmp_reg_if0 = tla.max(ub_s_reg0_if0, ub_s_reg1_if0, mask=reduce_mask)
                                max_reg_if0 = tmp_reg_if0.reduce(tla.ReductionOp.MAX, mask=reduce_mask)
                                ub_max_dst_if0 = tla.tile_view(ub_now_max, tla.make_shape(1), tla.make_coord(i0))
                                if kv_iter == c0:
                                    ub_max_dst_if0.store(max_reg_if0, params=UnalignStoreParams(), mask=one_mask_p1)
                                else:
                                    ub_max_dst_if0 = tla.tile_view(ub_last_max, tla.make_shape(1), tla.make_coord(i0))
                                    ub_max_dst_if0.store(max_reg_if0, params=UnalignStoreParams(), mask=one_mask_p1)
                            else:
                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i0, c0))
                                ub_s_reg0_if0 = ub_s_i0.load()
                                ub_s_reg0_if0 = tla.mul(ub_s_reg0_if0, QK_SCALE, mask=reduce_mask)
                                if kv_tile_size < VL_FLOAT_ELE:
                                    ub_s_reg0_if0 = tla.where(mask_tail_n, ub_s_reg0_if0, min_reg)
                                ub_s_i0.store(ub_s_reg0_if0, mask=reduce_mask)
                                max_reg_if0 = ub_s_reg0_if0.reduce(tla.ReductionOp.MAX, mask=reduce_mask)
                                ub_max_dst_if0 = tla.tile_view(ub_now_max, tla.make_shape(1), tla.make_coord(i0))
                                if kv_iter == c0:
                                    ub_max_dst_if0.store(max_reg_if0, params=UnalignStoreParams(), mask=one_mask_p1)
                                else:
                                    ub_max_dst_if0 = tla.tile_view(ub_last_max, tla.make_shape(1), tla.make_coord(i0))
                                    ub_max_dst_if0.store(max_reg_if0, params=UnalignStoreParams(), mask=one_mask_p1)
                    tla.pipe_barrier(tla.pipes.ALL)

                    # Phase 1b: UpdateMax (if update)
                    if update:
                        with tla.vec.func(mode='simd'):
                            pregFull_um = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            one_mask_um, _ = tla.update_mask(1, dtype=tla.Float32)
                            for i1 in tla.range(q_tile_size_sub):
                                ub_now_max_i_if0 = tla.tile_view(ub_now_max, tla.make_shape(1), tla.make_coord(i1))
                                ub_last_max_i_if0 = tla.tile_view(ub_last_max, tla.make_shape(1), tla.make_coord(i1))
                                now_reg_if0 = ub_now_max_i_if0.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                last_reg_if0 = ub_last_max_i_if0.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                global_max_if0 = tla.max(now_reg_if0, last_reg_if0, mask=pregFull_um)
                                ub_now_max_i_if0.store(global_max_if0, params=UnalignStoreParams(), mask=one_mask_um)

                    # Phase 2: exp(s - global_max), store back to ub_s, block_sum
                    with tla.vec.func(mode='simd'):
                        reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        one_mask_p2, _ = tla.update_mask(1, dtype=tla.Float32)
                        for i2 in tla.range(q_tile_size_sub):
                            ub_now_max_i = tla.tile_view(ub_now_max, tla.make_shape(1), tla.make_coord(i2))
                            max_reg = ub_now_max_i.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                            if kv_tile_size > VL_FLOAT_ELE:
                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i2, c0))
                                ub_s_i1 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i2, c1))
                                ub_s_reg0 = ub_s_i0.load()
                                ub_s_reg1 = ub_s_i1.load()
                                exp_reg0 = tla.sub(ub_s_reg0, max_reg, mask=reduce_mask)
                                exp_reg1 = tla.sub(ub_s_reg1, max_reg, mask=reduce_mask)
                                exp_reg0 = tla.exp(exp_reg0, mask=reduce_mask)
                                exp_reg1 = tla.exp(exp_reg1, mask=reduce_mask)
                                ub_s_i0.store(exp_reg0, mask=reduce_mask)
                                ub_s_i1.store(exp_reg1, mask=reduce_mask)
                                tmp_reg = tla.add(exp_reg0, exp_reg1, mask=reduce_mask)
                                block_sum_reg = tmp_reg.reduce(tla.ReductionOp.ADD, mask=reduce_mask)
                                if kv_iter == c0:
                                    ub_sum_i = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(i2))
                                    ub_sum_i.store(block_sum_reg, params=UnalignStoreParams(), mask=one_mask_p2)
                                else:
                                    ub_exp_sum_i = tla.tile_view(ub_exp_sum, tla.make_shape(1), tla.make_coord(i2))
                                    ub_exp_sum_i.store(block_sum_reg, params=UnalignStoreParams(), mask=one_mask_p2)
                            else:
                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i2, c0))
                                ub_s_reg0 = ub_s_i0.load()
                                exp_reg0 = tla.sub(ub_s_reg0, max_reg, mask=reduce_mask)
                                exp_reg0 = tla.exp(exp_reg0, mask=reduce_mask)
                                ub_s_i0.store(exp_reg0, mask=reduce_mask)
                                block_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=reduce_mask)
                                if kv_iter == c0:
                                    ub_sum_i = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(i2))
                                    ub_sum_i.store(block_sum_reg, params=UnalignStoreParams(), mask=one_mask_p2)
                                else:
                                    ub_exp_sum_i = tla.tile_view(ub_exp_sum, tla.make_shape(1), tla.make_coord(i2))
                                    ub_exp_sum_i.store(block_sum_reg, params=UnalignStoreParams(), mask=one_mask_p2)

                    # Phase 3: cast f32->f16, DINTLV+cast+bitwise_or store zNUnAlign P
                    with tla.vec.func(mode='simd'):
                        pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                        for i3 in tla.range(q_tile_size_sub):
                            if kv_tile_size > VL_FLOAT_ELE:
                                ub_s_row = tla.tile_view(ub_s, tla.make_shape(1, 2 * VL_FLOAT_ELE), tla.make_coord(i3, c0))
                                s_odd_reg, s_even_reg = ub_s_row.load(params=NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32))
                                p_even_reg = s_even_reg.to(tla.Float16, cast_trait_one, mask=pregFull)
                                p_odd_reg = s_odd_reg.to(tla.Float16, cast_trait_zero, mask=pregFull)
                                p_zn_reg = tla.bitwise_or(p_even_reg, p_odd_reg, mask=preg_all_b16)
                                ub_p_zn_i0 = tla.tile_view(ub_p_zN, tla.make_shape(1, 2 * VL_FLOAT_ELE), tla.make_coord(i3, c0))
                                ub_p_zn_i0.store(p_zn_reg, params=tla.params.BlockStoreParams(block_stride=Q_BLOCK_SUB + 1), mask=preg_all_b16)
                            else:
                                ub_s_i0 = tla.tile_view(ub_s, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i3, c0))
                                a_reg0 = ub_s_i0.load()
                                p_reg0 = a_reg0.to(tla.Float16, cast_trait_zero, mask=pregFull)
                                r0_qk, _ = tla.deinterleave(p_reg0, p_reg0)
                                ub_p_zn_i0 = tla.tile_view(ub_p_zN, tla.make_shape(1, 2 * VL_FLOAT_ELE), tla.make_coord(i3, c0))
                                ub_p_zn_i0.store(r0_qk, params=tla.params.BlockStoreParams(block_stride=Q_BLOCK_SUB + 1), mask=preg_all_b16)
                    tla.pipe_barrier(tla.pipes.ALL)

                    if ubSBufId == c0:
                        tla.set_flag(v_mte3_0)
                        tla.wait_flag(v_mte3_0)
                    else:
                        tla.set_flag(v_mte3_1)
                        tla.wait_flag(v_mte3_1)
                    if ubSBufId == c0:
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR, aiv_id=1)

                    # softmaxReadyFlag = l1PBufId + 2; 等 PV 用完上一轮 P buffer
                    if l1PBufId_qk == c0:
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3, aiv_id=0)
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3, aiv_id=1)
                    elif l1PBufId_qk == c1:
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3, aiv_id=0)
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3, aiv_id=1)
                    else:
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3, aiv_id=0)
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3, aiv_id=1)
                    ub_p_zN_copy = tla.tile_view(ub_p_zN_full, tla.make_shape(q_tile_size_sub, kv_tile_size), tla.make_coord(c0, c0))
                    l1_p_ref = tla.make_tensor(
                        l1_p_ptr_qk,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, kv_tile_size),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    l1_p_qk = tla.make_tensor_like(l1_p_ptr_qk, l1_p_ref, tla.arch.zN)
                    l1_p_sub = tla.tile_view(l1_p_qk, tla.make_shape(q_tile_size_half, kv_tile_size), tla.make_coord(vec_idx, c0))
                    tla.copy(l1_p_sub, ub_p_zN_copy)
                    # 释放 ub_s 给下一轮 softmax
                    if ubSBufId == c0:
                        tla.set_flag(mte3_ready_softmax_0)
                    else:
                        tla.set_flag(mte3_ready_softmax_1)

                    if l1PBufId_qk == c0:
                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3, aiv_id=0)
                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3, aiv_id=1)
                    elif l1PBufId_qk == c1:
                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3, aiv_id=0)
                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3, aiv_id=0)
                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3, aiv_id=1)

                    # Phase 2b: UpdateExpSumAndExpMax
                    if update:
                        with tla.vec.func(mode='simd'):
                            pregFull_ue = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            one_mask_ue, _ = tla.update_mask(1, dtype=tla.Float32)
                            for i3 in tla.range(q_tile_size_sub):
                                ub_now_max_i_if1 = tla.tile_view(ub_now_max, tla.make_shape(1), tla.make_coord(i3))
                                ub_last_max_i_if1 = tla.tile_view(ub_last_max, tla.make_shape(1), tla.make_coord(i3))
                                ub_sum_i_if1 = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(i3))
                                ub_exp_sum_i_if1 = tla.tile_view(ub_exp_sum, tla.make_shape(1), tla.make_coord(i3))
                                ub_exp_max_i_if1 = tla.tile_view(ub_exp_max_qk, tla.make_shape(1), tla.make_coord(i3))

                                now_reg_if1 = ub_now_max_i_if1.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                last_reg_if1 = ub_last_max_i_if1.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                exp_max_reg_if1 = tla.exp(tla.sub(last_reg_if1, now_reg_if1, mask=pregFull_ue), mask=pregFull_ue)
                                ub_last_max_i_if1.store(now_reg_if1, params=UnalignStoreParams(), mask=one_mask_ue)
                                ub_exp_max_i_if1.store(exp_max_reg_if1, params=UnalignStoreParams(), mask=one_mask_ue)

                                sum_reg_if1 = ub_sum_i_if1.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                exp_sum_reg_if1 = ub_exp_sum_i_if1.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                mul_reg_if1 = tla.mul(sum_reg_if1, exp_max_reg_if1, mask=pregFull_ue)
                                new_sum_if1 = tla.add(mul_reg_if1, exp_sum_reg_if1, mask=pregFull_ue)
                                ub_sum_i_if1.store(new_sum_if1, params=UnalignStoreParams(), mask=one_mask_ue)

            if kv_iter >= PRE_LAUNCH:
                kv_pv = kv_iter - PRE_LAUNCH

                kv_tile_size_pv = (KV_SEQ - (KV_BLOCK_COUNT - 1) * KV_BLOCK) if kv_pv == KV_BLOCK_COUNT - 1 else KV_BLOCK
                ubOTmpBufId = kv_pv % UB_S_OTMP_BUF_STAGES
                l1PBufId_pv = kv_pv % P_L1_BUF
                l1BBufId_pv = kv_pv % V_L1_BUF
                l0CBufId_pv = kv_pv % L0_STAGES

                prefixSumL0AStages_pv = ((kv_pv + c1 + PRE_LAUNCH) * mm1L0ATotalStages_ + kv_pv * mm2L0ATotalStages_) \
                    if (kv_pv < (KV_BLOCK_COUNT - PRE_LAUNCH)) \
                    else (KV_BLOCK_COUNT * mm1L0ATotalStages_ + kv_pv * mm2L0ATotalStages_)
                prefixSumL0BStages_pv = ((kv_pv + c1 + PRE_LAUNCH) * mm1L0BTotalStages_ + kv_pv * mm2L0BTotalStages_) \
                    if (kv_pv < (KV_BLOCK_COUNT - PRE_LAUNCH)) \
                    else (KV_BLOCK_COUNT * mm1L0BTotalStages_ + kv_pv * mm2L0BTotalStages_)

                with tla.cube():
                    l1_v_ptr = l1_v_ping_ptr if l1BBufId_pv == c0 else l1_v_pong_ptr
                    l0_pv_ptr = l0_pv_ping_ptr if l0CBufId_pv == c0 else l0_pv_pong_ptr

                    v_gm = tla.tile_view(
                        mem_v_block, tla.make_shape(KV_BLOCK, HEAD_DIM), tla.make_coord(kv_pv, c0)
                    )
                    l1_v = tla.make_tensor_like(l1_v_ptr, v_gm)

                    if l1BBufId_pv == c0:
                        tla.wait_flag(v_l0b_ready_l1_0)
                    else:
                        tla.wait_flag(v_l0b_ready_l1_1)
                    tla.copy(l1_v, v_gm)
                    if l1BBufId_pv == c0:
                        tla.set_flag(v_l1_ready_l0_0)
                        tla.wait_flag(v_l1_ready_l0_0)
                    else:
                        tla.set_flag(v_l1_ready_l0_1)
                        tla.wait_flag(v_l1_ready_l0_1)

                    if l1PBufId_pv == c0:
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1, aiv_id=0)
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1, aiv_id=1)
                    elif l1PBufId_pv == c1:
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1, aiv_id=0)
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1, aiv_id=1)
                    else:
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1, aiv_id=0)
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1, aiv_id=1)

                    l1_p_ptr_pv = l1_p_0_ptr
                    if l1PBufId_pv == c0:
                        l1_p_ptr_pv = l1_p_0_ptr
                    elif l1PBufId_pv == c1:
                        l1_p_ptr_pv = l1_p_1_ptr
                    else:
                        l1_p_ptr_pv = l1_p_2_ptr
                    ub_p_full_pv = tla.make_tensor(
                        l1_p_ptr_pv,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, kv_tile_size_pv),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    l1_p_pv = tla.make_tensor_like(l1_p_ptr_pv, ub_p_full_pv, tla.arch.zN)

                    for nL0Itr_pv in tla.range(c0, PV_NL0_LOOP_NUM, c1):
                        nLoopCounter_pv = nL0Itr_pv * PV_NL0_LOOP_NUM + nL0Itr_pv
                        l0c_pv_ref = tla.make_tensor(
                            l0_pv_ptr,
                            tla.make_layout(
                                tla.make_shape(q_tile_size, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                            ),
                        )
                        l0c_pv = tla.make_tensor_like(l0_pv_ptr, l0c_pv_ref, tla.arch.L0Clayout)

                        for mL0Itr_pv in tla.range(c0, PV_ML0_LOOP_NUM, c1):
                            for kL0Itr_pv in tla.range(c0, PV_KL0_LOOP_NUM, c1):
                                l0ALoopCounter_pv = prefixSumL0AStages_pv + mL0Itr_pv * PV_KL0_LOOP_NUM + kL0Itr_pv
                                l0BLoopCounter_pv = prefixSumL0BStages_pv + nLoopCounter_pv * PV_KL0_LOOP_NUM + kL0Itr_pv
                                l0ABufId_pv = l0ALoopCounter_pv % L0_STAGES
                                l0BBufId_pv = l0BLoopCounter_pv % L0_STAGES
                                l0_p_ptr = l0a_ping_ptr if l0ABufId_pv == c0 else l0a_pong_ptr
                                l0_v_ptr = l0b_ping_ptr if l0BBufId_pv == c0 else l0b_pong_ptr
                                l0_p = tla.make_tensor_like(l0_p_ptr, l1_p_pv)
                                l0_v = tla.make_tensor_like(l0_v_ptr, l1_v)

                                if l0BBufId_pv == c0:
                                    tla.wait_flag(mmad_ready_l0b_0)
                                else:
                                    tla.wait_flag(mmad_ready_l0b_1)
                                tla.copy(l0_v, l1_v)
                                if l0BBufId_pv == c0:
                                    tla.set_flag(l0b_ready_mmad_0)
                                else:
                                    tla.set_flag(l0b_ready_mmad_1)
                                if mL0Itr_pv == (PV_ML0_LOOP_NUM - c1) and nL0Itr_pv == (PV_NL0_LOOP_NUM - c1) and kL0Itr_pv == (PV_KL0_LOOP_NUM - c1):
                                    if l1BBufId_pv == c0:
                                        tla.set_flag(v_l0b_ready_l1_0)
                                    else:
                                        tla.set_flag(v_l0b_ready_l1_1)

                                if l0ABufId_pv == c0:
                                    tla.wait_flag(mmad_ready_l0a_0)
                                else:
                                    tla.wait_flag(mmad_ready_l0a_1)
                                tla.copy(l0_p, l1_p_pv)
                                if l0ABufId_pv == c0:
                                    tla.set_flag(l0a_ready_mmad_0)
                                else:
                                    tla.set_flag(l0a_ready_mmad_1)
                                if mL0Itr_pv == (PV_ML0_LOOP_NUM - c1) and nL0Itr_pv == (PV_NL0_LOOP_NUM - c1) and kL0Itr_pv == (PV_KL0_LOOP_NUM - c1):
                                    if l1PBufId_pv == c0:
                                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1, aiv_id=0)
                                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1, aiv_id=1)
                                    elif l1PBufId_pv == c1:
                                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1, aiv_id=0)
                                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1, aiv_id=1)
                                    else:
                                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1, aiv_id=0)
                                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1, aiv_id=1)

                                if l0ABufId_pv == c0:
                                    tla.wait_flag(l0a_ready_mmad_0)
                                else:
                                    tla.wait_flag(l0a_ready_mmad_1)
                                if l0BBufId_pv == c0:
                                    tla.wait_flag(l0b_ready_mmad_0)
                                else:
                                    tla.wait_flag(l0b_ready_mmad_1)
                                if mL0Itr_pv == c0 and kL0Itr_pv == c0:
                                    if l0CBufId_pv == c0:
                                        tla.wait_flag(fix_ready_mmad_pv_0)
                                    else:
                                        tla.wait_flag(fix_ready_mmad_pv_1)
                                tla.mmad(l0c_pv, l0_p, l0_v, init_c=True)
                                if l0ABufId_pv == c0:
                                    tla.set_flag(mmad_ready_l0a_0)
                                else:
                                    tla.set_flag(mmad_ready_l0a_1)
                                if l0BBufId_pv == c0:
                                    tla.set_flag(mmad_ready_l0b_0)
                                else:
                                    tla.set_flag(mmad_ready_l0b_1)

                        if nL0Itr_pv == c0:
                            if ubOTmpBufId == c0:
                                tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX, aiv_id=0)
                                tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX, aiv_id=1)
                            else:
                                tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX, aiv_id=0)
                                tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX, aiv_id=1)
                        if l0CBufId_pv == c0:
                            tla.set_flag(mmad_ready_fix_pv_0)
                            tla.wait_flag(mmad_ready_fix_pv_0)
                        else:
                            tla.set_flag(mmad_ready_fix_pv_1)
                            tla.wait_flag(mmad_ready_fix_pv_1)
                        ub_pv_ptr_pv = ub_pv_ping_ptr if ubOTmpBufId == c0 else ub_pv_pong_ptr
                        ub_pv_fix = tla.make_tensor(
                            ub_pv_ptr_pv,
                            tla.make_layout(
                                tla.make_shape(q_tile_size, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(ub_pv_fix, l0c_pv, tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ))
                        if l0CBufId_pv == c0:
                            tla.set_flag(fix_ready_mmad_pv_0)
                        else:
                            tla.set_flag(fix_ready_mmad_pv_1)
                    if ubOTmpBufId == c0:
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.FIX, aiv_id=0)
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.FIX, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.FIX, aiv_id=0)
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.FIX, aiv_id=1)

                with tla.vector():
                    if ubOTmpBufId == c0:
                        tla.cross_core_wait_flag(pv_ready_0, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_wait_flag(pv_ready_0, tla.arch.VECTOR, aiv_id=1)
                    else:
                        tla.cross_core_wait_flag(pv_ready_1, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_wait_flag(pv_ready_1, tla.arch.VECTOR, aiv_id=1)
                    tla.wait_flag(mte3_ready_rescale)

                    ub_pv_ptr_re = ub_pv_ping_ptr if ubOTmpBufId == c0 else ub_pv_pong_ptr
                    ub_exp_max_ptr_re = ub_exp_max_0_ptr
                    if l1PBufId_pv == c0:
                        ub_exp_max_ptr_re = ub_exp_max_0_ptr
                    elif l1PBufId_pv == c1:
                        ub_exp_max_ptr_re = ub_exp_max_1_ptr
                    else:
                        ub_exp_max_ptr_re = ub_exp_max_2_ptr

                    vec_idx_re = tla.arch.sub_block_idx()
                    is_first_pv = kv_pv == c0
                    is_last_pv = kv_pv == (kv_block_count_cur - c1)

                    q_tile_size_half_re = (q_tile_size + 1) // 2
                    q_tile_size_sub_re = q_tile_size_half_re if vec_idx_re == c0 else (q_tile_size - q_tile_size_half_re)

                    ub_pv_re = tla.make_tensor(
                        ub_pv_ptr_re,
                        tla.make_layout(tla.make_shape(q_tile_size_sub_re, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
                    )
                    ub_acc_re = tla.make_tensor(
                        ub_acc_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub_re, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
                    )
                    ub_out_f16_re = tla.make_tensor(
                        ub_out_f16_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub_re, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
                    )
                    ub_sum_re = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(tla.make_shape(q_tile_size_sub_re), tla.make_stride(1)),
                    )
                    ub_exp_max_re = tla.make_tensor(
                        ub_exp_max_ptr_re,
                        tla.make_layout(tla.make_shape(q_tile_size_sub_re), tla.make_stride(1)),
                    )

                    if is_first_pv and is_last_pv:
                        # Single block: O = PV / sum
                        with tla.vec.func(mode='simd'):
                            pregFull_re0 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            for i_re0 in tla.range(q_tile_size_sub_re):
                                ub_sum_i_re0 = tla.tile_view(ub_sum_re, tla.make_shape(1), tla.make_coord(i_re0))
                                sum_reg_re0 = ub_sum_i_re0.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                for j_re0 in tla.range(c0, HEAD_DIM // VL_FLOAT_ELE, c1):
                                    cur_tile_re0 = tla.tile_view(ub_pv_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re0, j_re0))
                                    cur_reg_re0 = cur_tile_re0.load()
                                    div_reg_re0 = tla.div(cur_reg_re0, sum_reg_re0, mask=pregFull_re0)
                                    acc_tile_re0 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re0, j_re0))
                                    acc_tile_re0.store(div_reg_re0, mask=pregFull_re0)
                        tla.pipe_barrier(tla.pipes.ALL)
                        with tla.vec.func(mode='simd'):
                            pregFull_re1 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            preg_all_b16_re1 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                            for i_re1 in tla.range(q_tile_size_sub_re):
                                for j_re1 in tla.range(c0, HEAD_DIM // (2 * VL_FLOAT_ELE), c1):
                                    acc_tile_re1_0 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re1, 2 * j_re1))
                                    acc_tile_re1_1 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re1, 2 * j_re1 + c1))
                                    out_tile_re1 = tla.tile_view(ub_out_f16_re, tla.make_shape(1, 2 * VL_FLOAT_ELE), tla.make_coord(i_re1, j_re1))
                                    acc_reg_re1_0 = acc_tile_re1_0.load()
                                    acc_reg_re1_1 = acc_tile_re1_1.load()
                                    out_reg_re1_0 = acc_reg_re1_0.to(tla.Float16, cast_trait_zero, mask=pregFull_re1)
                                    out_reg_re1_1 = acc_reg_re1_1.to(tla.Float16, cast_trait_zero, mask=pregFull_re1)
                                    r0_re1, _ = tla.deinterleave(out_reg_re1_0, out_reg_re1_1)
                                    out_tile_re1.store(r0_re1, mask=preg_all_b16_re1)
                        tla.set_flag(rescale_v_mte3)
                        tla.wait_flag(rescale_v_mte3)
                        o_gm_re0 = tla.tile_view(mem_o_block, tla.make_shape(Q_BLOCK, HEAD_DIM), tla.make_coord(q_block_idx, c0))
                        o_sub_gm_re0 = tla.tile_view(o_gm_re0, tla.make_shape(q_tile_size_half_re, HEAD_DIM), tla.make_coord(vec_idx_re, c0))
                        tla.copy(o_sub_gm_re0, ub_out_f16_re)

                    elif is_first_pv and (not is_last_pv):
                        # First block: copy PV to acc
                        with tla.vec.func(mode='simd'):
                            pregFull_re2 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            for i_re2 in tla.range(q_tile_size_sub_re):
                                for j_re2 in tla.range(c0, HEAD_DIM // VL_FLOAT_ELE, c1):
                                    cur_tile_re2 = tla.tile_view(ub_pv_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re2, j_re2))
                                    acc_tile_re2 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re2, j_re2))
                                    cur_reg_re2 = cur_tile_re2.load()
                                    acc_tile_re2.store(cur_reg_re2, mask=pregFull_re2)

                    elif (not is_first_pv) and (not is_last_pv):
                        with tla.vec.func(mode='simd'):
                            pregFull_re3 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            for i_re3 in tla.range(q_tile_size_sub_re):
                                ub_exp_max_i_re3 = tla.tile_view(ub_exp_max_re, tla.make_shape(1), tla.make_coord(i_re3))
                                exp_max_reg_re3 = ub_exp_max_i_re3.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                for j_re3 in tla.range(c0, HEAD_DIM // VL_FLOAT_ELE, c1):
                                    pre_tile_re3 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re3, j_re3))
                                    cur_tile_re3 = tla.tile_view(ub_pv_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re3, j_re3))
                                    pre_reg_re3 = pre_tile_re3.load()
                                    cur_reg_re3 = cur_tile_re3.load()
                                    mul_reg_re3 = tla.mul(exp_max_reg_re3, pre_reg_re3, mask=pregFull_re3)
                                    add_reg_re3 = tla.add(mul_reg_re3, cur_reg_re3, mask=pregFull_re3)
                                    pre_tile_re3.store(add_reg_re3, mask=pregFull_re3)

                    else:
                        # Last block
                        with tla.vec.func(mode='simd'):
                            pregFull_re4 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            for i_re4 in tla.range(q_tile_size_sub_re):
                                ub_exp_max_i_re4 = tla.tile_view(ub_exp_max_re, tla.make_shape(1), tla.make_coord(i_re4))
                                exp_max_reg_re4 = ub_exp_max_i_re4.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                ub_sum_i_re4 = tla.tile_view(ub_sum_re, tla.make_shape(1), tla.make_coord(i_re4))
                                sum_reg_re4 = ub_sum_i_re4.load(params=NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                                for j_re4 in tla.range(c0, HEAD_DIM // VL_FLOAT_ELE, c1):
                                    pre_tile_re4 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re4, j_re4))
                                    cur_tile_re4 = tla.tile_view(ub_pv_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re4, j_re4))
                                    pre_reg_re4 = pre_tile_re4.load()
                                    cur_reg_re4 = cur_tile_re4.load()
                                    mul_reg_re4 = tla.mul(exp_max_reg_re4, pre_reg_re4, mask=pregFull_re4)
                                    add_reg_re4 = tla.add(mul_reg_re4, cur_reg_re4, mask=pregFull_re4)
                                    div_reg_re4 = tla.div(add_reg_re4, sum_reg_re4, mask=pregFull_re4)
                                    pre_tile_re4.store(div_reg_re4, mask=pregFull_re4)
                        tla.pipe_barrier(tla.pipes.ALL)
                        with tla.vec.func(mode='simd'):
                            pregFull_re5 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            preg_all_b16_re5 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                            for i_re5 in tla.range(q_tile_size_sub_re):
                                for j_re5 in tla.range(c0, HEAD_DIM // (2 * VL_FLOAT_ELE), c1):
                                    acc_tile_re5_0 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re5, 2 * j_re5))
                                    acc_tile_re5_1 = tla.tile_view(ub_acc_re, tla.make_shape(1, VL_FLOAT_ELE), tla.make_coord(i_re5, 2 * j_re5 + c1))
                                    out_tile_re5 = tla.tile_view(ub_out_f16_re, tla.make_shape(1, 2 * VL_FLOAT_ELE), tla.make_coord(i_re5, j_re5))
                                    acc_reg_re5_0 = acc_tile_re5_0.load()
                                    acc_reg_re5_1 = acc_tile_re5_1.load()
                                    out_reg_re5_0 = acc_reg_re5_0.to(tla.Float16, cast_trait_zero, mask=pregFull_re5)
                                    out_reg_re5_1 = acc_reg_re5_1.to(tla.Float16, cast_trait_zero, mask=pregFull_re5)
                                    r0_re5, _ = tla.deinterleave(out_reg_re5_0, out_reg_re5_1)
                                    out_tile_re5.store(r0_re5, mask=preg_all_b16_re5)
                        tla.set_flag(rescale_v_mte3)
                        tla.wait_flag(rescale_v_mte3)
                        o_gm_re5 = tla.tile_view(mem_o_block, tla.make_shape(Q_BLOCK, HEAD_DIM), tla.make_coord(q_block_idx, c0))
                        o_sub_gm_re5 = tla.tile_view(o_gm_re5, tla.make_shape(q_tile_size_half_re, HEAD_DIM), tla.make_coord(vec_idx_re, c0))
                        tla.copy(o_sub_gm_re5, ub_out_f16_re)

                    if ubOTmpBufId == c0:
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR, aiv_id=0)
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR, aiv_id=1)
                    # O->GM copy 完成, 释放给下一轮 rescale
                    tla.set_flag(mte3_ready_rescale)

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
        tla.wait_flag(fix_ready_mmad_qk_0)
        tla.wait_flag(fix_ready_mmad_qk_1)
        tla.wait_flag(fix_ready_mmad_pv_0)
        tla.wait_flag(fix_ready_mmad_pv_1)

        tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX, aiv_id=1)
        tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX, aiv_id=1)

        tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX, aiv_id=1)
        tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX, aiv_id=1)


    with tla.vector():
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3, aiv_id=0)
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3, aiv_id=1)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3, aiv_id=0)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3, aiv_id=1)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3, aiv_id=0)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3, aiv_id=1)
        tla.wait_flag(mte3_ready_softmax_0)
        tla.wait_flag(mte3_ready_softmax_1)
        tla.wait_flag(mte3_ready_rescale)


# Host侧：构造输入、编译、调用 kernel、精度校验
def _require_torch_npu(device_id: int) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Host-side tensors require PyTorch. pip install torch") from exc
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("This example requires torch_npu for device DLPack bindings.") from exc
    torch.npu.set_device(device_id)
    return torch


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {"options": "--npu-arch 3510"}


def _aicore_num(device: int = 0) -> int:
    import torch

    props = torch.npu.get_device_properties(int(device))
    for attr in ("aicore_num", "ai_core_num", "num_aicore", "cube_core_num"):
        value = getattr(props, attr, None)
        if value is not None:
            return max(1, int(value))
    return 1


def _create_tla_tensor(dev_buf: Any, layout_tag: Any = tla.arch.RowMajor) -> Any:
    return from_dlpack(dev_buf.contiguous(), layout_tag=layout_tag)


def _reshape_qk_to_2d(tensor: Any) -> Any:
    return tensor.reshape(-1, HEAD_DIM).contiguous()


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    torch.npu.set_device(args.device)
    device = "npu"
    print(
        "---",
        "flash_attention_infer (QK + softmax + PV + rescale, full FA)",
        f"BATCH={BATCH} Q_SEQ={Q_SEQ} KV_SEQ={KV_SEQ}",
        f"HEAD_NUM={HEAD_NUM} KV_HEAD_NUM={KV_HEAD_NUM}",
        f"HEAD_DIM={HEAD_DIM} KV_BLOCK_COUNT={KV_BLOCK_COUNT}",
        "---",
    )

    q_shape = (BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM)
    k_shape = (BATCH, KV_SEQ, KV_HEAD_NUM, HEAD_DIM)
    v_shape = (BATCH, KV_SEQ, KV_HEAD_NUM, HEAD_DIM)
    o_shape = (BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM)

    torch.manual_seed(42)
    torch_q = torch.rand(q_shape, dtype=torch.float16).to(device)
    torch_k = torch.rand(k_shape, dtype=torch.float16).to(device)
    torch_v = torch.rand(v_shape, dtype=torch.float16).to(device)
    torch_o = torch.full(o_shape, args.sentinel, dtype=torch.float16, device=device)

    tla_q = _create_tla_tensor(_reshape_qk_to_2d(torch_q), tla.arch.RowMajor)
    tla_k = _create_tla_tensor(_reshape_qk_to_2d(torch_k), tla.arch.ColumnMajor)
    tla_v = _create_tla_tensor(_reshape_qk_to_2d(torch_v), tla.arch.RowMajor)
    tla_o = _create_tla_tensor(_reshape_qk_to_2d(torch_o), tla.arch.RowMajor)
    mask_2d = torch.zeros((Q_SEQ, KV_SEQ), dtype=torch.int8, device=device)
    tla_mask = _create_tla_tensor(mask_2d, tla.arch.RowMajor)

    q_seqlen_list = [Q_SEQ] * BATCH
    kv_seqlen_list = [KV_SEQ] * BATCH
    td = compute_tiling(
        batch=BATCH,
        num_heads=HEAD_NUM,
        kv_heads=KV_HEAD_NUM,
        q_seqlen_list=q_seqlen_list,
        kv_seqlen_list=kv_seqlen_list,
        head_dim=HEAD_DIM,
        q_base_tile=Q_BLOCK,
        kv_base_tile=KV_BLOCK,
    )
    tiling_int_list = pack_tiling_int(td)
    tiling_tensor = torch.tensor(tiling_int_list, dtype=torch.int32, device='cpu').to(device)
    tla_tiling = _create_tla_tensor(tiling_tensor, tla.arch.RowMajor)

    actual_q_list = make_actual_seqlen(q_seqlen_list)
    actual_kv_list = make_actual_seqlen(kv_seqlen_list)
    actual_q = torch.tensor(actual_q_list, dtype=torch.int32)
    actual_kv = torch.tensor(actual_kv_list, dtype=torch.int32)
    tla_actual_q = _create_tla_tensor(actual_q.to(device), tla.arch.RowMajor)
    tla_actual_kv = _create_tla_tensor(actual_kv.to(device), tla.arch.RowMajor)

    core_num = _aicore_num(args.device)
    block_num = core_num if args.block_num <= 0 else args.block_num

    print(f"Launching kernel (block={block_num}, core_num={core_num})...", flush=True)
    artifact = tla.compile(
        flash_attention_infer_kernel,
        tla_q,
        tla_k,
        tla_v,
        tla_o,
        tla_mask,
        tla_tiling,
        tla_actual_q,
        tla_actual_kv,
        **_runtime_kwargs(args),
    )

    artifact(
        tla_q, tla_k, tla_v, tla_o, tla_mask, tla_tiling, tla_actual_q, tla_actual_kv,
        block=block_num,
    )

    torch.npu.synchronize()

    scale = 1.0 / (HEAD_DIM ** 0.5)
    q_bnsd = torch_q.permute(0, 2, 1, 3).contiguous()
    k_bnsd = torch_k.permute(0, 2, 1, 3).contiguous()
    v_bnsd = torch_v.permute(0, 2, 1, 3).contiguous()
    if KV_HEAD_NUM != HEAD_NUM:
        k_bnsd = k_bnsd.repeat_interleave(GROUP_SIZE, dim=1)
        v_bnsd = v_bnsd.repeat_interleave(GROUP_SIZE, dim=1)

    q_dsl = torch_q.cpu().numpy()
    k_dsl = torch_k.cpu().numpy()
    v_dsl = torch_v.cpu().numpy()

    del q_bnsd, k_bnsd, v_bnsd
    torch.npu.empty_cache()

    group_num = HEAD_NUM // KV_HEAD_NUM

    def group_matmul_ref(head, kv_head, left, right):
        """对齐 C++ gen_data group_matmul: 按 kv_head 分组, .astype(f32)"""
        score = None
        for i in range(kv_head):
            group_score = np.matmul(
                left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                right[i:(i + 1), :, :].astype(np.float32),
            )
            score = group_score if score is None else np.concatenate((score, group_score), 0)
        return score

    def softmax_ref(sim):
        """对齐 C++ gen_data softmax (sink=None)"""
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    num_tokens = BATCH * Q_SEQ
    golden_output = np.zeros((num_tokens, HEAD_NUM, HEAD_DIM), dtype=np.float16)

    for b in range(BATCH):
        q_i = q_dsl[b]
        keys = k_dsl[b]
        values = v_dsl[b]

        query = np.transpose(q_i, (1, 0, 2))
        key = np.transpose(keys, (1, 2, 0))
        sim_high = group_matmul_ref(query.shape[0], key.shape[0], query, key)
        sim_high = sim_high * scale

        p_high = softmax_ref(sim_high)
        p = p_high.astype(np.float16)
        value = np.transpose(values, (1, 0, 2))
        out = group_matmul_ref(query.shape[0], value.shape[0], p, value)
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(np.float16)

        golden_output[b * Q_SEQ:(b + 1) * Q_SEQ, :, :] = out

    golden = golden_output.astype(np.float32).reshape(BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM)

    actual_np = torch_o.cpu().numpy().astype(np.float16).astype(np.float32)

    actual_t = torch.from_numpy(actual_np.copy())
    sentinel_t = torch.full_like(actual_t, args.sentinel)
    unchanged = torch.isclose(actual_t, sentinel_t, rtol=0.0, atol=UNCHANGED_THRESHOLD)

    a = actual_np.flatten()
    g = golden.flatten()
    diff = np.abs(a - g)

    max_abs = float(np.max(diff))

    print(
        "compile_ok=True "
        f"host=torch_npu "
        f"BATCH={BATCH} Q_SEQ={Q_SEQ} KV_SEQ={KV_SEQ} "
        f"HEAD_NUM={HEAD_NUM} KV_HEAD_NUM={KV_HEAD_NUM} HEAD_DIM={HEAD_DIM} "
        f"Q_BLOCK_COUNT={Q_BLOCK_COUNT} KV_BLOCK_COUNT={KV_BLOCK_COUNT}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"O unchanged (sentinel)? {bool(unchanged.all())}")
    print(f"O changed count={int((~unchanged).sum().item())} / {actual_t.numel()}")
    passed = max_abs < THRESHOLD
    print(f"passed? {passed}")

    return 0 if passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="flash_attention_infer: full FA kernel (QK + softmax + PV + rescale). "
                    "Used to isolate 507015 to the PV cube block."
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id.")
    parser.add_argument(
        "--block-num", type=int, default=-1,
        help="Launch block count (<=0 means use full AICore num from device).",
    )
    parser.add_argument(
        "--sentinel", type=float, default=-7.0, help="Initial O value (sentinel).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())