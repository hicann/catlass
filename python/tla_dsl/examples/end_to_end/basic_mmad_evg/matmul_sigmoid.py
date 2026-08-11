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

import argparse
from pathlib import Path

import catlass as tla
from catlass.runtime import from_dlpack
from catlass.types import dtype_size_bytes

# ---- kernel constants + @tla.kernel ----
# Target device: Ascend950.
L0C_SIZE = 256 * 1024
UB_SIZE = 248 * 1024
BYTE_PER_C0 = 32

# Default tiling: L1 256×256×128, L0 256×256×32.
l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32

# One SIMD register = 256B = 64 fp32. REG_M=1 → contiguous row segment + tail mask.
REG_M = 1

# After this many MN tiles per core, AIC/AIV exchange the reverse cross-core flag.
REVERSE_DEPTH = 15

# Host may rewrite these before compile.
DTYPE_A = tla.Float32
DTYPE_B = tla.Float32
DTYPE_C = tla.Float32  # L0C accumulator (always fp32)
DTYPE_GM_C = tla.Float32  # GM D / workspace / UB epilogue (follows dtype-c)
ENABLE_UNIT_FLAG = True

# UB nodes counted for sizing: load Acc and compute (store reuses Out).
EVG_UB_NODES = 2
# EVG UB multi-buffer depth (1 or 2 physical slots). Manual edit only, like l1_tn.
EVG_UB_STAGES = 2

# MN tile swizzle: direction 0 = Zn (prefer when m>n), 1 = Nz (prefer when m<=n).
# Host rewrites SWIZZLE_DIRECTION before compile.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 1


def _refresh_ub_derived() -> None:
    """Recompute UB_SLOT_ELEMS / SIMD_LANES after DTYPE / EVG_UB_STAGES edits."""
    global _ELEM_C_BYTES, UB_SLOT_ELEMS, SIMD_LANES
    _ELEM_C_BYTES = dtype_size_bytes(DTYPE_GM_C.dtype)
    _ub_slot_raw = UB_SIZE // EVG_UB_NODES // EVG_UB_STAGES // _ELEM_C_BYTES
    UB_SLOT_ELEMS = (_ub_slot_raw // BYTE_PER_C0) * BYTE_PER_C0
    # SIMD lanes for ElementC; must match ``tla.update_mask`` (256B / sizeof).
    SIMD_LANES = 256 // _ELEM_C_BYTES

_ELEM_C_BYTES = 0
UB_SLOT_ELEMS = 0
SIMD_LANES = 0
_refresh_ub_derived()

@tla.kernel
def matmul_evg_sigmoid_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_d: tla.Tensor,
    mem_workspace: tla.Tensor,
) -> None:
    """Cube GEMM to GM workspace; vector epilogue D = sigmoid(Acc)."""
    m = mem_a.origin_shape[0]
    n = mem_b.origin_shape[1]
    k = mem_a.origin_shape[1]
    c0 = 0
    c1 = 1


    # ---- soft-flags (cube pingpong) ----
    l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_data_ready = tla.flag("l1a1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_data_ready = tla.flag("l1b0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_data_ready = tla.flag("l1b1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_available = tla.flag("l1a0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_available = tla.flag("l1a1_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_available = tla.flag("l1b0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_available = tla.flag("l1b1_available", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_available = tla.flag("l0a0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_available = tla.flag("l0a1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_available = tla.flag("l0b0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_available = tla.flag("l0b1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0c_data_ready = tla.flag("l0c_data_ready", tla.arch.CUBE, tla.arch.FIX)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)

    # ---- EVG/UB soft-flags ----
    ub_acc0_data_ready = tla.flag("ub_acc0_data_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_acc1_data_ready = tla.flag("ub_acc1_data_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_out0_data_ready = tla.flag("ub_out0_data_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_out1_data_ready = tla.flag("ub_out1_data_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_acc0_available = tla.flag("ub_acc0_available", tla.arch.VECTOR, tla.arch.MTE2)
    ub_acc1_available = tla.flag("ub_acc1_available", tla.arch.VECTOR, tla.arch.MTE2)
    ub_out0_available = tla.flag("ub_out0_available", tla.arch.MTE3, tla.arch.VECTOR)
    ub_out1_available = tla.flag("ub_out1_available", tla.arch.MTE3, tla.arch.VECTOR)

    # ---- cross-core flags ----
    # AIC FixPipe → set forward; AIV MTE2 → wait forward; every REVERSE_DEPTH tiles reverse.
    # No per-tile aiv_finish, so FixPipe(N+1) can overlap epilogue(N).
    aic_finish = tla.cross_flag("aic_finish", mode=2)
    aic_finish_rv = tla.cross_flag("aic_finish_rv", mode=2)

    # ---- L1/L0/L0C allocates ----
    l1a0_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)

    l0a0_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)

    l0c_ptr = tla.allocate(l0_tm * l0_tn, DTYPE_C, tla.AddressSpace.l0c, 512)

    # ---- UB allocates ----
    ub_acc_ptr0 = tla.allocate(UB_SLOT_ELEMS, DTYPE_GM_C, tla.AddressSpace.ub, 256)
    ub_out_ptr0 = tla.allocate(UB_SLOT_ELEMS, DTYPE_GM_C, tla.AddressSpace.ub, 256)
    ub_acc_ptr1 = ub_acc_ptr0
    ub_out_ptr1 = ub_out_ptr0
    if tla.const_expr(EVG_UB_STAGES >= 2):
        ub_acc_ptr1 = tla.allocate(UB_SLOT_ELEMS, DTYPE_GM_C, tla.AddressSpace.ub, 256)
        ub_out_ptr1 = tla.allocate(UB_SLOT_ELEMS, DTYPE_GM_C, tla.AddressSpace.ub, 256)

    # ---- grid / swizzle setup ----
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    total_blocks = grid_m * grid_n
    # Folded at compile time via SWIZZLE_DIRECTION (host-set).
    if tla.const_expr(SWIZZLE_DIRECTION == 0):
        swizzle_tile_count = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
    else:
        swizzle_tile_count = (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET

    # ---- cube: prime flags; MN tile loop; K pingpong; FixPipe; handoff; drain ----
    with tla.cube():
        tla.set_flag(l1a0_available)
        tla.set_flag(l1a1_available)
        tla.set_flag(l1b0_available)
        tla.set_flag(l1b1_available)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0a1_available)
        tla.set_flag(l0b0_available)
        tla.set_flag(l0b1_available)
        tla.set_flag(l0c_available)

        l1_buf_idx = c0
        l0_buf_idx = c0

        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num())
        for block_linear in block_range:
            # Map linear MN task id → (block_row, block_col) with swizzle (must stay in kernel AST).
            block_row = c0
            block_col = c0
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                swizzle_tile_idx = block_linear // (SWIZZLE_OFFSET * grid_n)
                in_tile_idx = block_linear % (SWIZZLE_OFFSET * grid_n)
                swizzle_n_rows = (
                    SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_m - SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = swizzle_tile_idx * SWIZZLE_OFFSET + in_tile_idx % swizzle_n_rows
                block_col = in_tile_idx // swizzle_n_rows
                block_col = (
                    (grid_n - block_col - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_col
                )
            else:
                swizzle_tile_idx = block_linear // (SWIZZLE_OFFSET * grid_m)
                in_tile_idx = block_linear % (SWIZZLE_OFFSET * grid_m)
                swizzle_n_cols = (
                    SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_n - SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = in_tile_idx // swizzle_n_cols
                block_col = swizzle_tile_idx * SWIZZLE_OFFSET + in_tile_idx % swizzle_n_cols
                block_row = (
                    (grid_m - block_row - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_row
                )
            gm_a_by_core = tla.tile_view(
                mem_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                mem_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_workspace_by_core = tla.tile_view(
                mem_workspace,
                tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk - 1) // l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_workspace_by_core)

            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.wait_flag(l0c_available)
            for k_l1 in k_l1_range:
                gm_a_l1 = tla.tile_view(
                    gm_a_by_core, tla.make_shape(l1_tm, l1_tk), tla.make_coord(c0, k_l1)
                )
                gm_b_l1 = tla.tile_view(
                    gm_b_by_core, tla.make_shape(l1_tk, l1_tn), tla.make_coord(k_l1, c0)
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                )
                if l1_buf_idx == c0:
                    tla.wait_flag(l1a0_available)
                else:
                    tla.wait_flag(l1a1_available)
                tla.copy(l1_a, gm_a_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_data_ready)
                else:
                    tla.set_flag(l1a1_data_ready)

                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_available)
                else:
                    tla.wait_flag(l1b1_available)
                tla.copy(l1_b, gm_b_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_data_ready)
                else:
                    tla.set_flag(l1b1_data_ready)

                k_l0_count = (l1_a.origin_shape[1] + l0_tk - 1) // l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)

                for k_l0 in k_l0_range:
                    l1_a_l0 = tla.tile_view(
                        l1_a, tla.make_shape(l0_tm, l0_tk), tla.make_coord(c0, k_l0)
                    )
                    l1_b_l0 = tla.tile_view(
                        l1_b, tla.make_shape(l0_tk, l0_tn), tla.make_coord(k_l0, c0)
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_l0
                    )
                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1a0_data_ready)
                        else:
                            tla.wait_flag(l1a1_data_ready)

                    if l0_buf_idx == c0:
                        tla.wait_flag(l0a0_available)
                    else:
                        tla.wait_flag(l0a1_available)
                    tla.copy(l0_a, l1_a_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1a0_available)
                        else:
                            tla.set_flag(l1a1_available)

                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1b0_data_ready)
                        else:
                            tla.wait_flag(l1b1_data_ready)
                    if l0_buf_idx == c0:
                        tla.wait_flag(l0b0_available)
                    else:
                        tla.wait_flag(l0b1_available)
                    tla.copy(l0_b, l1_b_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_available)
                        else:
                            tla.set_flag(l1b1_available)

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = 0
                    if tla.const_expr(ENABLE_UNIT_FLAG):
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            # FixPipe: write L0C tile to GM workspace (full tile, no SPLIT_M).
            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.set_flag(l0c_data_ready)
                tla.wait_flag(l0c_data_ready)
                tla.copy(gm_workspace_by_core, l0_c)
                tla.set_flag(l0c_available)
            else:
                tla.copy(
                    gm_workspace_by_core,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )

            # Notify AIV that this MN tile is ready in GM workspace; reverse every REVERSE_DEPTH tiles.
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
            per_core_tile_idx = (block_linear - tla.arch.block_idx()) // tla.arch.block_num()
            if (per_core_tile_idx + 1) % REVERSE_DEPTH == 0:
                tla.cross_core_wait_flag(aic_finish_rv, tla.arch.FIX)

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)
        tla.pipe_barrier(tla.pipes.ALL)

    # ---- vector: prime flags; MN loop; load / compute / store epilogue ----
    with tla.vector():
        aiv_sub_idx = tla.arch.sub_block_idx()
        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num())

        tla.set_flag(ub_acc0_available)
        tla.set_flag(ub_out0_available)
        if tla.const_expr(EVG_UB_STAGES >= 2):
            tla.set_flag(ub_acc1_available)
            tla.set_flag(ub_out1_available)

        for block_linear in block_range:
            # Map linear MN task id → (block_row, block_col) with swizzle (must stay in kernel AST).
            block_row = c0
            block_col = c0
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                swizzle_tile_idx = block_linear // (SWIZZLE_OFFSET * grid_n)
                in_tile_idx = block_linear % (SWIZZLE_OFFSET * grid_n)
                swizzle_n_rows = (
                    SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_m - SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = swizzle_tile_idx * SWIZZLE_OFFSET + in_tile_idx % swizzle_n_rows
                block_col = in_tile_idx // swizzle_n_rows
                block_col = (
                    (grid_n - block_col - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_col
                )
            else:
                swizzle_tile_idx = block_linear // (SWIZZLE_OFFSET * grid_m)
                in_tile_idx = block_linear % (SWIZZLE_OFFSET * grid_m)
                swizzle_n_cols = (
                    SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_n - SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = in_tile_idx // swizzle_n_cols
                block_col = swizzle_tile_idx * SWIZZLE_OFFSET + in_tile_idx % swizzle_n_cols
                block_row = (
                    (grid_m - block_row - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_row
                )

            # Wait until AIC has written this tile to GM workspace (MTE2).
            # Every REVERSE_DEPTH tiles, set the reverse cross-core flag.
            tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
            per_core_tile_idx = (block_linear - tla.arch.block_idx()) // tla.arch.block_num()
            if (per_core_tile_idx + 1) % REVERSE_DEPTH == 0:
                tla.cross_core_set_flag(aic_finish_rv, tla.arch.MTE2)

            gm_workspace_by_core = tla.tile_view(
                mem_workspace,
                tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )
            gm_d_by_core = tla.tile_view(
                mem_d, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
            )

            # Split the MN tile on M across the two AIV sub-blocks.
            actual_m = gm_d_by_core.origin_shape[0]
            actual_n = gm_d_by_core.origin_shape[1]
            # SPLIT_M: request ceil(M/2) rows × full N; AIV1 may clip when M is odd.
            aiv_m_req = (actual_m + 1) // 2
            gm_acc = tla.tile_view(
                gm_workspace_by_core, tla.make_shape(aiv_m_req, actual_n), tla.make_coord(aiv_sub_idx, c0)
            )
            gm_result = tla.tile_view(
                gm_d_by_core, tla.make_shape(aiv_m_req, actual_n), tla.make_coord(aiv_sub_idx, c0)
            )
            aiv_m = gm_result.origin_shape[0]
            aiv_n = gm_result.origin_shape[1]

            ub_stage_idx = c0
            if aiv_n <= UB_SLOT_ELEMS:
                ub_row_stride = ((aiv_n + BYTE_PER_C0 - 1) // BYTE_PER_C0) * BYTE_PER_C0
                max_rows = UB_SLOT_ELEMS // ub_row_stride
                if max_rows < 1:
                    max_rows = 1
                n_epilogue_row_tiles = (aiv_m + max_rows - 1) // max_rows

                for tile_i in tla.range(c0, n_epilogue_row_tiles, c1):
                    gm_c_tile = tla.tile_view(
                        gm_acc,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    gm_d_tile = tla.tile_view(
                        gm_result,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    tile_rows = gm_c_tile.origin_shape[0]
                    tile_cols = gm_c_tile.origin_shape[1]
                    ub_layout = tla.make_layout(
                        tla.make_shape(tile_rows, tile_cols),
                        tla.make_stride(ub_row_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        use_ub_stage1 = ub_stage_idx != c0
                    else:
                        use_ub_stage1 = False
                    ub_acc = tla.make_tensor(
                        ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                    )
                    ub_result = tla.make_tensor(
                        ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                    )

                    # Wait UB slot free → load accumulator → signal compute may start.
                    if use_ub_stage1:
                        tla.wait_flag(ub_acc1_available)
                    else:
                        tla.wait_flag(ub_acc0_available)
                    tla.copy(ub_acc, gm_c_tile)
                    if use_ub_stage1:
                        tla.set_flag(ub_acc1_data_ready)
                        tla.wait_flag(ub_acc1_data_ready)
                        tla.wait_flag(ub_out1_available)
                    else:
                        tla.set_flag(ub_acc0_data_ready)
                        tla.wait_flag(ub_acc0_data_ready)
                        tla.wait_flag(ub_out0_available)

                    with tla.vec.func(mode="simd"):
                        for row_i in tla.range(c0, tile_rows, c1):
                            remaining = tile_cols
                            for col_j in tla.range(c0, n_simd_strips, c1):
                                c_chunk = tla.tile_view(
                                    ub_acc,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                result_chunk = tla.tile_view(
                                    ub_result,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                tail, remaining = tla.update_mask(
                                    remaining, dtype=DTYPE_GM_C
                                )
                                # Sigmoid(x) = 1 / (1 + exp(-x))
                                # Build ones via vector SSA (tla.div rejects scalar lhs).
                                xv = c_chunk.load()
                                den = 1.0 + tla.exp(
                                    tla.neg(xv, mask=tail), mask=tail
                                )
                                ones = xv * 0.0 + 1.0
                                result_chunk.store(
                                    tla.div(ones, den, mask=tail), mask=tail
                                )

                    if use_ub_stage1:
                        tla.set_flag(ub_acc1_available)
                        tla.set_flag(ub_out1_data_ready)
                        tla.wait_flag(ub_out1_data_ready)
                    else:
                        tla.set_flag(ub_acc0_available)
                        tla.set_flag(ub_out0_data_ready)
                        tla.wait_flag(ub_out0_data_ready)
                    tla.copy(gm_d_tile, ub_result)
                    if use_ub_stage1:
                        tla.set_flag(ub_out1_available)
                    else:
                        tla.set_flag(ub_out0_available)

                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        ub_stage_idx = c1 - ub_stage_idx
            else:
                n_col_tiles = (aiv_n + UB_SLOT_ELEMS - 1) // UB_SLOT_ELEMS
                for row_i in tla.range(c0, aiv_m, c1):
                    for col_tile in tla.range(c0, n_col_tiles, c1):
                        gm_c_tile = tla.tile_view(
                            gm_acc,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        gm_d_tile = tla.tile_view(
                            gm_result,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        tile_rows = gm_c_tile.origin_shape[0]
                        tile_cols = gm_c_tile.origin_shape[1]
                        ub_row_stride = (
                            (tile_cols + BYTE_PER_C0 - 1) // BYTE_PER_C0
                        ) * BYTE_PER_C0
                        ub_layout = tla.make_layout(
                            tla.make_shape(tile_rows, tile_cols),
                            tla.make_stride(ub_row_stride, 1),
                            layoutTag=tla.arch.RowMajor,
                        )
                        n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                        if tla.const_expr(EVG_UB_STAGES >= 2):
                            use_ub_stage1 = ub_stage_idx != c0
                        else:
                            use_ub_stage1 = False
                        ub_acc = tla.make_tensor(
                            ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                        )
                        ub_result = tla.make_tensor(
                            ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                        )

                        if use_ub_stage1:
                            tla.wait_flag(ub_acc1_available)
                        else:
                            tla.wait_flag(ub_acc0_available)
                        tla.copy(ub_acc, gm_c_tile)
                        if use_ub_stage1:
                            tla.set_flag(ub_acc1_data_ready)
                            tla.wait_flag(ub_acc1_data_ready)
                            tla.wait_flag(ub_out1_available)
                        else:
                            tla.set_flag(ub_acc0_data_ready)
                            tla.wait_flag(ub_acc0_data_ready)
                            tla.wait_flag(ub_out0_available)

                        with tla.vec.func(mode="simd"):
                            for r_i in tla.range(c0, tile_rows, c1):
                                remaining = tile_cols
                                for col_j in tla.range(c0, n_simd_strips, c1):
                                    c_chunk = tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    result_chunk = tla.tile_view(
                                        ub_result,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    tail, remaining = tla.update_mask(
                                        remaining, dtype=DTYPE_GM_C
                                    )
                                    xv = c_chunk.load()
                                    den = 1.0 + tla.exp(
                                        tla.neg(xv, mask=tail), mask=tail
                                    )
                                    ones = xv * 0.0 + 1.0
                                    result_chunk.store(
                                        tla.div(ones, den, mask=tail), mask=tail
                                    )

                        if use_ub_stage1:
                            tla.set_flag(ub_acc1_available)
                            tla.set_flag(ub_out1_data_ready)
                            tla.wait_flag(ub_out1_data_ready)
                        else:
                            tla.set_flag(ub_acc0_available)
                            tla.set_flag(ub_out0_data_ready)
                            tla.wait_flag(ub_out0_data_ready)
                        tla.copy(gm_d_tile, ub_result)
                        if use_ub_stage1:
                            tla.set_flag(ub_out1_available)
                        else:
                            tla.set_flag(ub_out0_available)

                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        ub_stage_idx = c1 - ub_stage_idx

        tla.wait_flag(ub_acc0_available)
        tla.wait_flag(ub_out0_available)
        if tla.const_expr(EVG_UB_STAGES >= 2):
            tla.wait_flag(ub_acc1_available)
            tla.wait_flag(ub_out1_available)

        tla.pipe_barrier(tla.pipes.ALL)

# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"
DESCRIPTION = "Matmul EVG sigmoid: D=sigmoid(A@B) via L0C→GM workspace + AIV; dynamic GM."


def golden(a, b, out_dtype):
    import torch

    expected = torch.sigmoid(a.to(torch.float32) @ b.to(torch.float32))
    if out_dtype in (torch.float16, torch.bfloat16):
        expected = expected.to(out_dtype).to(torch.float32)
    return expected


def run(args: argparse.Namespace) -> int:
    import sys
    import torch
    import torch_npu  # noqa: F401

    mod = sys.modules[__name__]
    tla_of = {"f16": tla.Float16, "bf16": tla.BFloat16, "f32": tla.Float32}
    torch_of = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    da, db, dc = args.dtype_a, args.dtype_b, args.dtype_c
    la, lb = args.layout_a, args.layout_b
    mi, ni, ki = int(args.m), int(args.n), int(args.k)
    mod.DTYPE_A = tla_of[da]
    mod.DTYPE_B = tla_of[db]
    mod.DTYPE_C = tla.Float32
    mod.DTYPE_GM_C = tla_of[dc]
    mod.SWIZZLE_DIRECTION = 0 if mi > ni else 1
    mod._refresh_ub_derived()

    def create_tla_tensor(buf, layout: str):
        storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
        tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
        return from_dlpack(storage, layout_tag=tag).mark_layout_dynamic()

    cache_dir = str(Path(args.cache_dir).expanduser().resolve())

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        print(f"--- mnk=({mi},{ni},{ki}) layout={la}/{lb} dtype={da}/{db}/{dc} ---")
        torch.npu.manual_seed(0)
        a = torch.rand(mi, ki, dtype=torch_of[da], device="npu") * 10.0 - 5.0
        b = torch.rand(ki, ni, dtype=torch_of[db], device="npu") * 10.0 - 5.0
        d = torch.zeros((mi, ni), dtype=torch_of[dc], device="npu")
        workspace = torch.zeros((mi, ni), dtype=torch_of[dc], device="npu")
        expected = golden(a, b, torch_of[dc])

        ta, tb, td, tw = (
            create_tla_tensor(a, la),
            create_tla_tensor(b, lb),
            create_tla_tensor(d, "row"),
            create_tla_tensor(workspace, "row"),
        )
        artifact = tla.compile(
            matmul_evg_sigmoid_kernel,
            ta, tb, td, tw,
            arch_scope="aic.c310",
            cache=not args.no_cache,
            cache_dir=cache_dir,
            force_recompile=args.force_recompile,
        )
        block_dim = max(
            1,
            args.block_dim if args.block_dim != -1 else tla.get_aicore_num(args.device),
        )
        artifact(ta, tb, td, tw, block_dim=block_dim)
        torch.npu.synchronize()
        got = d.detach().to(device="cpu", dtype=torch.float32)
        if dc == "bf16":
            rtol = (1.0 / 128.0) if ki < 2048 else (1.0 / 64.0)
            floor = 1.0 / 256.0
        else:
            rtol = (1.0 / 256.0) if ki < 2048 else (1.0 / 128.0)
            floor = 1.0
        exp = expected.detach().to(device="cpu", dtype=torch.float32)
        passed = bool(
            ((got - exp).abs() <= rtol * torch.maximum(torch.full_like(exp, floor), exp.abs())).all()
        )
        print(f"passed={passed} cache_key={artifact.cache_key}")
        print(f"kernel.o={artifact.kernel_binary_path}")
        return 0 if passed else 1
    finally:
        tla.finalize()


def main() -> int:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--layout-a", choices=("row", "col"), default="row")
    p.add_argument("--layout-b", choices=("row", "col"), default="row")
    p.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f32")
    p.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f32")
    p.add_argument("--dtype-c", choices=("f16", "f32"), default="f32")
    p.add_argument("--block-dim", type=int, default=-1)
    p.add_argument("--sentinel", type=float, default=-7.0)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--force-recompile", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
