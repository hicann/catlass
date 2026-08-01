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
from typing import Any, Literal

import catlass as tla
from catlass.runtime import from_dlpack
from catlass.types import dtype_size_bytes

# ---- kernel constants + @tla.kernel ----
# Target device: Ascend950.
L0C_SIZE = 256 * 1024
UB_SIZE = 248 * 1024
BYTE_PER_C0 = 32

# Default problem size and tiling: L1 256×256×128, L0 256×256×32.
m = 256
n = 256
k = 256
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
# EVG UB multi-buffer depth (1 or 2 physical slots).
# Host may rewrite before compile, e.g. ``EVG_UB_STAGES = 128``.
EVG_UB_STAGES = 2

# MN tile swizzle: direction 0 = Zn (prefer when m>n), 1 = Nz (prefer when m<=n).
# Host rewrites SWIZZLE_DIRECTION before compile.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 1

def _elem_c_bytes() -> int:
    return dtype_size_bytes(DTYPE_GM_C.dtype)

def _simd_lanes() -> int:
    """SIMD lanes for ElementC; must match ``tla.update_mask`` (256B / sizeof)."""
    return 256 // _elem_c_bytes()

def evg_compute_length(stages: int | None = None) -> int:
    """Max elements per UB epilogue slot: floor(UB/nodes/stages/elem_bytes) to BYTE_PER_C0."""
    s = EVG_UB_STAGES if stages is None else int(stages)
    if s < 1:
        raise ValueError(f"EVG_UB_STAGES must be >= 1, got {s}")
    raw = UB_SIZE // EVG_UB_NODES // s // _elem_c_bytes()
    return (raw // BYTE_PER_C0) * BYTE_PER_C0

@tla.kernel
def matmul_evg_silu_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_d: tla.Tensor,
    mem_workspace: tla.Tensor,
) -> None:
    """Cube GEMM to GM workspace; vector epilogue D = silu(Acc) = x / (1 + exp(-x))."""
    m = mem_a.origin_shape[0]
    n = mem_b.origin_shape[1]
    k = mem_a.origin_shape[1]
    c0 = 0
    c1 = 1

    ub_slot_elems = evg_compute_length(EVG_UB_STAGES)
    simd_lanes = _simd_lanes()

    # ---- soft-flags (cube pingpong) ----
    l1a0_copy_end = tla.flag("l1a0_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_copy_end = tla.flag("l1a1_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_copy_end = tla.flag("l1b0_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_copy_end = tla.flag("l1b1_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_copy_start = tla.flag("l1a0_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_copy_start = tla.flag("l1a1_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_copy_start = tla.flag("l1b0_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_copy_start = tla.flag("l1b1_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_copy_start = tla.flag("l0a0_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_copy_start = tla.flag("l0a1_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_copy_start = tla.flag("l0b0_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_copy_start = tla.flag("l0b1_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0_copy_end = tla.flag("l0_copy_end", tla.arch.MTE1, tla.arch.CUBE)
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    fix_done = tla.flag("fix_done", tla.arch.FIX, tla.arch.CUBE)

    # ---- EVG/UB soft-flags ----
    ub_load_ready_0 = tla.flag("ub_load_ready_0", tla.arch.VECTOR, tla.arch.MTE2)
    ub_load_ready_1 = tla.flag("ub_load_ready_1", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded_0 = tla.flag("ub_loaded_0", tla.arch.MTE2, tla.arch.VECTOR)
    ub_loaded_1 = tla.flag("ub_loaded_1", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done_0 = tla.flag("vec_done_0", tla.arch.VECTOR, tla.arch.MTE3)
    vec_done_1 = tla.flag("vec_done_1", tla.arch.VECTOR, tla.arch.MTE3)
    ub_store_done_0 = tla.flag("ub_store_done_0", tla.arch.MTE3, tla.arch.VECTOR)
    ub_store_done_1 = tla.flag("ub_store_done_1", tla.arch.MTE3, tla.arch.VECTOR)

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
    ub_acc_ptr0 = tla.allocate(ub_slot_elems, DTYPE_GM_C, tla.AddressSpace.ub, 256)
    ub_out_ptr0 = tla.allocate(ub_slot_elems, DTYPE_GM_C, tla.AddressSpace.ub, 256)
    ub_acc_ptr1 = ub_acc_ptr0
    ub_out_ptr1 = ub_out_ptr0
    if EVG_UB_STAGES >= 2:
        ub_acc_ptr1 = tla.allocate(ub_slot_elems, DTYPE_GM_C, tla.AddressSpace.ub, 256)
        ub_out_ptr1 = tla.allocate(ub_slot_elems, DTYPE_GM_C, tla.AddressSpace.ub, 256)

    # ---- grid / swizzle setup ----
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    total_blocks = grid_m * grid_n
    # Folded at compile time via SWIZZLE_DIRECTION (host-set), like EVG_UB_STAGES.
    swizzle_tile_count = (
        (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
        if SWIZZLE_DIRECTION == 0
        else (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
    )

    # ---- cube: prime flags; MN tile loop; K pingpong; FixPipe; handoff; drain ----
    with tla.cube():
        tla.set_flag(l1a0_copy_start)
        tla.set_flag(l1a1_copy_start)
        tla.set_flag(l1b0_copy_start)
        tla.set_flag(l1b1_copy_start)
        tla.set_flag(l0a0_copy_start)
        tla.set_flag(l0a1_copy_start)
        tla.set_flag(l0b0_copy_start)
        tla.set_flag(l0b1_copy_start)
        tla.set_flag(fix_done)

        l1_buf_idx = c0
        l0_buf_idx = c0

        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_dim())
        for block_linear in block_range:
            # Map linear MN task id → (block_row, block_col) with swizzle (must stay in kernel AST).
            block_row = c0
            block_col = c0
            if SWIZZLE_DIRECTION == 0:
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

            if not ENABLE_UNIT_FLAG:
                tla.wait_flag(fix_done)
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
                    tla.wait_flag(l1a0_copy_start)
                else:
                    tla.wait_flag(l1a1_copy_start)
                tla.copy(l1_a, gm_a_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_copy_end)
                else:
                    tla.set_flag(l1a1_copy_end)

                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_copy_start)
                else:
                    tla.wait_flag(l1b1_copy_start)
                tla.copy(l1_b, gm_b_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_copy_end)
                else:
                    tla.set_flag(l1b1_copy_end)

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
                            tla.wait_flag(l1a0_copy_end)
                        else:
                            tla.wait_flag(l1a1_copy_end)

                    if l0_buf_idx == c0:
                        tla.wait_flag(l0a0_copy_start)
                    else:
                        tla.wait_flag(l0a1_copy_start)
                    tla.copy(l0_a, l1_a_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1a0_copy_start)
                        else:
                            tla.set_flag(l1a1_copy_start)

                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1b0_copy_end)
                        else:
                            tla.wait_flag(l1b1_copy_end)
                    if l0_buf_idx == c0:
                        tla.wait_flag(l0b0_copy_start)
                    else:
                        tla.wait_flag(l0b1_copy_start)
                    tla.copy(l0_b, l1_b_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_copy_start)
                        else:
                            tla.set_flag(l1b1_copy_start)

                    tla.set_flag(l0_copy_end)
                    tla.wait_flag(l0_copy_end)

                    unit_flag = 0
                    if ENABLE_UNIT_FLAG:
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_copy_start)
                        tla.set_flag(l0b0_copy_start)
                    else:
                        tla.set_flag(l0a1_copy_start)
                        tla.set_flag(l0b1_copy_start)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            # FixPipe: write L0C tile to GM workspace (full tile, no SPLIT_M).
            if not ENABLE_UNIT_FLAG:
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)
                tla.copy(gm_workspace_by_core, l0_c)
                tla.set_flag(fix_done)
            else:
                tla.copy(
                    gm_workspace_by_core,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )

            # Notify AIV that this MN tile is ready in GM workspace; reverse every REVERSE_DEPTH tiles.
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
            per_core_tile_idx = (block_linear - tla.arch.block_idx()) // tla.arch.block_dim()
            if (per_core_tile_idx + 1) % REVERSE_DEPTH == 0:
                tla.cross_core_wait_flag(aic_finish_rv, tla.arch.FIX)

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)
        tla.wait_flag(fix_done)
        tla.pipe_barrier(tla.pipes.ALL)

    # ---- vector: prime flags; MN loop; load / compute / store epilogue ----
    with tla.vector():
        aiv_sub_idx = tla.arch.sub_block_idx()
        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_dim())

        tla.set_flag(ub_load_ready_0)
        tla.set_flag(ub_store_done_0)
        if EVG_UB_STAGES >= 2:
            tla.set_flag(ub_load_ready_1)
            tla.set_flag(ub_store_done_1)

        for block_linear in block_range:
            # Map linear MN task id → (block_row, block_col) with swizzle (must stay in kernel AST).
            block_row = c0
            block_col = c0
            if SWIZZLE_DIRECTION == 0:
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
            per_core_tile_idx = (block_linear - tla.arch.block_idx()) // tla.arch.block_dim()
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
            half_m = (actual_m + 1) // 2
            gm_acc = tla.tile_view(
                gm_workspace_by_core, tla.make_shape(half_m, actual_n), tla.make_coord(aiv_sub_idx, c0)
            )
            gm_result = tla.tile_view(
                gm_d_by_core, tla.make_shape(half_m, actual_n), tla.make_coord(aiv_sub_idx, c0)
            )
            half_m = gm_result.origin_shape[0]
            half_n = gm_result.origin_shape[1]

            ub_stage_idx = c0
            if half_n <= ub_slot_elems:
                ub_row_stride = ((half_n + BYTE_PER_C0 - 1) // BYTE_PER_C0) * BYTE_PER_C0
                max_rows = ub_slot_elems // ub_row_stride
                if max_rows < 1:
                    max_rows = 1
                n_epilogue_row_tiles = (half_m + max_rows - 1) // max_rows

                for tile_i in tla.range(c0, n_epilogue_row_tiles, c1):
                    gm_c_tile = tla.tile_view(
                        gm_acc,
                        tla.make_shape(max_rows, half_n),
                        tla.make_coord(tile_i, c0),
                    )
                    gm_d_tile = tla.tile_view(
                        gm_result,
                        tla.make_shape(max_rows, half_n),
                        tla.make_coord(tile_i, c0),
                    )
                    tile_rows = gm_c_tile.origin_shape[0]
                    tile_cols = gm_c_tile.origin_shape[1]
                    ub_layout = tla.make_layout(
                        tla.make_shape(tile_rows, tile_cols),
                        tla.make_stride(ub_row_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    n_simd_strips = (tile_cols + simd_lanes - 1) // simd_lanes
                    use_ub_stage1 = EVG_UB_STAGES >= 2 and (ub_stage_idx != c0)

                    ub_acc = tla.make_tensor(
                        ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                    )
                    ub_result = tla.make_tensor(
                        ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                    )

                    # Wait UB slot free → load accumulator → signal compute may start.
                    if use_ub_stage1:
                        tla.wait_flag(ub_load_ready_1)
                    else:
                        tla.wait_flag(ub_load_ready_0)
                    tla.copy(ub_acc, gm_c_tile)
                    if use_ub_stage1:
                        tla.set_flag(ub_loaded_1)
                        tla.wait_flag(ub_loaded_1)
                        tla.wait_flag(ub_store_done_1)
                    else:
                        tla.set_flag(ub_loaded_0)
                        tla.wait_flag(ub_loaded_0)
                        tla.wait_flag(ub_store_done_0)

                    with tla.vec.func(mode="simd"):
                        for row_i in tla.range(c0, tile_rows, c1):
                            remaining = tile_cols
                            for col_j in tla.range(c0, n_simd_strips, c1):
                                c_chunk = tla.tile_view(
                                    ub_acc,
                                    tla.make_shape(REG_M, simd_lanes),
                                    tla.make_coord(row_i, col_j),
                                )
                                result_chunk = tla.tile_view(
                                    ub_result,
                                    tla.make_shape(REG_M, simd_lanes),
                                    tla.make_coord(row_i, col_j),
                                )
                                tail, remaining = tla.update_mask(
                                    remaining, dtype=DTYPE_GM_C
                                )
                                # Silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
                                xv = c_chunk.load()
                                den = 1.0 + tla.exp(
                                    tla.neg(xv, mask=tail), mask=tail
                                )
                                result_chunk.store(
                                    tla.div(xv, den, mask=tail), mask=tail
                                )

                    if use_ub_stage1:
                        tla.set_flag(ub_load_ready_1)
                        tla.set_flag(vec_done_1)
                        tla.wait_flag(vec_done_1)
                    else:
                        tla.set_flag(ub_load_ready_0)
                        tla.set_flag(vec_done_0)
                        tla.wait_flag(vec_done_0)
                    tla.copy(gm_d_tile, ub_result)
                    if use_ub_stage1:
                        tla.set_flag(ub_store_done_1)
                    else:
                        tla.set_flag(ub_store_done_0)

                    if EVG_UB_STAGES >= 2:
                        ub_stage_idx = c1 - ub_stage_idx
            else:
                n_col_tiles = (half_n + ub_slot_elems - 1) // ub_slot_elems
                for row_i in tla.range(c0, half_m, c1):
                    for col_tile in tla.range(c0, n_col_tiles, c1):
                        gm_c_tile = tla.tile_view(
                            gm_acc,
                            tla.make_shape(c1, ub_slot_elems),
                            tla.make_coord(row_i, col_tile),
                        )
                        gm_d_tile = tla.tile_view(
                            gm_result,
                            tla.make_shape(c1, ub_slot_elems),
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
                        n_simd_strips = (tile_cols + simd_lanes - 1) // simd_lanes
                        use_ub_stage1 = EVG_UB_STAGES >= 2 and (ub_stage_idx != c0)

                        ub_acc = tla.make_tensor(
                            ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                        )
                        ub_result = tla.make_tensor(
                            ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                        )

                        if use_ub_stage1:
                            tla.wait_flag(ub_load_ready_1)
                        else:
                            tla.wait_flag(ub_load_ready_0)
                        tla.copy(ub_acc, gm_c_tile)
                        if use_ub_stage1:
                            tla.set_flag(ub_loaded_1)
                            tla.wait_flag(ub_loaded_1)
                            tla.wait_flag(ub_store_done_1)
                        else:
                            tla.set_flag(ub_loaded_0)
                            tla.wait_flag(ub_loaded_0)
                            tla.wait_flag(ub_store_done_0)

                        with tla.vec.func(mode="simd"):
                            for r_i in tla.range(c0, tile_rows, c1):
                                remaining = tile_cols
                                for col_j in tla.range(c0, n_simd_strips, c1):
                                    c_chunk = tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(REG_M, simd_lanes),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    result_chunk = tla.tile_view(
                                        ub_result,
                                        tla.make_shape(REG_M, simd_lanes),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    tail, remaining = tla.update_mask(
                                        remaining, dtype=DTYPE_GM_C
                                    )
                                    # Silu(x) = x / (1 + exp(-x))
                                    xv = c_chunk.load()
                                    den = 1.0 + tla.exp(
                                        tla.neg(xv, mask=tail), mask=tail
                                    )
                                    result_chunk.store(
                                        tla.div(xv, den, mask=tail), mask=tail
                                    )

                        if use_ub_stage1:
                            tla.set_flag(ub_load_ready_1)
                            tla.set_flag(vec_done_1)
                            tla.wait_flag(vec_done_1)
                        else:
                            tla.set_flag(ub_load_ready_0)
                            tla.set_flag(vec_done_0)
                            tla.wait_flag(vec_done_0)
                        tla.copy(gm_d_tile, ub_result)
                        if use_ub_stage1:
                            tla.set_flag(ub_store_done_1)
                        else:
                            tla.set_flag(ub_store_done_0)

                    if EVG_UB_STAGES >= 2:
                        ub_stage_idx = c1 - ub_stage_idx

        tla.wait_flag(ub_load_ready_0)
        tla.wait_flag(ub_store_done_0)
        if EVG_UB_STAGES >= 2:
            tla.wait_flag(ub_load_ready_1)
            tla.wait_flag(ub_store_done_1)

        tla.pipe_barrier(tla.pipes.ALL)

# ---- host / CLI ----
DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"
# Fixed default for reproducible A/B fills (questboard acceptance uses the same).
DEFAULT_RNG_SEED = 20260727

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

def _parse_layout_choice(name: str) -> LayoutChoice:
    key = name.strip().lower().replace("_", "")
    mapping: dict[str, LayoutChoice] = {
        "row": "row",
        "rowmajor": "row",
        "col": "col",
        "columnmajor": "col",
        "colmajor": "col",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"unknown layout {name!r}; expected one of row, col"
        )
    return mapping[key]

def _gm_layout_tag(choice: LayoutChoice) -> Any:
    if choice == "row":
        return tla.arch.RowMajor
    return tla.arch.ColumnMajor

def _parse_elem_dtype(name: str) -> ElemDType:
    key = name.strip().lower().replace("_", "")
    mapping: dict[str, ElemDType] = {
        "f16": "f16",
        "float16": "f16",
        "fp16": "f16",
        "half": "f16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f32": "f32",
        "float32": "f32",
        "fp32": "f32",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"unknown dtype {name!r}; expected f16, bf16, or f32 "
            "(aliases e.g. float16, fp16, half / bfloat16 / float32, fp32)"
        )
    return mapping[key]

def _tla_elem_dtype(token: ElemDType) -> Any:
    if token == "f16":
        return tla.Float16
    if token == "bf16":
        return tla.BFloat16
    return tla.Float32

def _validate_mmad_dtype_triple(dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType) -> None:
    if dtype_a != dtype_b:
        raise ValueError(
            "dtype-a and dtype-b must match (tla.mmad requires lhs and rhs element types equal)."
        )
    # Supported: A==B in {f16,f32,bf16}, output D in {f16,f32}.
    # bf16 output is not supported (tla.neg/exp reject bf16).
    allowed = {
        ("f16", "f16", "f16"),
        ("f16", "f16", "f32"),
        ("bf16", "bf16", "f16"),
        ("bf16", "bf16", "f32"),
        ("f32", "f32", "f16"),
        ("f32", "f32", "f32"),
    }
    triple = (dtype_a, dtype_b, dtype_c)
    if triple not in allowed:
        raise ValueError(
            "unsupported (dtype-a, dtype-b, dtype-c); allowed: "
            "f16,f16,{f16|f32} | bf16,bf16,{f16|f32} | f32,f32,{f16|f32} "
            "(bf16 output is not supported)."
        )

def _apply_kernel_dtypes(dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType) -> None:
    global DTYPE_A, DTYPE_B, DTYPE_GM_C, DTYPE_C
    dtype_a_tla = _tla_elem_dtype(dtype_a)
    dtype_b_tla = _tla_elem_dtype(dtype_b)
    dtype_c_tla = _tla_elem_dtype(dtype_c)
    DTYPE_A = dtype_a_tla
    DTYPE_B = dtype_b_tla
    DTYPE_GM_C = dtype_c_tla
    DTYPE_C = tla.Float32

def _apply_problem_size(m_val: int, n_val: int, k_val: int) -> None:
    global m, n, k, SWIZZLE_DIRECTION
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise ValueError(f"m, n, k must be positive; got m={m_val}, n={n_val}, k={k_val}")
    # Zn when m>n, else Nz.
    SWIZZLE_DIRECTION = 0 if m_val > n_val else 1
    m = m_val
    n = n_val
    k = k_val

def _torch_dtype(token: ElemDType) -> Any:
    import torch

    if token == "f16":
        return torch.float16
    if token == "bf16":
        return torch.bfloat16
    return torch.float32

def _compile_only_type_args(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> tuple[Any, Any, Any]:
    """Metadata-only tensors for ``dump_mlir`` (no device buffer required)."""
    from catlass import runtime as runtime_mod

    ta = _tla_elem_dtype(dtype_a)
    tb = _tla_elem_dtype(dtype_b)
    tc = _tla_elem_dtype(dtype_c)
    with runtime_mod._eager_capture():
        return (
            tla.Tensor(
                tla.make_shape(m, k),
                ta,
                origin_shape=tla.make_shape(m, k),
                layout_tag=_gm_layout_tag(layout_a),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(k, n),
                tb,
                origin_shape=tla.make_shape(k, n),
                layout_tag=_gm_layout_tag(layout_b),
            ).mark_layout_dynamic(),
            tla.Tensor(
                tla.make_shape(m, n),
                tc,
                origin_shape=tla.make_shape(m, n),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
            # GM workspace for A×B (C); same element type as D (dtype-c).
            tla.Tensor(
                tla.make_shape(m, n),
                tc,
                origin_shape=tla.make_shape(m, n),
                layout_tag=tla.arch.RowMajor,
            ).mark_layout_dynamic(),
        )

def _device_buffer_for_layout(dense: Any, choice: LayoutChoice) -> Any:
    """Prepare NPU storage whose GM layout matches ``layout_tag`` (Torch only)."""
    if choice == "row":
        return dense.contiguous()
    return dense.permute(1, 0).contiguous()

def _create_tla_tensor(dev_buf: Any, layout: LayoutChoice) -> Any:
    """Wrap one device buffer as a ``tla.Tensor`` via DLPack (``layout_tag`` required)."""
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()

def dump_tlair(
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> str:
    return matmul_evg_silu_kernel.dump_mlir(
        type_args=_compile_only_type_args(
            layout_a, layout_b, dtype_a, dtype_b, dtype_c
        )
    )

def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }

def _comparison_rtol(k_val: int) -> float:
    """Relative tolerance: 1/256 if K<2048 else 1/128."""
    return (1.0 / 256.0) if int(k_val) < 2048 else (1.0 / 128.0)

def _compare_close_cpp(actual: Any, expected: Any, *, rtol: float) -> Any:
    """Elementwise close: |a-e| <= rtol * max(1, |e|)."""
    import torch
    diff = (actual - expected).abs()
    thresh = rtol * torch.maximum(torch.ones_like(expected), expected.abs())
    return diff <= thresh

def _first_mismatch_torch(
    actual: Any, expected: Any, *, rtol: float
) -> dict[str, Any] | None:
    close = _compare_close_cpp(actual, expected, rtol=rtol)
    if bool(close.all()):
        return None
    row, col = (int(value) for value in close.logical_not().nonzero(as_tuple=False)[0])
    act = float(actual[row, col].item())
    exp = float(expected[row, col].item())
    return {
        "index": [row, col],
        "actual": act,
        "expected": exp,
        "abs_err": abs(act - exp),
        "thresh": rtol * max(1.0, abs(exp)),
        "rtol": rtol,
    }

def _print_case_result(
    *,
    host: str,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
    artifact: Any,
    expected_match_all: bool,
    nonzero_count: int,
    first_mismatch: dict[str, Any] | None,
) -> None:
    print(
        "compile_ok=True "
        f"host={host} layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"D equals expected silu(A@B)? {expected_match_all}")
    print(f"D nonzero count={nonzero_count}")
    print(f"first mismatch={first_mismatch}")

def build_only(args: argparse.Namespace) -> int:
    _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
    artifact = tla.compile(
        matmul_evg_silu_kernel,
        *_compile_only_type_args(
            args.layout_a,
            args.layout_b,
            args.dtype_a,
            args.dtype_b,
            args.dtype_c,
        ),
        **_runtime_kwargs(args),
    )
    print("compile_ok=True")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0

def _run_one(
    args: argparse.Namespace,
    *,
    m_val: int,
    n_val: int,
    k_val: int,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
    evg_ub_stages: int = 2,
    label: str = "",
    quiet: bool = False,
) -> dict[str, Any]:
    """Compile/launch one case; return result dict for questboard generalization."""
    global EVG_UB_STAGES
    _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
    _apply_problem_size(m_val, n_val, k_val)
    _apply_kernel_dtypes(dtype_a, dtype_b, dtype_c)
    if evg_ub_stages < 1:
        raise ValueError(f"evg_ub_stages must be >= 1, got {evg_ub_stages}")
    EVG_UB_STAGES = int(evg_ub_stages)

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = "npu"
    torch_dtype_a = _torch_dtype(dtype_a)
    torch_dtype_b = _torch_dtype(dtype_b)
    torch_dtype_c = _torch_dtype(dtype_c)
    seed = int(getattr(args, "seed", DEFAULT_RNG_SEED))
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(seed)
    # Fill A/B with uniform[-5, 5].
    torch_tensor_a = torch.empty(
        (m, k), dtype=torch.float32, device=device
    ).uniform_(-5.0, 5.0).to(torch_dtype_a)
    torch_tensor_b = torch.empty(
        (k, n), dtype=torch.float32, device=device
    ).uniform_(-5.0, 5.0).to(torch_dtype_b)
    # mem_d starts empty; workspace holds A×B (same GM dtype as D).
    torch_tensor_d = torch.zeros((m, n), dtype=torch_dtype_c, device=device)
    torch_workspace = torch.zeros((m, n), dtype=torch_dtype_c, device=device)
    mm = torch_tensor_a.to(torch.float32) @ torch_tensor_b.to(torch.float32)
    expected_f32 = torch.nn.functional.silu(mm)
    if dtype_c in ("f16", "bf16"):
        expected = expected_f32.to(torch_dtype_c).to(torch.float32)
    else:
        expected = expected_f32
    # Compare device output against torch golden with K-dependent rtol.
    rtol = _comparison_rtol(k_val)

    tla_tensor_a = _create_tla_tensor(torch_tensor_a, layout_a)
    tla_tensor_b = _create_tla_tensor(torch_tensor_b, layout_b)
    tla_tensor_d = _create_tla_tensor(torch_tensor_d, "row")
    tla_workspace = _create_tla_tensor(torch_workspace, "row")

    artifact = tla.compile(
        matmul_evg_silu_kernel,
        tla_tensor_a,
        tla_tensor_b,
        tla_tensor_d,
        tla_workspace,
        **_runtime_kwargs(args),
    )
    block = max(1, args.block if args.block != -1 else tla.get_aicore_num(args.device))
    artifact(
        tla_tensor_a, tla_tensor_b, tla_tensor_d, tla_workspace, block=block
    )

    torch.npu.synchronize()
    actual = torch_tensor_d.to(torch.float32)
    expected_match = _compare_close_cpp(actual, expected, rtol=rtol)
    first_mismatch = _first_mismatch_torch(actual, expected, rtol=rtol)
    ok = first_mismatch is None
    nonzero = int((actual.abs() > rtol).sum().item())

    if not quiet:
        if label:
            print(f"--- {label} ---")
        _print_case_result(
            host="torch_npu",
            layout_a=layout_a,
            layout_b=layout_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=dtype_c,
            artifact=artifact,
            expected_match_all=bool(expected_match.all()),
            nonzero_count=nonzero,
            first_mismatch=first_mismatch,
        )
    return {
        "ok": ok,
        "first_mismatch": first_mismatch,
        "nonzero_count": nonzero,
        "evg_ub_stages": int(evg_ub_stages),
        "compute_length": evg_compute_length(evg_ub_stages),
        "kernel_o": str(artifact.kernel_binary_path),
        "block": block,
        "rtol": rtol,
    }

def run_single_case(
    args: argparse.Namespace,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> int:
    result = _run_one(
        args,
        m_val=m,
        n_val=n,
        k_val=k,
        layout_a=layout_a,
        layout_b=layout_b,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        evg_ub_stages=EVG_UB_STAGES,
        quiet=False,
    )
    return 0 if result.get("ok") else 1

MMAD_DTYPE_TRIPLES: tuple[tuple[ElemDType, ElemDType, ElemDType], ...] = (
    ("f16", "f16", "f16"),
    ("f16", "f16", "f32"),
    ("bf16", "bf16", "f16"),
    ("bf16", "bf16", "f32"),
    ("f32", "f32", "f16"),
    ("f32", "f32", "f32"),
)

def _layout_pairs(
    args: argparse.Namespace,
) -> list[tuple[LayoutChoice, LayoutChoice]]:
    if args.all_layouts:
        return [(la, lb) for la in ("row", "col") for lb in ("row", "col")]
    return [(args.layout_a, args.layout_b)]

def _dtype_triples(
    args: argparse.Namespace,
) -> list[tuple[ElemDType, ElemDType, ElemDType]]:
    if args.all_mmad_dtypes:
        return list(MMAD_DTYPE_TRIPLES)
    return [(args.dtype_a, args.dtype_b, args.dtype_c)]

def run(args: argparse.Namespace) -> int:
    tla.initialize(device=args.device)
    try:
        failed = 0
        for dtype_a, dtype_b, dtype_c in _dtype_triples(args):
            _validate_mmad_dtype_triple(dtype_a, dtype_b, dtype_c)
            for layout_a, layout_b in _layout_pairs(args):
                print(
                    "---",
                    "backend=torch_npu",
                    f"dtype_a={dtype_a}",
                    f"dtype_b={dtype_b}",
                    f"dtype_c={dtype_c}",
                    f"layout_a={layout_a}",
                    f"layout_b={layout_b}",
                    "---",
                )
                failed += run_single_case(
                    args, layout_a, layout_b, dtype_a, dtype_b, dtype_c
                )
        return 0 if failed == 0 else 1
    finally:
        tla.finalize()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile/run matmul EVG silu: D=silu(A@B) via L0C→GM "
            "workspace + AIV. Kernel takes 4 GM tensors (A, B, D, workspace). "
            "Layout/dtype via CLI; default prefers f32 row."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="Compile the example and exit after generating kernel.o.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Compile, launch, and compare the full output matrix. This is the default.",
    )
    parser.add_argument("--device", type=int, default=3, help="NPU device id.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help=f"RNG seed for A/B fills (default: {DEFAULT_RNG_SEED}).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=m,
        help=f"GEMM M dimension (default: {m}).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=n,
        help=f"GEMM N dimension (default: {n}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=k,
        help=f"GEMM K dimension (default: {k}).",
    )
    parser.add_argument("--block", type=int, default=-1, help="Launch block count.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Deprecated/ignored: compare uses rtol 1/256 or 1/128 by K.",
    )
    parser.add_argument(
        "--layout-a",
        type=_parse_layout_choice,
        default="row",
        help="GM layout for A (M×K): row or col.",
    )
    parser.add_argument(
        "--layout-b",
        type=_parse_layout_choice,
        default="row",
        help="GM layout for B (K×N): row or col.",
    )
    parser.add_argument(
        "--all-layouts",
        action="store_true",
        help="Run all four (layout-a, layout-b) combinations sequentially.",
    )
    parser.add_argument(
        "--dtype-a",
        type=_parse_elem_dtype,
        default="f32",
        help="GM element type for A (M×K); must equal --dtype-b for tla.mmad.",
    )
    parser.add_argument(
        "--dtype-b",
        type=_parse_elem_dtype,
        default="f32",
        help="GM element type for B (K×N); must equal --dtype-a.",
    )
    parser.add_argument(
        "--dtype-c",
        type=_parse_elem_dtype,
        default="f32",
        help="GM element type for D (M×N): f32, or narrowed f16 with f16/f16, bf16/bf16, or f32/f32 inputs.",
    )
    parser.add_argument(
        "--all-mmad-dtypes",
        action="store_true",
        help=(
            "Run all supported (dtype-a, dtype-b, dtype-c) triples sequentially "
            "(with the chosen layout pair or all layout pairs when --all-layouts is set)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for compile cache and generated kernel.o files.",
    )
    parser.add_argument(
        "--force-recompile",
        action="store_true",
        help="Ignore any existing compile cache entry.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable compile cache reuse.",
    )
    parser.add_argument(
        "--dump-tlair",
        action="store_true",
        help="Print TLA MLIR (tla dialect) and exit without compiling or launching.",
    )
    return parser

def main() -> int:
    args = _build_parser().parse_args()
    _apply_problem_size(args.m, args.n, args.k)
    if not args.all_mmad_dtypes:
        _validate_mmad_dtype_triple(args.dtype_a, args.dtype_b, args.dtype_c)
    if args.dump_tlair:
        if args.all_layouts or args.all_mmad_dtypes:
            raise SystemExit("--dump-tlair requires a single layout and dtype triple.")
        _apply_kernel_dtypes(args.dtype_a, args.dtype_b, args.dtype_c)
        print(
            dump_tlair(
                args.layout_a,
                args.layout_b,
                args.dtype_a,
                args.dtype_b,
                args.dtype_c,
            )
        )
        return 0
    if args.build_only:
        if args.all_layouts or args.all_mmad_dtypes:
            raise SystemExit("--build-only requires a single layout and dtype triple.")
        return build_only(args)
    return run(args)

if __name__ == "__main__":
    raise SystemExit(main())
