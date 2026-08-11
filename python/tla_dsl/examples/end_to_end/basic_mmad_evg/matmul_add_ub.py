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

# One SIMD register = 256B; lane count = 256 / sizeof(ElementC). REG_M=1.
REG_M = 1

# Host may rewrite these before compile.
DTYPE_A = tla.Float32
DTYPE_B = tla.Float32
# Matches C++ ElementC: L0C / UB Acc / GM C (X→D) / EVG Aux·Out all share one type.
DTYPE_C = tla.Float32
# Use unit_flag on the final FixPipe of each MN tile.
ENABLE_UNIT_FLAG = True

# UB nodes counted for sizing: load Aux(X) and compute (Acc already in UB from FixPipe).
EVG_UB_NODES = 2
# EVG UB multi-buffer depth (1 or 2 physical slots). Manual edit only, like l1_tn.
EVG_UB_STAGES = 2
# FixPipe C workspace: [0, L0C/2) in fp32 elements.
C_UB_ELEMS = L0C_SIZE // 2 // 4

# Max elements per UB epilogue slot: floor(budget/nodes/stages/elem_bytes) to BYTE_PER_C0.
_ELEM_C_BYTES = dtype_size_bytes(DTYPE_C.dtype)
_UB_SLOT_RAW = (UB_SIZE - L0C_SIZE // 2) // EVG_UB_NODES // EVG_UB_STAGES // _ELEM_C_BYTES
UB_SLOT_ELEMS = (_UB_SLOT_RAW // BYTE_PER_C0) * BYTE_PER_C0
# SIMD lanes for ElementC; must match ``tla.update_mask`` (256B / sizeof).
SIMD_LANES = 256 // _ELEM_C_BYTES

# MN tile swizzle: direction 0 = Zn (prefer when m>n), 1 = Nz (prefer when m<=n).
# Host rewrites SWIZZLE_DIRECTION before compile.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 1

@tla.kernel
def matmul_evg_add_ub_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
) -> None:
    """Cube GEMM FixPipe into UB; vector epilogue D = Acc + X in-place on mem_c."""
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
    # FIX↔CUBE token used only when unit_flag is off.
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)

    # ---- EVG/UB soft-flags ----
    #   MTE2_V  (ub_aux*_data_ready): MTE2 load done → Vector may compute
    #   V_MTE3  (ub_out*_data_ready): Vector compute done → MTE3 may store
    #   V_MTE2  (ub_aux*_available):  Vector frees X/aux slot for MTE2 load (cross-tile)
    #   MTE3_V  (ub_out*_available):  MTE3 store done → Vector may reuse D/out slot (cross-tile)
    # Soft-flags are primed on AIV entry and drained on AIV exit.
    ub_aux0_data_ready = tla.flag("ub_aux0_data_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_aux1_data_ready = tla.flag("ub_aux1_data_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_out0_data_ready = tla.flag("ub_out0_data_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_out1_data_ready = tla.flag("ub_out1_data_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_aux0_available = tla.flag("ub_aux0_available", tla.arch.VECTOR, tla.arch.MTE2)
    ub_aux1_available = tla.flag("ub_aux1_available", tla.arch.VECTOR, tla.arch.MTE2)
    ub_out0_available = tla.flag("ub_out0_available", tla.arch.MTE3, tla.arch.VECTOR)
    ub_out1_available = tla.flag("ub_out1_available", tla.arch.MTE3, tla.arch.VECTOR)

    # ---- cross-core flags ----
    # AIC waits/sets both AIV0 and AIV1; AIV side is guarded by aiv_id.
    aic_finish = tla.cross_flag("aic_finish", mode=4)
    aiv_finish = tla.cross_flag("aiv_finish", mode=4)

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
    # stages==1 must not allocate a second slot pair (UB_SLOT_ELEMS already uses full budget).
    # Pre-init stage1 aliases so AST allows the compile-time if (see ENABLE_UNIT_FLAG).
    ub_acc_ptr = tla.allocate(C_UB_ELEMS, DTYPE_C, tla.AddressSpace.ub, 256)
    ub_aux_ptr0 = tla.allocate(UB_SLOT_ELEMS, DTYPE_C, tla.AddressSpace.ub, 256)
    ub_out_ptr0 = tla.allocate(UB_SLOT_ELEMS, DTYPE_C, tla.AddressSpace.ub, 256)
    ub_aux_ptr1 = ub_aux_ptr0
    ub_out_ptr1 = ub_out_ptr0
    if tla.const_expr(EVG_UB_STAGES >= 2):
        ub_aux_ptr1 = tla.allocate(UB_SLOT_ELEMS, DTYPE_C, tla.AddressSpace.ub, 256)
        ub_out_ptr1 = tla.allocate(UB_SLOT_ELEMS, DTYPE_C, tla.AddressSpace.ub, 256)

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
            gm_d_by_core = tla.tile_view(
                mem_c, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk - 1) // l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_d_by_core)

            # Wait both AIVs to free UB before overwriting (or priming on first tile).
            tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=0)
            tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=1)

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

            # FixPipe L0C→UB (SPLIT_M).
            # Use *actual* tile MN from GM view. A fixed (l1_tm, l1_tn) dst makes
            # FixPipe drain a full tile on N/M tails (e.g. n=129 → width-1 tile)
            # and hangs. Pitch stays l1_tn so the UB buffer layout is stable.
            actual_m = gm_d_by_core.origin_shape[0]
            actual_n = gm_d_by_core.origin_shape[1]
            ub_c = tla.make_tensor(
                ub_acc_ptr,
                tla.make_layout(
                    tla.make_shape(actual_m, actual_n),
                    tla.make_stride(l1_tn, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.set_flag(l0c_data_ready)
                tla.wait_flag(l0c_data_ready)
                tla.copy(
                    ub_c,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                tla.set_flag(l0c_available)
            else:
                tla.copy(
                    ub_c,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        unit_flag=0b11,
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                    ),
                )

            tla.cross_core_set_flag(aic_finish, tla.arch.FIX, aiv_id=0)
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX, aiv_id=1)

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)
        # Drain last AIV0/AIV1 tiles before cube exit.
        tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=1)
        tla.pipe_barrier(tla.pipes.ALL)

    # ---- vector: prime flags; MN loop; load / compute / store epilogue ----
    with tla.vector():
        aiv_sub_idx = tla.arch.sub_block_idx()
        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num())

        # Prime per-slot load-ready / store-done flags before the MN loop.
        tla.set_flag(ub_aux0_available)
        tla.set_flag(ub_out0_available)
        if tla.const_expr(EVG_UB_STAGES >= 2):
            tla.set_flag(ub_aux1_available)
            tla.set_flag(ub_out1_available)

        tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=1)
        for block_linear in block_range:
            # Same MN swizzle as cube path (AIC/AIV must visit identical tiles).
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

            tla.cross_core_wait_flag(aic_finish, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_wait_flag(aic_finish, tla.arch.VECTOR, aiv_id=1)

            # In-place GM C: load addend X from mem_c, store D=A@B+X back to mem_c
            # mem_c is both the addend source and the final output destination.
            gm_c_by_core = tla.tile_view(
                mem_c, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
            )
            actual_m = gm_c_by_core.origin_shape[0]
            actual_n = gm_c_by_core.origin_shape[1]
            # SPLIT_M: request ceil(M/2) rows × full N; AIV1 may clip when M is odd.
            aiv_m_req = (actual_m + 1) // 2
            gm_result = tla.tile_view(
                gm_c_by_core, tla.make_shape(aiv_m_req, actual_n), tla.make_coord(aiv_sub_idx, c0)
            )
            gm_addend = tla.tile_view(
                gm_c_by_core, tla.make_shape(aiv_m_req, actual_n), tla.make_coord(aiv_sub_idx, c0)
            )
            aiv_m = gm_result.origin_shape[0]
            aiv_n = gm_result.origin_shape[1]

            # Pitch l1_tn matches FixPipe UB layout; do not inherit GM stride (=N).
            ub_c_full = tla.make_tensor(
                ub_acc_ptr,
                tla.make_layout(
                    tla.make_shape(aiv_m, aiv_n),
                    tla.make_stride(l1_tn, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )

            # If tile width fits one UB slot: process multi-row tiles.
            # Otherwise: process one row at a time in column strips of UB_SLOT_ELEMS.
            # Flip the UB stage once per outer unit (row-tile or full row).
            ub_stage_idx = c0
            if aiv_n <= UB_SLOT_ELEMS:
                # --- row-tile path ---
                ub_row_stride = ((aiv_n + BYTE_PER_C0 - 1) // BYTE_PER_C0) * BYTE_PER_C0
                max_rows = UB_SLOT_ELEMS // ub_row_stride
                if max_rows < 1:
                    max_rows = 1
                n_epilogue_row_tiles = (aiv_m + max_rows - 1) // max_rows

                for tile_i in tla.range(c0, n_epilogue_row_tiles, c1):
                    gm_x_tile = tla.tile_view(
                        gm_addend,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    gm_d_tile = tla.tile_view(
                        gm_result,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    ub_c_tile = tla.tile_view(
                        ub_c_full,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    tile_rows = gm_x_tile.origin_shape[0]
                    tile_cols = gm_x_tile.origin_shape[1]
                    # X/D slot pitch: RoundUp(cols); C keeps FixPipe pitch l1_tn.
                    ub_xd_layout = tla.make_layout(
                        tla.make_shape(tile_rows, tile_cols),
                        tla.make_stride(ub_row_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        use_ub_stage1 = ub_stage_idx != c0
                    else:
                        use_ub_stage1 = False
                    ub_addend = tla.make_tensor(
                        ub_aux_ptr1 if use_ub_stage1 else ub_aux_ptr0, ub_xd_layout
                    )
                    ub_result = tla.make_tensor(
                        ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_xd_layout
                    )

                    # run_tile: Wait V_MTE2 → LOAD → Set MTE2_V
                    if use_ub_stage1:
                        tla.wait_flag(ub_aux1_available)
                    else:
                        tla.wait_flag(ub_aux0_available)
                    tla.copy(ub_addend, gm_x_tile)
                    if use_ub_stage1:
                        tla.set_flag(ub_aux1_data_ready)
                        tla.wait_flag(ub_aux1_data_ready)
                        tla.wait_flag(ub_out1_available)
                    else:
                        tla.set_flag(ub_aux0_data_ready)
                        tla.wait_flag(ub_aux0_data_ready)
                        tla.wait_flag(ub_out0_available)

                    # COMPUTE → Set V_MTE2 + V_MTE3
                    with tla.vec.func(mode="simd"):
                        for row_i in tla.range(c0, tile_rows, c1):
                            remaining = tile_cols
                            for col_j in tla.range(c0, n_simd_strips, c1):
                                c_chunk = tla.tile_view(
                                    ub_c_tile,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                addend_chunk = tla.tile_view(
                                    ub_addend,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                result_chunk = tla.tile_view(
                                    ub_result,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                tail, remaining = tla.update_mask(
                                    remaining, dtype=DTYPE_C
                                )
                                result_chunk.store(
                                    c_chunk.load() + addend_chunk.load(), mask=tail
                                )

                    if use_ub_stage1:
                        tla.set_flag(ub_aux1_available)
                        tla.set_flag(ub_out1_data_ready)
                        tla.wait_flag(ub_out1_data_ready)
                    else:
                        tla.set_flag(ub_aux0_available)
                        tla.set_flag(ub_out0_data_ready)
                        tla.wait_flag(ub_out0_data_ready)
                    # STORE → Set MTE3_V
                    tla.copy(gm_d_tile, ub_result)
                    if use_ub_stage1:
                        tla.set_flag(ub_out1_available)
                    else:
                        tla.set_flag(ub_out0_available)

                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        ub_stage_idx = c1 - ub_stage_idx
            else:
                # --- column-strip path: 1 row × UB_SLOT_ELEMS ---
                n_col_tiles = (aiv_n + UB_SLOT_ELEMS - 1) // UB_SLOT_ELEMS
                for row_i in tla.range(c0, aiv_m, c1):
                    for col_tile in tla.range(c0, n_col_tiles, c1):
                        gm_x_tile = tla.tile_view(
                            gm_addend,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        gm_d_tile = tla.tile_view(
                            gm_result,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        ub_c_tile = tla.tile_view(
                            ub_c_full,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        tile_rows = gm_x_tile.origin_shape[0]
                        tile_cols = gm_x_tile.origin_shape[1]
                        ub_row_stride = (
                            (tile_cols + BYTE_PER_C0 - 1) // BYTE_PER_C0
                        ) * BYTE_PER_C0
                        ub_xd_layout = tla.make_layout(
                            tla.make_shape(tile_rows, tile_cols),
                            tla.make_stride(ub_row_stride, 1),
                            layoutTag=tla.arch.RowMajor,
                        )
                        n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                        if tla.const_expr(EVG_UB_STAGES >= 2):
                            use_ub_stage1 = ub_stage_idx != c0
                        else:
                            use_ub_stage1 = False
                        ub_addend = tla.make_tensor(
                            ub_aux_ptr1 if use_ub_stage1 else ub_aux_ptr0, ub_xd_layout
                        )
                        ub_result = tla.make_tensor(
                            ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_xd_layout
                        )

                        if use_ub_stage1:
                            tla.wait_flag(ub_aux1_available)
                        else:
                            tla.wait_flag(ub_aux0_available)
                        tla.copy(ub_addend, gm_x_tile)
                        if use_ub_stage1:
                            tla.set_flag(ub_aux1_data_ready)
                            tla.wait_flag(ub_aux1_data_ready)
                            tla.wait_flag(ub_out1_available)
                        else:
                            tla.set_flag(ub_aux0_data_ready)
                            tla.wait_flag(ub_aux0_data_ready)
                            tla.wait_flag(ub_out0_available)

                        with tla.vec.func(mode="simd"):
                            for r_i in tla.range(c0, tile_rows, c1):
                                remaining = tile_cols
                                for col_j in tla.range(c0, n_simd_strips, c1):
                                    c_chunk = tla.tile_view(
                                        ub_c_tile,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    addend_chunk = tla.tile_view(
                                        ub_addend,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    result_chunk = tla.tile_view(
                                        ub_result,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    tail, remaining = tla.update_mask(
                                        remaining, dtype=DTYPE_C
                                    )
                                    result_chunk.store(
                                        c_chunk.load() + addend_chunk.load(),
                                        mask=tail,
                                    )

                        if use_ub_stage1:
                            tla.set_flag(ub_aux1_available)
                            tla.set_flag(ub_out1_data_ready)
                            tla.wait_flag(ub_out1_data_ready)
                        else:
                            tla.set_flag(ub_aux0_available)
                            tla.set_flag(ub_out0_data_ready)
                            tla.wait_flag(ub_out0_data_ready)
                        tla.copy(gm_d_tile, ub_result)
                        if use_ub_stage1:
                            tla.set_flag(ub_out1_available)
                        else:
                            tla.set_flag(ub_out0_available)

                    # Flip UB stage once per outer row, not per column strip.
                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        ub_stage_idx = c1 - ub_stage_idx

            tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=1)

        # Drain remaining load-ready / store-done credits before AIV exit.
        tla.wait_flag(ub_aux0_available)
        tla.wait_flag(ub_out0_available)
        if tla.const_expr(EVG_UB_STAGES >= 2):
            tla.wait_flag(ub_aux1_available)
            tla.wait_flag(ub_out1_available)

        tla.pipe_barrier(tla.pipes.ALL)

# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"
DESCRIPTION = "Matmul EVG add_ub: D=A@B+X via L0C→UB; dynamic GM."


def golden(a, b, out_dtype):
    import torch

    expected = a.to(torch.float32) @ b.to(torch.float32)
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
    mod.DTYPE_C = tla_of[dc]
    mod.SWIZZLE_DIRECTION = 0 if mi > ni else 1

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
        # mem_c starts as addend X, overwritten with D = A@B+X.
        c = torch.full((mi, ni), args.sentinel, dtype=torch_of[dc], device="npu")
        expected = golden(a, b, torch.float32) + c.to(torch.float32)
        if dc in ("f16", "bf16"):
            expected = expected.to(torch_of[dc]).to(torch.float32)

        ta, tb, tc = create_tla_tensor(a, la), create_tla_tensor(b, lb), create_tla_tensor(c, "row")
        artifact = tla.compile(
            matmul_evg_add_ub_kernel,
            ta,
            tb,
            tc,
            arch_scope="aic.c310",
            cache=not args.no_cache,
            cache_dir=cache_dir,
            force_recompile=args.force_recompile,
        )
        block_dim = max(
            1,
            args.block_dim if args.block_dim != -1 else tla.get_aicore_num(args.device),
        )
        artifact(ta, tb, tc, block_dim=block_dim)
        torch.npu.synchronize()

        # f16/f32 path (GM C is f32 on add_ub P0): rtol 1/256 or 1/128, floor 1.
        if dc == "bf16":
            rtol = (1.0 / 128.0) if ki < 2048 else (1.0 / 64.0)
            floor = 1.0 / 256.0
        else:
            rtol = (1.0 / 256.0) if ki < 2048 else (1.0 / 128.0)
            floor = 1.0
        got = c.detach().to(device="cpu", dtype=torch.float32)
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
    p.add_argument("--dtype-c", choices=("f32",), default="f32")
    p.add_argument("--block-dim", type=int, default=-1)
    p.add_argument("--sentinel", type=float, default=-7.0)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--force-recompile", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
