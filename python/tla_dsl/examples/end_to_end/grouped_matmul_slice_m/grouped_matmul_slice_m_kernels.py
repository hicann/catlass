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

import catlass as tla

# Host may mutate before ``tla.compile``.
M_DIM = 1024
N_DIM = 256
K_DIM = 256
GROUPS = 4

L1_TM = 256
L1_TN = 256
L1_TK = 256
L0_TM = 256
L0_TN = 256
L0_TK = 64

DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float16
ENABLE_UNIT_FLAG = True


@tla.kernel
def grouped_matmul_slice_m_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    group_list: tla.Tensor,
    mem_c: tla.Tensor,
) -> None:
    c0 = 0
    c1 = 1
    # Runtime extents from dynamic GM (host marks mark_layout_dynamic /
    # mark_compact_shape_dynamic). Module-level M_DIM/N_DIM/K_DIM/GROUPS are
    # only for host tensor allocation and compile-time type_args.
    n = mem_c.origin_shape[1]
    k = mem_a.origin_shape[1]
    problem_count = group_list.origin_shape[0] - 1

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

    l1a0_ptr = tla.allocate(L1_TM * L1_TK, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(L1_TM * L1_TK, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(L1_TK * L1_TN, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(L1_TK * L1_TN, DTYPE_B, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(L0_TM * L0_TK, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(L0_TM * L0_TK, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(L0_TK * L0_TN, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(L0_TK * L0_TN, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0_TM * L0_TN, DTYPE_C, tla.AddressSpace.l0c, 512)

    grid_n = (n + L1_TN - 1) // L1_TN

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
        # Carry core offset across groups (C++ startCoreIdx) for load balance.
        start_core_idx = c0
        core_num = tla.arch.block_dim()
        core_idx = tla.arch.block_idx()

        for g in tla.range(c0, problem_count, c1):
            # 1D GM Int32 prefix: currentM = end - start (no Python list).
            m_start = group_list[g]
            m_end = group_list[g + 1]
            current_m = m_end - m_start

            if current_m > 0:
                # L1-aligned groups: tile index = element offset / L1_TM.
                m_tile_base = m_start // L1_TM
                grid_m = current_m // L1_TM
                mn_blocks = grid_m * grid_n

                gm_b_group = tla.tile_view(
                    mem_b, tla.make_shape(k, n), tla.make_coord(g, c0)
                )

                # Rotate which core owns loop 0 of this group.
                # (core_idx - start_core_idx) mod core_num — avoid dynamic if new defs.
                start_loop = (core_idx + core_num - start_core_idx) % core_num
                block_range = tla.range(start_loop, mn_blocks, core_num)
                for loop_idx in block_range:
                    block_row = loop_idx // grid_n
                    block_col = loop_idx % grid_n
                    abs_row = m_tile_base + block_row

                    gm_a_by_core = tla.tile_view(
                        mem_a, tla.make_shape(L1_TM, k), tla.make_coord(abs_row, c0)
                    )
                    gm_b_by_core = tla.tile_view(
                        gm_b_group, tla.make_shape(k, L1_TN), tla.make_coord(c0, block_col)
                    )
                    gm_c_by_core = tla.tile_view(
                        mem_c,
                        tla.make_shape(L1_TM, L1_TN),
                        tla.make_coord(abs_row, block_col),
                    )

                    k_block = gm_a_by_core.origin_shape[1]
                    k_l1_count = (k_block + L1_TK - 1) // L1_TK
                    k_l1_range = tla.range(c0, k_l1_count, c1)

                    l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

                    if not ENABLE_UNIT_FLAG:
                        tla.wait_flag(fix_done)
                    for k_l1 in k_l1_range:
                        gm_a_l1 = tla.tile_view(
                            gm_a_by_core,
                            tla.make_shape(L1_TM, L1_TK),
                            tla.make_coord(c0, k_l1),
                        )
                        gm_b_l1 = tla.tile_view(
                            gm_b_by_core,
                            tla.make_shape(L1_TK, L1_TN),
                            tla.make_coord(k_l1, c0),
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

                        k_l0_count = (l1_a.origin_shape[1] + L0_TK - 1) // L0_TK
                        k_l0_range = tla.range(c0, k_l0_count, c1)
                        for k_l0 in k_l0_range:
                            l1_a_l0 = tla.tile_view(
                                l1_a,
                                tla.make_shape(L0_TM, L0_TK),
                                tla.make_coord(c0, k_l0),
                            )
                            l1_b_l0 = tla.tile_view(
                                l1_b,
                                tla.make_shape(L0_TK, L0_TN),
                                tla.make_coord(k_l0, c0),
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
                                if (k_l1 == k_l1_count - 1) and (
                                    k_l0 == k_l0_count - 1
                                ):
                                    unit_flag = 0b11
                                else:
                                    unit_flag = 0b10
                            init_c = True if k_l1 == 0 and k_l0 == 0 else False
                            tla.mmad(
                                l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag
                            )
                            if l0_buf_idx == c0:
                                tla.set_flag(l0a0_copy_start)
                                tla.set_flag(l0b0_copy_start)
                            else:
                                tla.set_flag(l0a1_copy_start)
                                tla.set_flag(l0b1_copy_start)
                            l0_buf_idx = c1 - l0_buf_idx
                        l1_buf_idx = c1 - l1_buf_idx

                    if not ENABLE_UNIT_FLAG:
                        tla.set_flag(mmad_done)
                        tla.wait_flag(mmad_done)
                        tla.copy(gm_c_by_core, l0_c)
                        tla.set_flag(fix_done)
                    else:
                        tla.copy(
                            gm_c_by_core,
                            l0_c,
                            tla.params.CopyL0C2DstParams(unit_flag=0b11),
                        )

                start_core_idx = (start_core_idx + mn_blocks) % core_num

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)
        tla.wait_flag(fix_done)
