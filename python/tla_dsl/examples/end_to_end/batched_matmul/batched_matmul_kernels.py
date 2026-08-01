"""Batched matmul — Catlass example 01 (``C[b] = A[b] @ B[b]``).

Device kernel：在 ``batch_count * grid_m * grid_n`` 个工作项上做 grid-stride，
与 C++ ``BatchedMatmul`` 一致。Host 将 A/B/C 展成 2D：
``(B*M, K)`` / ``(B*K, N)`` / ``(B*M, N)``，再用 ``tile_view`` 取 batch。

Dynamic GM（schema v4）：Host ``mark_layout_dynamic()``；kernel 从
``mem_*.origin_shape`` 推导 ``batch_count / m / n / k``。
"""

from __future__ import annotations

import catlass as tla

# Host（batched_matmul.py）在 ``tla.compile`` 前改写这些全局量。
batch_count = 5
m = 256
n = 512
k = 1024

l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32

# Match C++ GemmIdentityBlockSwizzle. Host may override OFFSET/DIRECTION.
# Default OFFSET=3 (ex01/ex67). When grid_* % OFFSET != 0, tail uses select.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 1

DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float16
# Unit-flag fuses L0C→GM with the K-loop (matches C++ enableUnitFlag=true).
# Soft-flag path remains for hosts that set this False (e.g. multi-layout
# in-process sweeps where residual cube/FIX state can break a later launch).
ENABLE_UNIT_FLAG = True


@tla.kernel
def batched_matmul_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
) -> None:
    """Cube batched GEMM：对每个 batch 做 C = A @ B（L1/L0 双缓冲 + flag）。

    mem_a/b/c 为展平 2D（schema v4 dynamic GM；工作尺寸取自 origin_shape）：
      A: (batch_count * m, k)
      B: (batch_count * k, n)
      C: (batch_count * m, n)
    """
    c0 = 0
    c1 = 1

    # Dynamic GM: extents from launch memref, not host module globals.
    k = mem_a.origin_shape[1]
    n = mem_c.origin_shape[1]
    batch_count = mem_b.origin_shape[0] // k
    m = mem_a.origin_shape[0] // batch_count

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

    l1a0_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(l0_tm * l0_tn, DTYPE_C, tla.AddressSpace.l0c, 512)

    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    mn_blocks = grid_m * grid_n
    # 与 C++ 一致：coreLoops = batchCount * GetCoreLoops()
    total_blocks = batch_count * mn_blocks
    # Tail band width (equals SWIZZLE_OFFSET when grid divides evenly).
    last_n_row = grid_m - SWIZZLE_OFFSET * ((grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1)
    last_n_col = grid_n - SWIZZLE_OFFSET * ((grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1)

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

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_dim()
        )
        for loop_idx in block_range:
            batch_idx = loop_idx // mn_blocks
            mn_linear = loop_idx % mn_blocks

            # GemmIdentityBlockSwizzle<SWIZZLE_OFFSET, SWIZZLE_DIRECTION>
            # DIRECTION stays host/compile-time (Zn when M>N); band tails are dynamic.
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                # Zn: bands along M, serpentine N
                tile_block_idx = mn_linear // (SWIZZLE_OFFSET * grid_n)
                in_tile = mn_linear % (SWIZZLE_OFFSET * grid_n)
                tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                n_row = (
                    last_n_row
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_row
                block_col = in_tile // n_row
                # Serpentine: branchless (odd band flips N)
                odd = tile_block_idx % 2
                block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
            else:
                # Nz: bands along N, serpentine M (C++ when M<=N)
                tile_block_idx = mn_linear // (SWIZZLE_OFFSET * grid_m)
                in_tile = mn_linear % (SWIZZLE_OFFSET * grid_m)
                tile_block_loop = (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            # 展平 2D 上按 batch 切出单盘 (m,k)/(k,n)/(m,n)
            gm_a_batch = tla.tile_view(
                mem_a, tla.make_shape(m, k), tla.make_coord(batch_idx, c0)
            )
            gm_b_batch = tla.tile_view(
                mem_b, tla.make_shape(k, n), tla.make_coord(batch_idx, c0)
            )
            gm_c_batch = tla.tile_view(
                mem_c, tla.make_shape(m, n), tla.make_coord(batch_idx, c0)
            )

            gm_a_by_core = tla.tile_view(
                gm_a_batch, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b_batch, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c_batch,
                tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk - 1) // l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

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

            if not ENABLE_UNIT_FLAG:
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)
                tla.copy(gm_c_by_core, l0_c)
                tla.set_flag(fix_done)
            else:
                tla.copy(
                    gm_c_by_core, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11)
                )

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)
        tla.wait_flag(fix_done)
