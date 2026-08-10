"""Tail multi-core split-K matmul: Kernel + Host in one file.

Normal M×N tiles: full-K → GM C. Tail tiles: split-K → workspace + AIV ReduceAdd.
CLI aligned with basic_matmul.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass as tla
from catlass.params import NormalStoreParams, StoreDist
from catlass.runtime import from_dlpack

ENABLE_UNIT_FLAG = True

l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32

# Host rewrites before compile: 0=Zn when m>n, 1=Nz when m<=n.
SWIZZLE_OFFSET = 3
SWIZZLE_DIRECTION = 1

SUB_BLOCK_NUM = 2
ELE_PER_VECTOR_BLOCK = 64
ELE_NUM_ALIGN = 8
COMPUTE_LENGTH = 192 * 1024 // 4

DTYPE_A = tla.Float32
DTYPE_B = tla.Float32
DTYPE_C = tla.Float32
DTYPE_W = tla.Float32
DTYPE_GM_C = tla.Float32

# Host injects these before compile (captured like DTYPE_*).
NEED_CAST = False
CAST_FLOOR = False
aic_core_num = 28
normal_block_num = 0
tail_block_num = 0
splitk_factor = 1
core_loops = 1
tile_per_core = 1
ub_row_stride = 64
reduce_vl_loops = 1
chunk_elems = 64

DESCRIPTION = "Tail multi-core split-K matmul; dynamic GM."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def tail_multi_core_splitk_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_w: tla.Tensor,
) -> None:
    """Normal tiles write GM C; tail tiles split-K via workspace and AIV ReduceAdd."""
    c0 = 0
    c1 = 1
    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    # AIC pipeline flags: MTE2 ↔ MTE1 ↔ Cube ↔ FIX
    l1a0_data_ready = tla.flag(
        "l1a0_copy_end", tla.arch.MTE2, tla.arch.MTE1
    )
    l1a1_data_ready = tla.flag(
        "l1a1_copy_end", tla.arch.MTE2, tla.arch.MTE1
    )
    l1b0_data_ready = tla.flag(
        "l1b0_copy_end", tla.arch.MTE2, tla.arch.MTE1
    )
    l1b1_data_ready = tla.flag(
        "l1b1_copy_end", tla.arch.MTE2, tla.arch.MTE1
    )
    l1a0_available = tla.flag(
        "l1a0_copy_start", tla.arch.MTE1, tla.arch.MTE2
    )
    l1a1_available = tla.flag(
        "l1a1_copy_start", tla.arch.MTE1, tla.arch.MTE2
    )
    l1b0_available = tla.flag(
        "l1b0_copy_start", tla.arch.MTE1, tla.arch.MTE2
    )
    l1b1_available = tla.flag(
        "l1b1_copy_start", tla.arch.MTE1, tla.arch.MTE2
    )
    l0a0_available = tla.flag(
        "l0a0_copy_start", tla.arch.CUBE, tla.arch.MTE1
    )
    l0a1_available = tla.flag(
        "l0a1_copy_start", tla.arch.CUBE, tla.arch.MTE1
    )
    l0b0_available = tla.flag(
        "l0b0_copy_start", tla.arch.CUBE, tla.arch.MTE1
    )
    l0b1_available = tla.flag(
        "l0b1_copy_start", tla.arch.CUBE, tla.arch.MTE1
    )
    l0_ab_data_ready = tla.flag("l0_copy_end", tla.arch.MTE1, tla.arch.CUBE)
    l0c_data_ready = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l0c_available = tla.flag("fix_done", tla.arch.FIX, tla.arch.CUBE)

    # Cross-core: paired AIC→AIV done (mode 2); AIV all-block barrier (mode 0).
    cross_aic_to_aiv_done = tla.cross_flag("aic_finish", mode=2)
    cross_aiv_barrier = tla.cross_flag("aiv_ibarrier", mode=0)

    # AIV reduce pipeline: MTE3 → MTE2 → vector → MTE3
    reduce_mte3_mte2 = tla.flag(
        "reduce_mte3_mte2", tla.arch.MTE3, tla.arch.MTE2
    )
    reduce_mte2_v = tla.flag(
        "reduce_mte2_v", tla.arch.MTE2, tla.arch.VECTOR
    )
    reduce_v_mte3 = tla.flag(
        "reduce_v_mte3", tla.arch.VECTOR, tla.arch.MTE3
    )

    # L1 ping-pong tiles for A (M×K) and B (K×N)
    l1a0_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    # L0 ping-pong tiles inside Cube
    l0a0_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(l0_tm * l0_tn, DTYPE_C, tla.AddressSpace.l0c, 512)

    # UB: split-K gather rows for tail-tile reduce.
    ub_reduce_ptr = tla.allocate(COMPUTE_LENGTH, DTYPE_W, tla.AddressSpace.ub, 256)
    if tla.const_expr(NEED_CAST):
        ub_cast_ptr = tla.allocate(chunk_elems, DTYPE_GM_C, tla.AddressSpace.ub, 256)

    # Narrowing cast to GM C: f16 uses floor rounding, bf16 uses round-to-nearest.
    if tla.const_expr(NEED_CAST):
        if tla.const_expr(CAST_FLOOR):
            cast_to_gm_params = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_FLOOR,
            )
        else:
            cast_to_gm_params = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )

    # Tile grid over M×N and K slices per AIC
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    k_tile_num = (k + l1_tk - 1) // l1_tk
    # Tail band width (equals SWIZZLE_OFFSET when grid divides evenly).
    last_n_row = grid_m - SWIZZLE_OFFSET * (
        (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1
    )
    last_n_col = grid_n - SWIZZLE_OFFSET * (
        (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1
    )

    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)

        # Initial flag state so the first loop wait does not block.
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
        bid = tla.arch.block_idx()
        bdim = tla.arch.block_num()
        tail_cores = core_loops - normal_block_num

        # Normal-only AICs signal completion once before the task loop.
        if bid >= tail_cores:
            tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)

        task_range = tla.range(bid, core_loops, bdim)
        for loop_idx in task_range:
            # Remap loop index: normal tiles first, then tail tiles per split-K slice.
            actual = loop_idx
            if normal_block_num > 0:
                if (
                    loop_idx == normal_block_num - bdim + bid
                    and bid < tail_cores
                ):
                    actual = normal_block_num + bid
                elif loop_idx >= normal_block_num:
                    actual = normal_block_num - bdim + bid

            inner = actual % core_loops
            is_tail = 1 if inner >= normal_block_num else 0

            # Uneven K split for tail tiles: first (k % factor) slices get one extra tile.
            base_block = inner
            k_start = 0
            slice_tiles = k_tile_num
            rem = k_tile_num % splitk_factor
            quot = k_tile_num // splitk_factor
            if is_tail == 1:
                base_block = normal_block_num + (inner - normal_block_num) // splitk_factor
                slice_in_group = (inner - normal_block_num) % splitk_factor
                k_start = slice_in_group * quot + rem
                slice_tiles = quot
                if slice_in_group < rem:
                    k_start = (quot + 1) * slice_in_group
                    slice_tiles = quot + 1

            # Map linear tile index to (block_row, block_col) via Zn/Nz swizzle.
            # SWIZZLE_DIRECTION is host-set before compile (0=Zn when m>n, else Nz).
            # Body uses IfExp + branchless serpentine (same as batched_matmul) so
            # tla.const_expr outer fold does not nest dynamic statement-ifs.
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                # Zn: bands along M, serpentine N
                tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                tile_block_idx = base_block // (SWIZZLE_OFFSET * grid_n)
                in_tile = base_block % (SWIZZLE_OFFSET * grid_n)
                n_row = (
                    last_n_row
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_row
                block_col = in_tile // n_row
                odd = tile_block_idx % 2
                block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
            else:
                # Nz: bands along N, serpentine M
                tile_block_loop = (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                tile_block_idx = base_block // (SWIZZLE_OFFSET * grid_m)
                in_tile = base_block % (SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            # GM views for this M×N tile.
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
            )
            # Per-AIC workspace slot, cropped to actual tile shape.
            gm_w_by_core = tla.tile_view(
                gm_w, tla.make_shape(l1_tm, l1_tn), tla.make_coord(bid, c0)
            )
            gm_w_by_tile = tla.tile_view(
                gm_w_by_core,
                tla.make_shape(
                    gm_c_by_core.origin_shape[0], gm_c_by_core.origin_shape[1]
                ),
                tla.make_coord(c0, c0),
            )

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.wait_flag(l0c_available)

            # L1 K loop: copy K tiles from GM into ping-pong L1 buffers.
            k_l1_range = tla.range(c0, slice_tiles, c1)
            for k_local in k_l1_range:
                k_l1 = k_start + k_local
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core, tla.make_shape(l1_tm, l1_tk),
                    tla.make_coord(c0, k_l1)
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core, tla.make_shape(l1_tk, l1_tn),
                    tla.make_coord(k_l1, c0)
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                )

                # L1 A: wait for free buffer, GM→L1, release load-done flag.
                if l1_buf_idx == c0:
                    tla.wait_flag(l1a0_available)
                else:
                    tla.wait_flag(l1a1_available)
                tla.copy(l1_a, gm_a_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_data_ready)
                else:
                    tla.set_flag(l1a1_data_ready)

                # L1 B: same handshake as A.
                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_available)
                else:
                    tla.wait_flag(l1b1_available)
                tla.copy(l1_b, gm_b_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_data_ready)
                else:
                    tla.set_flag(l1b1_data_ready)

                # L0 K loop: L1→L0 ping-pong, MMAD into L0C.
                k_l0_count = (l1_a.origin_shape[1] + l0_tk - 1) // l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)
                for k_l0 in k_l0_range:
                    l1_a_by_l0 = tla.tile_view(
                        l1_a, tla.make_shape(l0_tm, l0_tk),
                        tla.make_coord(c0, k_l0)
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b, tla.make_shape(l0_tk, l0_tn),
                        tla.make_coord(k_l0, c0)
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
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
                    tla.copy(l0_a, l1_a_by_l0)
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
                    tla.copy(l0_b, l1_b_by_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_available)
                        else:
                            tla.set_flag(l1b1_available)

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = 0
                    if tla.const_expr(ENABLE_UNIT_FLAG):
                        if (k_local == slice_tiles - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                    init_c = True if k_local == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx

                l1_buf_idx = c1 - l1_buf_idx

            # Normal tiles → GM C; tail tiles → per-AIC workspace slot.
            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.set_flag(l0c_data_ready)
                tla.wait_flag(l0c_data_ready)
                if is_tail == 1:
                    tla.copy(gm_w_by_tile, l0_c)
                else:
                    tla.copy(gm_c_by_core, l0_c)
                tla.set_flag(l0c_available)
            else:
                if is_tail == 1:
                    tla.copy(
                        gm_w_by_tile,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )
                else:
                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )

            # Tail AICs signal done after workspace store; normal-only AICs already signaled.
            if tla.const_expr(normal_block_num > 0):
                if loop_idx == normal_block_num - bdim + bid:
                    if bid < tail_cores:
                        tla.pipe_barrier(tla.pipes.ALL)
                        tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)
            else:
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)

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

    with tla.vector():
        tla.cross_core_wait_flag(cross_aic_to_aiv_done, tla.arch.MTE2)
        tla.cross_core_set_flag(cross_aiv_barrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(cross_aiv_barrier, tla.arch.MTE2)

        sub = tla.arch.sub_block_idx()
        aic_id = tla.arch.block_idx()
        aiv_id = aic_id * SUB_BLOCK_NUM + sub
        tail_limit = tail_block_num * splitk_factor

        do_reduce = 1 if aic_id < tail_limit else 0
        if do_reduce == 1:
            start_core = (aic_id // splitk_factor) * splitk_factor
            base_mn = normal_block_num + start_core // splitk_factor
            labor_core_num = splitk_factor * SUB_BLOCK_NUM
            loop_start = aiv_id - start_core * SUB_BLOCK_NUM

            # Swizzle for the tail M×N tile (same Zn/Nz as cube path).
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                tile_block_idx = base_mn // (SWIZZLE_OFFSET * grid_n)
                in_tile = base_mn % (SWIZZLE_OFFSET * grid_n)
                n_row = (
                    last_n_row
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_row
                block_col = in_tile // n_row
                odd = tile_block_idx % 2
                block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
            else:
                tile_block_loop = (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                tile_block_idx = base_mn // (SWIZZLE_OFFSET * grid_m)
                in_tile = base_mn % (SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            gm_c_by_core = tla.tile_view(
                gm_c, tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )
            m_act = gm_c_by_core.origin_shape[0]
            n_act = gm_c_by_core.origin_shape[1]
            c_plane = tla.tile_view(
                gm_c, tla.make_shape(m, n), tla.make_coord(c0, c0),
            )
            c_base_ptr = c_plane.ptr
            ws_plane = tla.tile_view(
                gm_w,
                tla.make_shape(aic_core_num * l1_tm, l1_tn),
                tla.make_coord(c0, c0),
            )
            ws_base_ptr = ws_plane.ptr

            ub_reduce_acc = tla.make_tensor(
                ub_reduce_ptr,
                tla.make_layout(
                    tla.make_shape(chunk_elems), tla.make_stride(1)
                ),
            )

            tla.set_flag(reduce_mte3_mte2)
            loops_num = (m_act + tile_per_core - 1) // tile_per_core
            chunk_range = tla.range(loop_start, loops_num, labor_core_num)
            for loop_idx in chunk_range:
                row_off = loop_idx * tile_per_core
                tiles_actual = tile_per_core
                remaining = m_act - row_off
                if remaining < tile_per_core:
                    tiles_actual = remaining

                tla.wait_flag(reduce_mte3_mte2)

                # Per-slice 2D gather: workspace rows are padded to L1_N, so a flat
                # contiguous read of tiles_actual*n_act is wrong when n_act < L1_N.
                gm_ws_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, l1_tn),
                    tla.make_stride(l1_tn, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                ub_ws_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, ub_row_stride),
                    tla.make_stride(ub_row_stride, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                for gather_sk_idx in tla.range(splitk_factor):
                    gm_ws_slice = tla.make_tensor(
                        ws_base_ptr
                        + (start_core + gather_sk_idx) * l1_tm * l1_tn
                        + row_off * l1_tn,
                        gm_ws_row_layout,
                    )
                    ub_ws_slice = tla.make_tensor(
                        ub_reduce_ptr + gather_sk_idx * chunk_elems,
                        ub_ws_row_layout,
                    )
                    tla.copy(ub_ws_slice, gm_ws_slice)

                # Flat UB view over gathered slices (each row = one padded chunk).
                ub_ws_gather = tla.make_tensor(
                    ub_reduce_ptr,
                    tla.make_layout(
                        tla.make_shape(splitk_factor, chunk_elems),
                        tla.make_stride(chunk_elems, 1),
                    ),
                )

                tla.set_flag(reduce_mte2_v)
                tla.wait_flag(reduce_mte2_v)

                # Sum split-K UB rows; optionally cast narrow types and densify stores.
                with tla.vec.func(mode="simd"):
                    add_mask = tla.create_mask(
                        pattern=tla.mask.ALL, dtype=tla.Float32
                    )
                    acc_chunk_shape = tla.make_shape(ELE_PER_VECTOR_BLOCK)
                    src_chunk_shape = tla.make_shape(1, ELE_PER_VECTOR_BLOCK)
                    for sk_idx in tla.range(1, splitk_factor):
                        for vl_idx in tla.range(reduce_vl_loops):
                            reduce_acc_chunk = tla.tile_view(
                                ub_reduce_acc,
                                acc_chunk_shape,
                                tla.make_coord(vl_idx),
                            )
                            reduce_src_chunk = tla.tile_view(
                                ub_ws_gather,
                                src_chunk_shape,
                                tla.make_coord(sk_idx, vl_idx),
                            )
                            reduce_acc_chunk.store(
                                reduce_acc_chunk.load() + reduce_src_chunk.load(),
                                mask=add_mask,
                            )

                    if tla.const_expr(NEED_CAST):
                        # f32→f16/bf16 cast leaves values in low-16 of each B32 slot;
                        # DIST_PACK_B32 packs those halves densely (replaces deinterleave).
                        # Mask must be ALL on the narrow dtype — VL64 only enables half the
                        # f16 lanes and would write only half a strip.
                        cast_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        store_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=DTYPE_GM_C
                        )
                        pack_store = NormalStoreParams(
                            store_dist=StoreDist.DIST_PACK_B32
                        )
                        ub_out_1d = tla.make_tensor(
                            ub_cast_ptr,
                            tla.make_layout(
                                tla.make_shape(chunk_elems), tla.make_stride(1)
                            ),
                        )
                        for cast_vl_idx in tla.range(reduce_vl_loops):
                            cast_acc_chunk = tla.tile_view(
                                ub_reduce_acc,
                                acc_chunk_shape,
                                tla.make_coord(cast_vl_idx),
                            )
                            cast_out_chunk = tla.tile_view(
                                ub_out_1d,
                                acc_chunk_shape,
                                tla.make_coord(cast_vl_idx),
                            )
                            acc_v = cast_acc_chunk.load()
                            out_v = acc_v.to(DTYPE_GM_C, cast_to_gm_params, cast_mask)
                            cast_out_chunk.store(out_v, pack_store, mask=store_mask)

                tla.set_flag(reduce_v_mte3)
                tla.wait_flag(reduce_v_mte3)

                gm_c_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, l1_tn),
                    tla.make_stride(n, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                gm_out_ptr = (
                    c_base_ptr
                    + (block_row * l1_tm + row_off) * n
                    + block_col * l1_tn
                )
                ub_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, ub_row_stride),
                    tla.make_stride(ub_row_stride, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                if tla.const_expr(NEED_CAST):
                    gm_out = tla.make_tensor(gm_out_ptr, gm_c_row_layout)
                    ub_cast = tla.make_tensor(ub_cast_ptr, ub_row_layout)
                    tla.copy(gm_out, ub_cast)
                else:
                    gm_out = tla.make_tensor(gm_out_ptr, gm_c_row_layout)
                    ub_out_fp32 = tla.make_tensor(ub_reduce_ptr, ub_row_layout)
                    tla.copy(gm_out, ub_out_fp32)

                tla.set_flag(reduce_mte3_mte2)

            tla.wait_flag(reduce_mte3_mte2)

        tla.pipe_barrier(tla.pipes.ALL)

# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def validate_dtype_triple(dtype_a: str, dtype_b: str, dtype_c: str) -> None:
    if dtype_a != dtype_b or dtype_a != dtype_c:
        raise SystemExit(
            "unsupported configuration:\n  - dtype-a, dtype-b, and dtype-c must match "
            f"(got {dtype_a}/{dtype_b}/{dtype_c}); allowed: f16 | bf16 | f32"
        )
    if dtype_a not in ("f16", "bf16", "f32"):
        raise SystemExit(f"unsupported dtype {dtype_a!r}")


def validate_shape(m_val: int, n_val: int, k_val: int) -> None:
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise SystemExit(f"m, n, k must be positive; got ({m_val},{n_val},{k_val})")


def compute_tail_scheduler(
    m_val: int, n_val: int, k_val: int, core_num: int
) -> dict[str, int]:
    """Split M×N grid into normal blocks (full-K → C) and tail blocks (split-K)."""
    if core_num <= 0:
        raise ValueError(f"core_num must be positive; got {core_num}")
    grid_m = ceil_div(m_val, l1_tm)
    grid_n = ceil_div(n_val, l1_tn)
    k_tile_num = ceil_div(k_val, l1_tk)
    mn_blocks = grid_m * grid_n
    t_num = mn_blocks % core_num
    n_num = mn_blocks - t_num
    factor = 1
    if t_num > 0:
        factor = core_num // t_num
    factor = min(factor, k_tile_num)
    loops = n_num + t_num * factor
    return {
        "grid_m": grid_m,
        "grid_n": grid_n,
        "k_tile_num": k_tile_num,
        "mn_blocks": mn_blocks,
        "tail_block_num": t_num,
        "normal_block_num": n_num,
        "splitk_factor": factor,
        "core_loops": loops,
        "aic_core_num": core_num,
    }


def workspace_shape(aic: int) -> tuple[int, int]:
    """Per-AIC L1 tile row in workspace; floor ≥10 MB."""
    min_elems = (10 * 1024 * 1024) // 4
    need = aic * l1_tm * l1_tn
    elems = max(min_elems, need)
    rows = max(aic * l1_tm, ceil_div(elems, l1_tn))
    return rows, l1_tn


def compute_tail_reduce_tiling(
    factor: int,
    *,
    l1_m: int = l1_tm,
    l1_n: int = l1_tn,
    compute_length: int = COMPUTE_LENGTH,
    ele_per_vector_block: int = ELE_PER_VECTOR_BLOCK,
    ele_align: int = ELE_NUM_ALIGN,
) -> dict[str, int]:
    """Tail tile row-chunk size, UB stride, and vector loop count per AIV."""
    labor = factor * 2
    tile_len_align = ceil_div(l1_n, ele_align) * ele_align
    tile_per_core_max = (compute_length // labor) // tile_len_align
    if tile_per_core_max == 0:
        tile_per_core_max = 1
    tpc = ceil_div(l1_m, labor)
    if tpc > tile_per_core_max:
        tpc = tile_per_core_max
    if tpc > l1_m:
        tpc = l1_m
    if tpc == 0:
        tpc = 1
    ub_stride = tile_len_align
    chunk = tpc * ub_stride
    while factor * chunk > compute_length and tpc > 1:
        tpc -= 1
        chunk = tpc * ub_stride
    if factor * chunk > compute_length:
        raise ValueError(
            f"tail reduce UB overflow: factor={factor} chunk={chunk} "
            f"compute_length={compute_length}"
        )
    return {
        "tile_per_core": tpc,
        "ub_row_stride": ub_stride,
        "reduce_vl_loops": ceil_div(chunk, ele_per_vector_block),
        "chunk_elems": chunk,
    }


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

    validate_dtype_triple(da, db, dc)
    validate_shape(mi, ni, ki)

    mod.DTYPE_A = tla_of[da]
    mod.DTYPE_B = tla_of[db]
    mod.DTYPE_C = tla.Float32
    mod.DTYPE_W = tla.Float32
    mod.DTYPE_GM_C = tla_of[dc]
    mod.NEED_CAST = dc != "f32"
    mod.CAST_FLOOR = dc == "f16"
    mod.SWIZZLE_DIRECTION = 0 if mi > ni else 1

    def create_tla_tensor(buf, layout: str):
        storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
        tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
        return from_dlpack(storage, layout_tag=tag).mark_layout_dynamic()

    cache_dir = str(Path(args.cache_dir).expanduser().resolve())

    tla.initialize(device=args.device)
    try:
        torch.npu.set_device(args.device)
        block_dim = max(
            1,
            args.block_dim if args.block_dim != -1 else tla.get_aicore_num(args.device),
        )
        sched = compute_tail_scheduler(mi, ni, ki, block_dim)
        reduce = compute_tail_reduce_tiling(sched["splitk_factor"])
        mod.aic_core_num = sched["aic_core_num"]
        mod.normal_block_num = sched["normal_block_num"]
        mod.tail_block_num = sched["tail_block_num"]
        mod.splitk_factor = sched["splitk_factor"]
        mod.core_loops = sched["core_loops"]
        mod.tile_per_core = reduce["tile_per_core"]
        mod.ub_row_stride = reduce["ub_row_stride"]
        mod.reduce_vl_loops = reduce["reduce_vl_loops"]
        mod.chunk_elems = reduce["chunk_elems"]

        print(
            f"--- mnk=({mi},{ni},{ki}) layout={la}/{lb} dtype={da}/{db}/{dc} "
            f"block_dim={block_dim} normal={sched['normal_block_num']} "
            f"tail={sched['tail_block_num']} factor={sched['splitk_factor']} "
            f"swizzle_dir={mod.SWIZZLE_DIRECTION} ---"
        )
        torch.npu.manual_seed(0)
        a = torch.rand(mi, ki, dtype=torch_of[da], device="npu") * 10.0 - 5.0
        b = torch.rand(ki, ni, dtype=torch_of[db], device="npu") * 10.0 - 5.0
        c = torch.full((mi, ni), args.sentinel, dtype=torch_of[dc], device="npu")
        ws_rows, ws_cols = workspace_shape(sched["aic_core_num"])
        w = torch.zeros((ws_rows, ws_cols), dtype=torch.float32, device="npu")
        expected = golden(a, b, torch_of[dc])

        ta = create_tla_tensor(a, la)
        tb = create_tla_tensor(b, lb)
        tc = create_tla_tensor(c, "row")
        tw = create_tla_tensor(w, "row")
        artifact = tla.compile(
            tail_multi_core_splitk_mmad_kernel,
            ta,
            tb,
            tc,
            tw,
            arch_scope="aic.c310",
            cache=not args.no_cache,
            cache_dir=cache_dir,
            force_recompile=args.force_recompile,
        )
        artifact(ta, tb, tc, tw, block_dim=block_dim)
        torch.npu.synchronize()

        if dc == "bf16":
            rtol = (1.0 / 128.0) if ki < 2048 else (1.0 / 64.0)
            floor = 1.0 / 256.0
        else:
            rtol = (1.0 / 256.0) if ki < 2048 else (1.0 / 128.0)
            floor = 1.0
        budget = 1.0 / 10000.0 if dc == "f32" else 1.0 / 1000.0
        got = c.detach().to(device="cpu", dtype=torch.float32)
        exp = expected.detach().to(device="cpu", dtype=torch.float32)
        thr = rtol * torch.maximum(torch.full_like(exp, floor), exp.abs())
        bad = (got - exp).abs() > thr
        bad = bad | torch.isnan(got) | torch.isinf(got)
        n_total = int(exp.numel())
        n_bad = int(bad.sum().item())
        mismatch_ratio = (n_bad / n_total) if n_total else 0.0
        passed = mismatch_ratio <= budget
        print(
            f"passed={passed} mismatch={100.0 * mismatch_ratio:.4f}% "
            f"(budget={100.0 * budget:.4f}%) cache_key={artifact.cache_key}"
        )
        print(f"kernel.o={artifact.kernel_binary_path}")
        return 0 if passed else 1
    finally:
        tla.finalize()


def main() -> int:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--layout-a", choices=("row", "col"), default="row")
    p.add_argument("--layout-b", choices=("row", "col"), default="row")
    p.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f16")
    p.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f16")
    p.add_argument("--dtype-c", choices=("f16", "bf16", "f32"), default="f32")
    p.add_argument("--block-dim", type=int, default=-1)
    p.add_argument("--sentinel", type=float, default=-7.0)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--force-recompile", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
