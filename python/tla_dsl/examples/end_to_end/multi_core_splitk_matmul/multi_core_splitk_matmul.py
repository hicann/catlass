"""Multi-core split-K matmul: Kernel + Host in one file.

AIC partial MMAD → workspace; AIV ReduceAdd → C. Dynamic GM; CLI aligned with basic_matmul.
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
REDUCE_STAGES = 1

DTYPE_A = tla.Float32
DTYPE_B = tla.Float32
DTYPE_C = tla.Float32
DTYPE_W = tla.Float32
DTYPE_GM_C = tla.Float32

# Host injects these before compile (captured like DTYPE_*).
NEED_CAST = False
CAST_FLOOR = False
splitk_factor = 2
element_count = 1
task_per_aiv = 64
reduce_loops = 1
ub_row_stride = 64
reduce_vl_loops = 1

DESCRIPTION = "Multi-core split-K matmul; dynamic GM."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def multi_core_splitk_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_w: tla.Tensor,
) -> None:
    """AIC writes partial GEMM tiles to workspace; AIV reduces split-K slices to C."""
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

    # UB: split-K gather rows; optional narrow-type cast staging.
    ub_reduce_ptr = tla.allocate(COMPUTE_LENGTH, DTYPE_W, tla.AddressSpace.ub, 256)
    if tla.const_expr(NEED_CAST):
        ub_cast_ptr = tla.allocate(task_per_aiv, DTYPE_GM_C, tla.AddressSpace.ub, 256)

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
    mn_loops = grid_m * grid_n
    core_loops = mn_loops * splitk_factor
    # Tail band width (equals SWIZZLE_OFFSET when grid divides evenly).
    last_n_row = grid_m - SWIZZLE_OFFSET * (
        (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1
    )
    last_n_col = grid_n - SWIZZLE_OFFSET * (
        (grid_n + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET - 1
    )

    with tla.cube():
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

        task_range = tla.range(tla.arch.block_idx(), core_loops, tla.arch.block_num())
        for task in task_range:
            # Map linear task id → (split-K slice, M×N tile).
            slice_idx = (task % core_loops) // mn_loops
            inner_idx = task % mn_loops

            # Map linear tile index to (block_row, block_col) via Zn/Nz swizzle.
            # SWIZZLE_DIRECTION is host-set before compile (0=Zn when m>n, else Nz).
            # Body uses IfExp + branchless serpentine (same as batched_matmul) so
            # tla.const_expr outer fold does not nest dynamic statement-ifs.
            if tla.const_expr(SWIZZLE_DIRECTION == 0):
                # Zn: bands along M, serpentine N
                tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
                tile_block_idx = inner_idx // (SWIZZLE_OFFSET * grid_n)
                in_tile = inner_idx % (SWIZZLE_OFFSET * grid_n)
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
                tile_block_idx = inner_idx // (SWIZZLE_OFFSET * grid_m)
                in_tile = inner_idx % (SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            # Uneven K split: first (k % factor) slices get one extra K tile.
            rem = k_tile_num % splitk_factor
            quot = k_tile_num // splitk_factor
            k_start = slice_idx * quot + rem
            slice_tiles = quot
            if slice_idx < rem:
                k_start = (quot + 1) * slice_idx
                slice_tiles = quot + 1

            # GM views for this M×N tile and workspace slice row.
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_ws_plane = tla.tile_view(
                gm_w, tla.make_shape(m, n), tla.make_coord(slice_idx, c0)
            )
            gm_w_by_core = tla.tile_view(
                gm_ws_plane, tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col)
            )

            l0_c = tla.make_tensor_like(l0c_ptr, gm_w_by_core)

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

                # Select ping or pong L1 buffer for this K step.
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

                    # L0 A/B ping-pong is independent of L1 selection.
                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                    )

                    # First L0 sub-tile: wait until L1 load completes.
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

                    # MMAD; unit_flag marks last micro-tile when enabled.
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

                    # Release L0 ping-pong for the next L0 sub-tile.
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx

                l1_buf_idx = c1 - l1_buf_idx

            # Write L0C tile to workspace (unit-flag or explicit fix pipe).
            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.set_flag(l0c_data_ready)
                tla.wait_flag(l0c_data_ready)
                tla.copy(gm_w_by_core, l0_c)
                tla.set_flag(l0c_available)
            else:
                tla.copy(
                    gm_w_by_core,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )

        # Drain L1/L0 pipelines before cross-core handoff.
        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)

        tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        # Wait for paired AIC, then all-AIV barrier.
        tla.cross_core_wait_flag(cross_aic_to_aiv_done, tla.arch.MTE2)
        tla.cross_core_set_flag(cross_aiv_barrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(cross_aiv_barrier, tla.arch.MTE2)

        sub = tla.arch.sub_block_idx()
        aiv_id = tla.arch.block_idx() * SUB_BLOCK_NUM + sub
        aiv_num = tla.arch.block_num() * SUB_BLOCK_NUM

        gm_ws_plane = tla.tile_view(
            gm_w, tla.make_shape(m, n), tla.make_coord(c0, c0)
        )
        ws_base_ptr = gm_ws_plane.ptr
        c_plane = tla.tile_view(
            gm_c, tla.make_shape(m, n), tla.make_coord(c0, c0)
        )
        c_base_ptr = c_plane.ptr

        ub_row_layout = tla.make_layout(
            tla.make_shape(ub_row_stride), tla.make_stride(1)
        )
        ub_reduce_acc = tla.make_tensor(ub_reduce_ptr, ub_row_layout)

        tla.set_flag(reduce_mte3_mte2)

        loop_range = tla.range(aiv_id, reduce_loops, aiv_num)
        for loop_idx in loop_range:
            src_off = loop_idx * task_per_aiv
            remaining = element_count - src_off
            actual_tile_len = task_per_aiv
            if remaining < task_per_aiv:
                actual_tile_len = remaining

            tla.wait_flag(reduce_mte3_mte2)

            # Gather all split-K workspace rows for this flat M×N chunk into UB.
            gm_ws_gather = tla.make_tensor(
                ws_base_ptr + src_off,
                tla.make_layout(
                    tla.make_shape(splitk_factor, task_per_aiv),
                    tla.make_stride(element_count, 1),
                    origin_shape=tla.make_shape(splitk_factor, actual_tile_len),
                ),
            )
            ub_ws_gather = tla.make_tensor(
                ub_reduce_ptr,
                tla.make_layout(
                    tla.make_shape(splitk_factor, ub_row_stride),
                    tla.make_stride(ub_row_stride, 1),
                    origin_shape=tla.make_shape(splitk_factor, actual_tile_len),
                ),
            )
            tla.copy(ub_ws_gather, gm_ws_gather)

            tla.set_flag(reduce_mte2_v)
            tla.wait_flag(reduce_mte2_v)

            # Sum split-K UB rows; optionally cast narrow types and densify stores.
            with tla.vec.func(mode="simd"):
                add_mask = tla.create_mask(
                    pattern=tla.mask.ALL, dtype=tla.Float32
                )
                # Add workspace rows 1..factor-1 into row 0.
                vec_chunk_shape = tla.make_shape(1, ELE_PER_VECTOR_BLOCK)
                for sk_idx in tla.range(1, splitk_factor):
                    for vl_idx in tla.range(reduce_vl_loops):
                        reduce_acc_chunk = tla.tile_view(
                            ub_reduce_acc,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(vl_idx),
                        )
                        reduce_src_chunk = tla.tile_view(
                            ub_ws_gather,
                            vec_chunk_shape,
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
                            tla.make_shape(task_per_aiv), tla.make_stride(1)
                        ),
                    )
                    for cast_vl_idx in tla.range(reduce_vl_loops):
                        cast_acc_chunk = tla.tile_view(
                            ub_reduce_acc,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(cast_vl_idx),
                        )
                        cast_out_chunk = tla.tile_view(
                            ub_out_1d,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(cast_vl_idx),
                        )
                        acc_v = cast_acc_chunk.load()
                        out_v = acc_v.to(DTYPE_GM_C, cast_to_gm_params, cast_mask)
                        cast_out_chunk.store(out_v, pack_store, mask=store_mask)

            tla.set_flag(reduce_v_mte3)
            tla.wait_flag(reduce_v_mte3)

            if tla.const_expr(NEED_CAST):
                gm_out = tla.make_tensor(
                    c_base_ptr + src_off,
                    tla.make_layout(
                        tla.make_shape(1, task_per_aiv),
                        tla.make_stride(n, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                ub_cast = tla.make_tensor(
                    ub_cast_ptr,
                    tla.make_layout(
                        tla.make_shape(1, task_per_aiv),
                        tla.make_stride(task_per_aiv, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                tla.copy(gm_out, ub_cast)
            else:
                gm_out = tla.make_tensor(
                    c_base_ptr + src_off,
                    tla.make_layout(
                        tla.make_shape(1, task_per_aiv),
                        tla.make_stride(n, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                ub_out_fp32 = tla.make_tensor(
                    ub_reduce_ptr,
                    tla.make_layout(
                        tla.make_shape(1, task_per_aiv),
                        tla.make_stride(ub_row_stride, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
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


def get_splitk_factor(m_val: int, n_val: int, k_val: int, aic_core_num: int) -> int:
    factor = 2
    block_num = ceil_div(m_val, l1_tm) * ceil_div(n_val, l1_tn)
    k_tile_num = ceil_div(k_val, l1_tk)
    if aic_core_num // block_num > 0:
        factor = aic_core_num // block_num
    return min(factor, k_tile_num)


def workspace_elems(m_val: int, n_val: int, factor: int) -> int:
    min_elems = (2 * 1024 * 1024) // 4
    if factor * m_val * n_val >= min_elems:
        return factor * m_val * n_val
    rows = max(factor * m_val, ceil_div(min_elems, n_val))
    return rows * n_val


def workspace_shape(m_val: int, n_val: int, factor: int) -> tuple[int, int]:
    elems = workspace_elems(m_val, n_val, factor)
    rows = max(factor * m_val, ceil_div(elems, n_val))
    if rows % m_val != 0:
        rows = ceil_div(rows, m_val) * m_val
    return rows, n_val


def compute_reduce_tiling(
    m_val: int, n_val: int, factor: int, aic_core_num: int
) -> dict[str, int]:
    if aic_core_num <= 0:
        raise ValueError(f"aic_core_num must be positive; got {aic_core_num}")
    elem_count = m_val * n_val
    aiv_num = aic_core_num * 2
    per_aiv = ceil_div(elem_count, aiv_num)
    task = ceil_div(per_aiv, ELE_PER_VECTOR_BLOCK) * ELE_PER_VECTOR_BLOCK
    task_vl_cap = 512 * ELE_PER_VECTOR_BLOCK
    task_stage_cap = (
        COMPUTE_LENGTH // factor // ELE_PER_VECTOR_BLOCK * ELE_PER_VECTOR_BLOCK
    )
    task = min(task, task_stage_cap, task_vl_cap)
    if task == 0:
        task = ELE_PER_VECTOR_BLOCK
    ub_stride = ceil_div(task, ELE_NUM_ALIGN) * ELE_NUM_ALIGN
    if factor * ub_stride > COMPUTE_LENGTH:
        raise ValueError(
            f"splitk reduce UB overflow: factor={factor} ub_row_stride={ub_stride}"
        )
    loops = ceil_div(elem_count, task)
    return {
        "element_count": elem_count,
        "task_per_aiv": task,
        "reduce_loops": loops,
        "ub_row_stride": ub_stride,
        "reduce_vl_loops": ceil_div(task, ELE_PER_VECTOR_BLOCK),
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
        factor = get_splitk_factor(mi, ni, ki, block_dim)
        reduce = compute_reduce_tiling(mi, ni, factor, block_dim)
        mod.splitk_factor = factor
        mod.element_count = reduce["element_count"]
        mod.task_per_aiv = reduce["task_per_aiv"]
        mod.reduce_loops = reduce["reduce_loops"]
        mod.ub_row_stride = reduce["ub_row_stride"]
        mod.reduce_vl_loops = reduce["reduce_vl_loops"]

        print(
            f"--- mnk=({mi},{ni},{ki}) layout={la}/{lb} dtype={da}/{db}/{dc} "
            f"block_dim={block_dim} factor={factor} swizzle_dir={mod.SWIZZLE_DIRECTION} ---"
        )
        torch.npu.manual_seed(0)
        a = torch.rand(mi, ki, dtype=torch_of[da], device="npu") * 10.0 - 5.0
        b = torch.rand(ki, ni, dtype=torch_of[db], device="npu") * 10.0 - 5.0
        c = torch.full((mi, ni), args.sentinel, dtype=torch_of[dc], device="npu")
        ws_rows, ws_cols = workspace_shape(mi, ni, factor)
        w = torch.zeros((ws_rows, ws_cols), dtype=torch.float32, device="npu")
        expected = golden(a, b, torch_of[dc])

        ta = create_tla_tensor(a, la)
        tb = create_tla_tensor(b, lb)
        tc = create_tla_tensor(c, "row")
        tw = create_tla_tensor(w, "row")
        artifact = tla.compile(
            multi_core_splitk_mmad_kernel,
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

        # Match prior ex68 CompareData: rtol/floor + mismatch budget.
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
