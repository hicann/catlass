"""Tail multi-core split-K matmul kernel (mixed AIC+AIV, example 69).

Normal M×N tiles: full-K MMAD → GM C.
Tail M×N tiles: partial-K MMAD → per-AIC workspace, then AIV ReduceAdd → C.

Dtype / layout (host sets DTYPE_* before compile):
  DTYPE_A / DTYPE_B : GM/L1/L0A/L0B (f16 | bf16 | f32), must match
  DTYPE_C           : L0C accumulate — always Float32
  DTYPE_W           : workspace + AIV UB — always Float32
  DTYPE_GM_C        : final GM C (same as A/B: f16 | bf16 | f32)
  A/B layout_tag    : RowMajor or ColumnMajor from host Tensor
  C/W layout        : RowMajor

Host injects compile-time globals on this module before ``tla.compile``:
  aic_core_num, normal_block_num, tail_block_num, splitk_factor, core_loops,
  tile_per_core, ub_row_stride, reduce_vl_loops, chunk_elems
(M/N/K are read from ``mem_*.origin_shape`` at runtime.)
"""

from __future__ import annotations

import catlass as tla

# =============================================================================
# Fixed tiling / algorithm constants (kernel compile-time).
# =============================================================================
l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32

SWIZZLE_OFFSET = 3

# AIV ReduceAdd row-chunk tiling constants.
SUB_BLOCK_NUM = 2
ELE_PER_VECTOR_BLOCK = 64
ELE_NUM_ALIGN = 8
COMPUTE_LENGTH = 192 * 1024 // 4

VEC_TM = 16
VEC_TN = 32

# Host mutates DTYPE_A/B/GM_C before compile. L0C / W stay Float32.
DTYPE_A = tla.Float32
DTYPE_B = tla.Float32
DTYPE_C = tla.Float32  # L0C accumulate
DTYPE_W = tla.Float32  # workspace + UB
DTYPE_GM_C = tla.Float32
ENABLE_UNIT_FLAG = True


# =============================================================================
# Kernel entry point
# =============================================================================
@tla.kernel
def tail_multi_core_splitk_mmad_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_w: tla.Tensor,
) -> None:
    """Hybrid kernel:
    - Normal M×N tiles → full-K MMAD, result written directly to GM C.
    - Tail M×N tiles  → partial-K MMAD, per-AIC workspace W[block_idx],
      then AIV ReduceAdd across split-K slices → final C.
    """
    c0 = 0
    c1 = 1
    m = mem_a.origin_shape[0]
    n = mem_b.origin_shape[1]
    k = mem_a.origin_shape[1]

    # --------------------------------------------------------------------
    # Pipeline flags — AIC Cube/MTE1 <-> MTE2 <-> FIX sync
    # --------------------------------------------------------------------
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

    # Cross-core AIC → AIV synchronisation flags
    aic_finish = tla.cross_flag("aic_finish", mode=2)
    aiv_ibarrier = tla.cross_flag("aiv_ibarrier", mode=0)

    # ReduceAdd pipeline flags (VECTOR → MTE2 → VECTOR → MTE3)
    ub_load_ready = tla.flag("ub_load_ready", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    reduce_mte3_mte2 = tla.flag("reduce_mte3_mte2", tla.arch.MTE3, tla.arch.MTE2)
    reduce_mte2_v = tla.flag("reduce_mte2_v", tla.arch.MTE2, tla.arch.VECTOR)
    reduce_v_mte3 = tla.flag("reduce_v_mte3", tla.arch.VECTOR, tla.arch.MTE3)

    # --------------------------------------------------------------------
    # Scratch memory allocations
    # --------------------------------------------------------------------
    # L1 double-buffer: A (M×K) and B (K×N)
    l1a0_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(l1_tk * l1_tn, DTYPE_B, tla.AddressSpace.l1, 512)
    # L0A/B double-buffer inside Cube
    l0a0_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(l0_tm * l0_tk, DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(l0_tk * l0_tn, DTYPE_B, tla.AddressSpace.l0b, 512)
    # L0C accumulator
    l0c_ptr = tla.allocate(l0_tm * l0_tn, DTYPE_C, tla.AddressSpace.l0c, 512)

    # UB for AIV ReduceAdd
    acc_ub_ptr = tla.allocate(COMPUTE_LENGTH, DTYPE_W, tla.AddressSpace.ub, 256)
    out_ub_ptr = tla.allocate(chunk_elems, DTYPE_GM_C, tla.AddressSpace.ub, 256)

    # Cast parameters when GM C dtype != fp32.
    _need_cast_to_gm = DTYPE_GM_C is not tla.Float32
    _cast_to_gm = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )

    # --------------------------------------------------------------------
    # Grid decomposition
    # --------------------------------------------------------------------
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    k_tile_num = (k + l1_tk - 1) // l1_tk
    # Workspace viewed as (aic_core_num, L1_M, L1_N) stacked on rows.

    # ==========================================================================
    # AIC cube section — MMAD: normal tiles → GM C, tail tiles → workspace
    # ==========================================================================
    with tla.cube():
        # Synchronize before first task when reusing the device.
        tla.pipe_barrier(tla.pipes.ALL)

        # Prime all flags so first wait in loop body does not deadlock
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
        bid = tla.arch.block_idx()
        bdim = tla.arch.block_dim()
        tail_cores = core_loops - normal_block_num

        # Normal-only AICs signal completion once before the task loop.
        if bid >= tail_cores:
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        # --- Task loop: distribute (tile, split-K slice) across AICs ---
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

            # Compute K-split params
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

            # --- Swizzle M×N tile index → (block_row, block_col) ---
            tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
            tile_block_idx = base_block // (SWIZZLE_OFFSET * grid_n)
            in_tile_block_idx = base_block % (SWIZZLE_OFFSET * grid_n)
            n_row = SWIZZLE_OFFSET
            if tile_block_idx == tile_block_loop - 1:
                n_row = grid_m - SWIZZLE_OFFSET * tile_block_idx
            block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile_block_idx % n_row
            block_col = in_tile_block_idx // n_row
            if tile_block_idx % 2 == 1:
                block_col = grid_n - block_col - 1

            # --- GM tile views ---
            gm_a_by_core = tla.tile_view(
                mem_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                mem_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                mem_c, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
            )
            # Per-AIC workspace slot (L1_M×L1_N), cropped to actual tile shape.
            gm_w_slot = tla.tile_view(
                mem_w, tla.make_shape(l1_tm, l1_tn), tla.make_coord(bid, c0)
            )
            gm_w_by_core = tla.tile_view(
                gm_w_slot,
                tla.make_shape(
                    gm_c_by_core.origin_shape[0], gm_c_by_core.origin_shape[1]
                ),
                tla.make_coord(c0, c0),
            )

            # L0C tensor matches C tile origin_shape.
            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            if not ENABLE_UNIT_FLAG:
                tla.wait_flag(fix_done)

            # --- L1-level K loop: prefetch K tiles ---
            k_l1_range = tla.range(c0, slice_tiles, c1)
            for k_local in k_l1_range:
                k_l1 = k_start + k_local
                gm_a_l1 = tla.tile_view(
                    gm_a_by_core, tla.make_shape(l1_tm, l1_tk), tla.make_coord(c0, k_l1)
                )
                gm_b_l1 = tla.tile_view(
                    gm_b_by_core, tla.make_shape(l1_tk, l1_tn), tla.make_coord(k_l1, c0)
                )
                # Double-buffer: select A/B buffer by parity
                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                )
                # Wait for L1 buffer availability, then load A from GM
                if l1_buf_idx == c0:
                    tla.wait_flag(l1a0_copy_start)
                else:
                    tla.wait_flag(l1a1_copy_start)
                tla.copy(l1_a, gm_a_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_copy_end)
                else:
                    tla.set_flag(l1a1_copy_end)
                # Same for B
                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_copy_start)
                else:
                    tla.wait_flag(l1b1_copy_start)
                tla.copy(l1_b, gm_b_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_copy_end)
                else:
                    tla.set_flag(l1b1_copy_end)

                # --- L0-level K loop: L1→L0A/B, MMAD accumulate ---
                k_l0_count = (l1_a.origin_shape[1] + l0_tk - 1) // l0_tk
                for k_l0 in tla.range(c0, k_l0_count, c1):
                    l1_a_l0 = tla.tile_view(
                        l1_a, tla.make_shape(l0_tm, l0_tk), tla.make_coord(c0, k_l0)
                    )
                    l1_b_l0 = tla.tile_view(
                        l1_b, tla.make_shape(l0_tk, l0_tn), tla.make_coord(k_l0, c0)
                    )
                    # Select L0A/B buffer by parity (independent of L1 ping-pong)
                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_l0
                    )
                    # First L0 tile: drain L1→L0 pipe before copy
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
                    # Same for B
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

                    # MMAD with optional unit-flag for last L0 tile in slice
                    tla.set_flag(l0_copy_end)
                    tla.wait_flag(l0_copy_end)

                    unit_flag = 0
                    if ENABLE_UNIT_FLAG:
                        if (k_local == slice_tiles - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                    init_c = True if k_local == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    # Release L0A/B buffers for next iteration
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_copy_start)
                        tla.set_flag(l0b0_copy_start)
                    else:
                        tla.set_flag(l0a1_copy_start)
                        tla.set_flag(l0b1_copy_start)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            # Flush remaining partial MMAD result:
            # - normal tiles: directly to GM C
            # - tail tiles: to per-AIC workspace slot W[block_idx]
            if not ENABLE_UNIT_FLAG:
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)
                if is_tail == 1:
                    tla.copy(gm_w_by_core, l0_c)
                else:
                    tla.copy(gm_c_by_core, l0_c)
                tla.set_flag(fix_done)
            else:
                if is_tail == 1:
                    tla.copy(
                        gm_w_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )
                else:
                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )

            # Each AIC sets aic_finish once: before loop (normal-only) or after W store (tail).
            if normal_block_num > 0:
                if loop_idx == normal_block_num - bdim + bid:
                    if bid < tail_cores:
                        tla.pipe_barrier(tla.pipes.ALL)
                        tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
            else:
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        # Drain all pipelines
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

    # ==========================================================================
    # AIV vector section — reduce tail workspace slices → GM C
    # ==========================================================================
    with tla.vector():
        tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(aiv_ibarrier, tla.arch.MTE2)

        aic_id = tla.arch.block_idx()
        sub = tla.arch.sub_block_idx()
        aiv_id = aic_id * SUB_BLOCK_NUM + sub
        tail_limit = tail_block_num * splitk_factor

        # Dense (1, 64) UB views for vector load (pitch = 64).
        _dense_vl = tla.make_layout(
            tla.make_shape(1, ELE_PER_VECTOR_BLOCK),
            tla.make_stride(ELE_PER_VECTOR_BLOCK, 1),
        )
        vl_acc_chunks = [
            tla.make_tensor(
                acc_ub_ptr + _vi * ELE_PER_VECTOR_BLOCK, _dense_vl
            )
            for _vi in range(reduce_vl_loops)
        ]
        vl_tmp_chunks = [
            [
                tla.make_tensor(
                    acc_ub_ptr
                    + _sk * chunk_elems
                    + _vi * ELE_PER_VECTOR_BLOCK,
                    _dense_vl,
                )
                for _vi in range(reduce_vl_loops)
            ]
            for _sk in range(1, splitk_factor)
        ]
        vl_out_chunks = [
            tla.make_tensor(
                out_ub_ptr + _vi * ELE_PER_VECTOR_BLOCK, _dense_vl
            )
            for _vi in range(reduce_vl_loops)
        ]

        # Only AIVs paired with tail AICs participate in reduce
        do_reduce = 1 if aic_id < tail_limit else 0
        if do_reduce == 1:
            start_core = (aic_id // splitk_factor) * splitk_factor
            base_mn = normal_block_num + start_core // splitk_factor
            labor_core_num = splitk_factor * SUB_BLOCK_NUM
            loop_start = aiv_id - start_core * SUB_BLOCK_NUM

            # --- Swizzle for the tail M×N tile ---
            tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
            tile_block_idx = base_mn // (SWIZZLE_OFFSET * grid_n)
            in_tile_block_idx = base_mn % (SWIZZLE_OFFSET * grid_n)
            n_row = SWIZZLE_OFFSET
            if tile_block_idx == tile_block_loop - 1:
                n_row = grid_m - SWIZZLE_OFFSET * tile_block_idx
            block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile_block_idx % n_row
            block_col = in_tile_block_idx // n_row
            if tile_block_idx % 2 == 1:
                block_col = grid_n - block_col - 1

            gm_c_tile = tla.tile_view(
                mem_c,
                tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )
            m_act = gm_c_tile.origin_shape[0]
            n_act = gm_c_tile.origin_shape[1]
            c_plane = tla.tile_view(
                mem_c, tla.make_shape(m, n), tla.make_coord(c0, c0),
            )
            c_base_ptr = c_plane.ptr
            ws_plane = tla.tile_view(
                mem_w,
                tla.make_shape(aic_core_num * l1_tm, l1_tn),
                tla.make_coord(c0, c0),
            )
            ws_base_ptr = ws_plane.ptr

            # --- AIV reduce loop: gather split-K slices, accumulate, write ---
            loops_num = (m_act + tile_per_core - 1) // tile_per_core
            tla.set_flag(reduce_mte3_mte2)
            chunk_range = tla.range(loop_start, loops_num, labor_core_num)
            for loop_idx in chunk_range:
                row_off = loop_idx * tile_per_core
                tiles_actual = tile_per_core
                remaining = m_act - row_off
                if remaining < tile_per_core:
                    tiles_actual = remaining

                tla.wait_flag(reduce_mte3_mte2)

                # Layouts for GM workspace row layout, GM C row layout, and UB row layout
                gm_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, l1_tn),
                    tla.make_stride(l1_tn, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                gm_c_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, l1_tn),
                    tla.make_stride(n, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                ub_row_layout = tla.make_layout(
                    tla.make_shape(tile_per_core, ub_row_stride),
                    tla.make_stride(ub_row_stride, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )

                # Gather all split-K slices from workspace → UB
                for _slice_idx in range(splitk_factor):
                    ws_slice_ptr = (
                        ws_base_ptr
                        + (start_core + _slice_idx) * l1_tm * l1_tn
                        + row_off * l1_tn
                    )
                    gm_src = tla.make_tensor(ws_slice_ptr, gm_row_layout)
                    ub_dst = tla.make_tensor(
                        acc_ub_ptr + _slice_idx * chunk_elems,
                        ub_row_layout,
                    )
                    tla.copy(ub_dst, gm_src)

                tla.set_flag(reduce_mte2_v)
                tla.wait_flag(reduce_mte2_v)

                # Vector ReduceAdd: accumulate split-K slices in VL=64 strips.
                for _add_sk in range(1, splitk_factor):
                    with tla.vec.func(mode="simd"):
                        for _vl_i in range(reduce_vl_loops):
                            acc_chunk = vl_acc_chunks[_vl_i]
                            tmp_chunk = vl_tmp_chunks[_add_sk - 1][_vl_i]
                            acc_chunk.store(
                                acc_chunk.load() + tmp_chunk.load()
                            )

                # --- Optional float→narrowed-dtype cast + densify ---
                if _need_cast_to_gm:
                    with tla.vec.func(mode="simd"):
                        cast_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32,
                        )
                        store_mask = tla.create_mask(
                            pattern=tla.mask.VL64, dtype=DTYPE_GM_C,
                        )
                        for _vl_i in range(reduce_vl_loops):
                            acc_chunk = vl_acc_chunks[_vl_i]
                            out_chunk = vl_out_chunks[_vl_i]
                            acc_v = acc_chunk.load()
                            out_v = acc_v.to(DTYPE_GM_C, _cast_to_gm, cast_mask)
                            zero_v = tla.full(0, dtype=DTYPE_GM_C)
                            dense_v, _odd_v = tla.deinterleave(out_v, zero_v)
                            out_chunk.store(dense_v, mask=store_mask)

                tla.set_flag(reduce_v_mte3)
                tla.wait_flag(reduce_v_mte3)

                # Write accumulated result to GM C
                gm_out_ptr = (
                    c_base_ptr
                    + (block_row * l1_tm + row_off) * n
                    + block_col * l1_tn
                )
                gm_out = tla.make_tensor(gm_out_ptr, gm_c_row_layout)
                if _need_cast_to_gm:
                    ub_cast = tla.make_tensor(out_ub_ptr, ub_row_layout)
                    tla.copy(gm_out, ub_cast)
                else:
                    ub_out_fp32 = tla.make_tensor(acc_ub_ptr, ub_row_layout)
                    tla.copy(gm_out, ub_out_fp32)

                tla.set_flag(reduce_mte3_mte2)

            tla.wait_flag(reduce_mte3_mte2)

        tla.pipe_barrier(tla.pipes.ALL)
