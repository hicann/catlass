"""Multi-core split-K matmul kernel (mixed AIC+AIV, example 68).

Sync sequence:
  AIC: partial MMAD → workspace → cross_core_set_flag(aic_finish, mode=2)
  AIV: cross_core_wait_flag(aic_finish) → cross_core_barrier(mode=0) → ReduceAdd → C

Dtype / layout (host sets DTYPE_* before compile):
  DTYPE_A / DTYPE_B : GM/L1/L0A/L0B (f16 | bf16 | f32), must match
  DTYPE_C           : L0C accumulate — always Float32
  DTYPE_W           : workspace + AIV UB — always Float32
  DTYPE_GM_C        : final GM C (same as A/B: f16 | bf16 | f32)
  A/B layout_tag    : RowMajor or ColumnMajor from host Tensor
  C/W layout        : RowMajor

Host injects compile-time globals on this module before ``tla.compile``:
  splitk_factor, element_count, task_per_aiv, reduce_loops,
  ub_row_stride, reduce_vl_loops
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

# Z-order swizzle: group M tiles by SWIZZLE_OFFSET rows.
SWIZZLE_OFFSET = 3

# AIV ReduceAdd: flat M×N chunks, gather all split-K slices into UB.
SUB_BLOCK_NUM = 2
ELE_PER_VECTOR_BLOCK = 64  # 256 / sizeof(fp32) — one vector load stride
ELE_NUM_ALIGN = 8          # BYTE_PER_BLK / sizeof(fp32) on Ascend950
COMPUTE_LENGTH = 192 * 1024 // 4  # Arch::Ascend950::UB_SIZE budget for reduce
REDUCE_STAGES = 1

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
def multi_core_splitk_mmad_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_w: tla.Tensor,
) -> None:
    """Single mixed kernel: AIC partial GEMM → workspace; AIV ReduceAdd → C."""
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
    # mode=2: paired AIC→AIV (CrossCoreSet/WaitFlag<0x2>)
    aic_finish = tla.cross_flag("aic_finish", mode=2)
    # mode=0: AIV inter-block barrier (CrossCoreBarrier<0x0>)
    aiv_ibarrier = tla.cross_flag("aiv_ibarrier", mode=0)

    # ReduceAdd pipeline flags (MTE3→MTE2→VECTOR→MTE3)
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

    # UB for AIV ReduceAdd: workspace gather buffer + output
    acc_ub_ptr = tla.allocate(COMPUTE_LENGTH, DTYPE_W, tla.AddressSpace.ub, 256)
    out_ub_ptr = tla.allocate(task_per_aiv, DTYPE_GM_C, tla.AddressSpace.ub, 256)

    # Cast parameters when GM C dtype != fp32 (f16: CAST_FLOOR, bf16: CAST_ROUND).
    _need_cast_to_gm = DTYPE_GM_C is not tla.Float32
    _cast_to_gm = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=(
            tla.params.RoundMode.CAST_FLOOR
            if DTYPE_GM_C is tla.Float16
            else tla.params.RoundMode.CAST_ROUND
        ),
    )

    # --------------------------------------------------------------------
    # Grid decomposition
    # --------------------------------------------------------------------
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    k_tile_num = (k + l1_tk - 1) // l1_tk
    mn_loops = grid_m * grid_n
    # Total iterations = M×N tiles × split-K slices, distributed across AICs
    core_loops = mn_loops * splitk_factor

    # ==========================================================================
    # AIC cube section — MMAD on tile slices → workspace
    # ==========================================================================
    with tla.cube():
        # Prime all flags so first wait in the loop body does not deadlock
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

        # Outer loop: distribute (M×N tile, split-K slice) across AICs
        task_range = tla.range(tla.arch.block_idx(), core_loops, tla.arch.block_dim())
        for task in task_range:
            # Decode task → split-K slice + M×N tile index
            slice_idx = (task % core_loops) // mn_loops
            inner_idx = task % mn_loops

            # Map linear tile index to (block_row, block_col) via Z-order swizzle.
            tile_block_loop = (grid_m + SWIZZLE_OFFSET - 1) // SWIZZLE_OFFSET
            tile_block_idx = inner_idx // (SWIZZLE_OFFSET * grid_n)
            in_tile_block_idx = inner_idx % (SWIZZLE_OFFSET * grid_n)
            n_row = SWIZZLE_OFFSET
            if tile_block_idx == tile_block_loop - 1:
                n_row = grid_m - SWIZZLE_OFFSET * tile_block_idx
            block_row = tile_block_idx * SWIZZLE_OFFSET + in_tile_block_idx % n_row
            block_col = in_tile_block_idx // n_row
            flipped_col = grid_n - block_col - 1
            if tile_block_idx % 2 == 1:
                block_col = flipped_col

            # --- Split-K slice: determine tile range for this AIC ---
            rem = k_tile_num % splitk_factor
            quot = k_tile_num // splitk_factor
            k_start = slice_idx * quot + rem
            slice_tiles = quot
            if slice_idx < rem:
                k_start = (quot + 1) * slice_idx
                slice_tiles = quot + 1

            # --- GM tile views for this task ---
            gm_a_by_core = tla.tile_view(
                mem_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                mem_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_w_mn = tla.tile_view(
                mem_w, tla.make_shape(m, n), tla.make_coord(slice_idx, c0)
            )
            gm_w_by_core = tla.tile_view(
                gm_w_mn, tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col)
            )

            l0_c = tla.make_tensor_like(l0c_ptr, gm_w_by_core)

            if not ENABLE_UNIT_FLAG:
                tla.wait_flag(fix_done)

            # --- L1-level K loop: prefetch K tiles ---
            k_l1_range = tla.range(c0, slice_tiles, c1)
            for k_local in k_l1_range:
                k_l1 = k_start + k_local
                gm_a_l1 = tla.tile_view(
                    gm_a_by_core, tla.make_shape(l1_tm, l1_tk),
                    tla.make_coord(c0, k_l1)
                )
                gm_b_l1 = tla.tile_view(
                    gm_b_by_core, tla.make_shape(l1_tk, l1_tn),
                    tla.make_coord(k_l1, c0)
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
                k_l0_range = tla.range(c0, k_l0_count, c1)
                for k_l0 in k_l0_range:
                    l1_a_l0 = tla.tile_view(
                        l1_a, tla.make_shape(l0_tm, l0_tk),
                        tla.make_coord(c0, k_l0)
                    )
                    l1_b_l0 = tla.tile_view(
                        l1_b, tla.make_shape(l0_tk, l0_tn),
                        tla.make_coord(k_l0, c0)
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
                    # Wait for L0 buffer availability
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

                # Toggle L1 double-buffer index
                l1_buf_idx = c1 - l1_buf_idx

            # Flush remaining partial MMAD result → workspace
            if not ENABLE_UNIT_FLAG:
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)
                tla.copy(gm_w_by_core, l0_c)
                tla.set_flag(fix_done)
            else:
                tla.copy(
                    gm_w_by_core,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )

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

        # Notify paired AIV that this AIC's workspace writes are done.
        tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    # ==========================================================================
    # AIV vector section — workspace ReduceAdd → GM C
    # ==========================================================================
    with tla.vector():
        # Wait for paired AIC, then barrier so *all* AICs have finished.
        tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(aiv_ibarrier, tla.arch.MTE2)

        # AIV ID = (AIC ID × 2 + sub-block)
        sub = tla.arch.sub_block_idx()
        aiv_id = tla.arch.block_idx() * SUB_BLOCK_NUM + sub
        aiv_num = tla.arch.block_dim() * SUB_BLOCK_NUM

        # GM pointers for workspace (W) and output (C)
        ws_plane = tla.tile_view(
            mem_w, tla.make_shape(m, n), tla.make_coord(c0, c0)
        )
        ws_base_ptr = ws_plane.ptr
        c_plane = tla.tile_view(
            mem_c, tla.make_shape(m, n), tla.make_coord(c0, c0)
        )
        c_base_ptr = c_plane.ptr

        # 1D UB row layout for ReduceAdd accumulation
        _row_layout = tla.make_layout(
            tla.make_shape(ub_row_stride), tla.make_stride(1)
        )
        ub_acc_1d = tla.make_tensor(acc_ub_ptr, _row_layout)
        # Per split-K slice UB tensor views (for accumulate-add)
        ub_tmp_1d = [
            tla.make_tensor(
                acc_ub_ptr + _bind_sk * ub_row_stride, _row_layout
            )
            for _bind_sk in range(1, splitk_factor)
        ]
        # UB output tensor (cast destination, same size as task chunk)
        ub_out_1d = tla.make_tensor(
            out_ub_ptr,
            tla.make_layout(tla.make_shape(task_per_aiv), tla.make_stride(1)),
        )

        tla.set_flag(reduce_mte3_mte2)

        # --- AIV reduce loop: gather all split-K slices, accumulate, write ---
        loop_range = tla.range(aiv_id, reduce_loops, aiv_num)
        for loop_idx in loop_range:
            src_off = loop_idx * task_per_aiv
            remaining = element_count - src_off
            actual_tile_len = task_per_aiv
            if remaining < task_per_aiv:
                actual_tile_len = remaining

            tla.wait_flag(reduce_mte3_mte2)

            # Bulk GM gather: all split-K slices → UB (contiguous rows)
            gm_gather = tla.make_tensor(
                ws_base_ptr + src_off,
                tla.make_layout(
                    tla.make_shape(splitk_factor, task_per_aiv),
                    tla.make_stride(element_count, 1),
                    origin_shape=tla.make_shape(splitk_factor, actual_tile_len),
                ),
            )
            ub_gather = tla.make_tensor(
                acc_ub_ptr,
                tla.make_layout(
                    tla.make_shape(splitk_factor, ub_row_stride),
                    tla.make_stride(ub_row_stride, 1),
                    origin_shape=tla.make_shape(splitk_factor, actual_tile_len),
                ),
            )
            tla.copy(ub_gather, gm_gather)

            tla.set_flag(reduce_mte2_v)
            tla.wait_flag(reduce_mte2_v)

            # Vector ReduceAdd: sum split-K slices; optional cast and deinterleave densify.
            with tla.vec.func(mode="simd"):
                add_mask = tla.create_mask(
                    pattern=tla.mask.ALL, dtype=tla.Float32
                )
                # Accumulate slice 1..splitK-1 into slice 0
                for _add_sk in range(1, splitk_factor):
                    tmp_1d = ub_tmp_1d[_add_sk - 1]
                    for _vl_i in tla.range(reduce_vl_loops):
                        acc_chunk = tla.tile_view(
                            ub_acc_1d,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(_vl_i),
                        )
                        tmp_chunk = tla.tile_view(
                            tmp_1d,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(_vl_i),
                        )
                        acc_chunk.store(
                            acc_chunk.load() + tmp_chunk.load(), mask=add_mask
                        )

                # --- Optional float→narrowed-dtype cast + densify ---
                if _need_cast_to_gm:
                    cast_mask = tla.create_mask(
                        pattern=tla.mask.ALL, dtype=tla.Float32
                    )
                    store_mask = tla.create_mask(
                        pattern=tla.mask.VL64, dtype=DTYPE_GM_C
                    )
                    for _vl_i in tla.range(reduce_vl_loops):
                        acc_chunk = tla.tile_view(
                            ub_acc_1d,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(_vl_i),
                        )
                        out_chunk = tla.tile_view(
                            ub_out_1d,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(_vl_i),
                        )
                        acc_v = acc_chunk.load()
                        out_v = acc_v.to(DTYPE_GM_C, _cast_to_gm, cast_mask)
                        zero_v = tla.full(0, dtype=DTYPE_GM_C)
                        dense_v, _odd_v = tla.deinterleave(out_v, zero_v)
                        out_chunk.store(dense_v, mask=store_mask)

            tla.set_flag(reduce_v_mte3)
            tla.wait_flag(reduce_v_mte3)

            # --- Write accumulated result to GM C ---
            if _need_cast_to_gm:
                gm_out = tla.make_tensor(
                    c_base_ptr + src_off,
                    tla.make_layout(
                        tla.make_shape(1, task_per_aiv),
                        tla.make_stride(n, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                ub_cast = tla.make_tensor(
                    out_ub_ptr,
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
                    acc_ub_ptr,
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
