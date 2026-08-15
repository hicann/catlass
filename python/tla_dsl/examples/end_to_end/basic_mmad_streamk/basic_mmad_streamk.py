"""StreamK MMAD demo: kernel + host in one file.

Builds and launches the single mixed kernel: the AIC cube runs the full-K normal
tiles and the StreamK partial sums, the AIV vector section reduces the workspace
into GM C, and the result is verified for accuracy by default. Supports the
``--all-layouts`` / ``--all-mmad-dtypes`` sweeps.

Compile-time knobs live in ``streamk_config``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import catlass as tla
from catlass.params import NormalStoreParams, StoreDist
from catlass.runtime import from_dlpack

import streamk_config as cfg

def workspace_rows(block_dim: int | None = None) -> int:
    """Workspace row count: ``l1_tm`` rows per core slot, two slots per AIC core."""
    if block_dim is None:
        block_dim = cfg.BLOCK_DIM
    return cfg.l1_tm * 2 * block_dim


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@tla.kernel
def streamk_mmad_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_workspace: tla.Tensor,
) -> None:
    c0 = 0
    c1 = 1
    m = mem_a.origin_shape[0]
    n = mem_b.origin_shape[1]
    k = mem_a.origin_shape[1]

    # StreamK grid constants from host knobs (Python ints), matching C++
    # ``StreamkGemmIdentityBlockSwizzle`` ctor / GetCoreLoops.
    # Tensor extents still come from GM.
    l1_tm_val = cfg.l1_tm
    l1_tn_val = cfg.l1_tn
    l1_tk_val = cfg.l1_tk
    block_dim = cfg.BLOCK_DIM
    loops_m = (m + l1_tm_val - 1) // l1_tm_val
    loops_n = (n + l1_tn_val - 1) // l1_tn_val
    loops_k = (k + l1_tk_val - 1) // l1_tk_val
    total_mn = loops_m * loops_n
    streamk_blocks = total_mn % block_dim
    normal_blocks = total_mn - streamk_blocks
    k_tile_num_per_core = (streamk_blocks * loops_k) // block_dim
    k_tile_remain = (streamk_blocks * loops_k) % block_dim
    core_loops = (total_mn // block_dim) * block_dim + min(
        streamk_blocks * loops_k, block_dim
    )
    streamk_cores = core_loops - normal_blocks

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
    # AIC→AIV handoff (mode 2) plus an all-AIV barrier (mode 0).
    aic_finish = tla.cross_flag("aic_finish", mode=2)
    aiv_ibarrier = tla.cross_flag("aiv_ibarrier", mode=0)

    l1a0_ptr = tla.allocate(cfg.l1_tm * cfg.l1_tk, cfg.DTYPE_A, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(cfg.l1_tm * cfg.l1_tk, cfg.DTYPE_A, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(cfg.l1_tk * cfg.l1_tn, cfg.DTYPE_B, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(cfg.l1_tk * cfg.l1_tn, cfg.DTYPE_B, tla.AddressSpace.l1, 512)

    l0a0_ptr = tla.allocate(cfg.l0_tm * cfg.l0_tk, cfg.DTYPE_A, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(cfg.l0_tm * cfg.l0_tk, cfg.DTYPE_A, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(cfg.l0_tk * cfg.l0_tn, cfg.DTYPE_B, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(cfg.l0_tk * cfg.l0_tn, cfg.DTYPE_B, tla.AddressSpace.l0b, 512)

    l0c_ptr = tla.allocate(cfg.l0_tm * cfg.l0_tn, cfg.DTYPE_C, tla.AddressSpace.l0c, 512)

    aiv_ub_tile_elems = cfg.AIV_TILE_M * cfg.l1_tn
    aiv_acc_ptr = tla.allocate(aiv_ub_tile_elems, cfg.DTYPE_C, tla.AddressSpace.ub, 256)
    aiv_temp_ptr = tla.allocate(aiv_ub_tile_elems, cfg.DTYPE_C, tla.AddressSpace.ub, 256)
    aiv_out_ptr = tla.allocate(
        aiv_ub_tile_elems, cfg.DTYPE_GM_C, tla.AddressSpace.ub, 256
    )
    aiv_ub_layout = tla.make_layout(
        tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
        tla.make_stride(cfg.l1_tn, c1),
        layoutTag=tla.arch.RowMajor,
    )
    aiv_acc_ub = tla.make_tensor(aiv_acc_ptr, aiv_ub_layout)
    aiv_temp_ub = tla.make_tensor(aiv_temp_ptr, aiv_ub_layout)
    aiv_out_ub = tla.make_tensor(aiv_out_ptr, aiv_ub_layout)

    aiv_loaded = tla.flag("aiv_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    aiv_vec_to_mte2 = tla.flag("aiv_vec_to_mte2", tla.arch.VECTOR, tla.arch.MTE2)
    aiv_done = tla.flag("aiv_done", tla.arch.VECTOR, tla.arch.MTE3)


    with tla.cube():
        # Drain pipe state left behind by a previous launch.
        tla.pipe_barrier(tla.pipes.ALL)

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
        block_idx = tla.arch.block_idx()

        # Cores without a StreamK task write only GM C, so nothing in the vector
        # section depends on them: release the paired AIV up front.
        if block_idx >= streamk_cores:
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        # Unified AIC schedule matching C++ streamk_matmul_tla:
        # remap loopIdx → actualLoopIdx (swap last normal with StreamK), then
        # GetStreamkBlockDec; CrossCoreSetFlag on PIPE_FIX after SK (no barrier).
        loop_range = tla.range(block_idx, core_loops, tla.arch.block_num())
        for loop_idx in loop_range:
            # actualLoopIdx remap (same conditions as streamk_matmul_tla.hpp).
            actual_loop_idx = loop_idx
            if normal_blocks > 0:
                if block_idx < streamk_cores:
                    swap_at = normal_blocks - tla.arch.block_num() + block_idx
                    if loop_idx == swap_at:
                        actual_loop_idx = normal_blocks + block_idx
                    elif loop_idx >= normal_blocks:
                        actual_loop_idx = swap_at
                elif loop_idx >= normal_blocks:
                    actual_loop_idx = (
                        normal_blocks - tla.arch.block_num() + block_idx
                    )
            # Same branch as GetStreamkBlockDec: SK when actual >= normal_blocks
            # (after remap, deferred normals land below normal_blocks again).
            is_sk_now = False
            if normal_blocks > 0:
                if actual_loop_idx >= normal_blocks:
                    is_sk_now = True
            else:
                is_sk_now = True

            # Prepare swizzle coordinates for normal and StreamK blocks.
            # Swizzle Direction fixed to RowMajor (0).
            swizzle_span = cfg.SWIZZLE_OFFSET * loops_n
            swizzle_tb_loop = (loops_m + cfg.SWIZZLE_OFFSET - 1) // cfg.SWIZZLE_OFFSET
            tile_block_idx = actual_loop_idx // swizzle_span
            in_tile = actual_loop_idx % swizzle_span
            n_row = cfg.SWIZZLE_OFFSET
            if tile_block_idx == (swizzle_tb_loop - 1):
                n_row = loops_m - cfg.SWIZZLE_OFFSET * tile_block_idx
            block_row = tile_block_idx * cfg.SWIZZLE_OFFSET + in_tile % n_row
            block_col = in_tile // n_row
            if (tile_block_idx % 2) == 1:
                block_col = loops_n - block_col - 1

            # Pre-init: normal / SK-non-cross use sk_slot_count=1; SK cross uses 2.
            block_k = c0
            block_actual_k = k
            streamk_block_row = block_row
            streamk_block_col = block_col
            streamk_block_k = c0
            streamk_actual_k = c0
            sk_task_id = block_idx
            sk_slot_count = 1

            if is_sk_now:
                # GetStreamkBlockDec StreamK path — coords only here.
                # blockCoord = current MN / K-tail; streamkBlockCoord = next MN / K-head.
                rel = actual_loop_idx - normal_blocks
                cur_k_tile_num = k_tile_num_per_core
                k_tile_idx = rel * k_tile_num_per_core + k_tile_remain
                if rel < k_tile_remain:
                    cur_k_tile_num = k_tile_num_per_core + 1
                    k_tile_idx = rel * cur_k_tile_num

                # --- blockCoord: override hoisted coords with current SK tile ---
                streamk_block_idx = k_tile_idx // loops_k
                block_linear = normal_blocks + streamk_block_idx
                block_tb_idx = block_linear // swizzle_span
                block_in_tile = block_linear % swizzle_span
                block_n_row = cfg.SWIZZLE_OFFSET
                if block_tb_idx == (swizzle_tb_loop - 1):
                    block_n_row = loops_m - cfg.SWIZZLE_OFFSET * block_tb_idx
                block_row = block_tb_idx * cfg.SWIZZLE_OFFSET + block_in_tile % block_n_row
                block_col = block_in_tile // block_n_row
                if (block_tb_idx % 2) == 1:
                    block_col = loops_n - block_col - 1
                block_k = k_tile_idx % loops_k
                block_actual_k = cur_k_tile_num * cfg.l1_tk
                if (k_tile_idx % loops_k + cur_k_tile_num) * cfg.l1_tk > k:
                    block_actual_k = k - (k_tile_idx % loops_k) * cfg.l1_tk

                # --- streamkBlockCoord: next tile K-head on cross ---
                streamk_block_row = block_row
                streamk_block_col = block_col
                streamk_block_k = c0
                streamk_actual_k = c0
                if k_tile_idx % loops_k + cur_k_tile_num > loops_k:
                    sk_slot_count = 2
                    next_streamk_block_idx = (k_tile_idx + cur_k_tile_num) // loops_k
                    streamk_linear = normal_blocks + next_streamk_block_idx
                    streamk_tb_idx = streamk_linear // swizzle_span
                    streamk_in_tile = streamk_linear % swizzle_span
                    streamk_n_row = cfg.SWIZZLE_OFFSET
                    if streamk_tb_idx == (swizzle_tb_loop - 1):
                        streamk_n_row = loops_m - cfg.SWIZZLE_OFFSET * streamk_tb_idx
                    streamk_block_row = (
                        streamk_tb_idx * cfg.SWIZZLE_OFFSET
                        + streamk_in_tile % streamk_n_row
                    )
                    streamk_block_col = streamk_in_tile // streamk_n_row
                    if (streamk_tb_idx % 2) == 1:
                        streamk_block_col = loops_n - streamk_block_col - 1
                    streamk_block_k = c0
                    streamk_actual_k = (
                        (k_tile_idx + cur_k_tile_num) % loops_k
                    ) * cfg.l1_tk

            # One mmad pipeline for: normal (count=1), SK non-cross (count=1),
            # and SK cross (count=2: blockCoord then streamkBlockCoord).
            sk_slots = tla.range(c0, sk_slot_count, c1)
            for sk_slot in sk_slots:
                sk_slot_row = block_row if sk_slot == 0 else streamk_block_row
                sk_slot_col = block_col if sk_slot == 0 else streamk_block_col
                sk_slot_block_k = block_k if sk_slot == 0 else streamk_block_k
                sk_slot_actual_k = (
                    block_actual_k if sk_slot == 0 else streamk_actual_k
                )
                sk_slot_ws_row = sk_task_id * 2 + sk_slot

                gm_a_by_core = tla.tile_view(
                    mem_a,
                    tla.make_shape(cfg.l1_tm, k),
                    tla.make_coord(sk_slot_row, c0),
                )
                gm_b_by_core = tla.tile_view(
                    mem_b,
                    tla.make_shape(k, cfg.l1_tn),
                    tla.make_coord(c0, sk_slot_col),
                )
                gm_c_by_core = tla.tile_view(
                    mem_c,
                    tla.make_shape(cfg.l1_tm, cfg.l1_tn),
                    tla.make_coord(sk_slot_row, sk_slot_col),
                )
                gm_ws_by_core = tla.tile_view(
                    mem_workspace,
                    tla.make_shape(cfg.l1_tm, cfg.l1_tn),
                    tla.make_coord(sk_slot_ws_row, c0),
                )
                l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

                k_l1_count = (sk_slot_actual_k + cfg.l1_tk - 1) // cfg.l1_tk
                k_l1_range = tla.range(c0, k_l1_count, c1)

                if not tla.const_expr(cfg.ENABLE_UNIT_FLAG):
                    tla.wait_flag(l0c_available)
                for k_l1_i in k_l1_range:
                    k_l1 = sk_slot_block_k + k_l1_i
                    gm_a_l1 = tla.tile_view(
                        gm_a_by_core,
                        tla.make_shape(cfg.l1_tm, cfg.l1_tk),
                        tla.make_coord(c0, k_l1),
                    )
                    gm_b_l1 = tla.tile_view(
                        gm_b_by_core,
                        tla.make_shape(cfg.l1_tk, cfg.l1_tn),
                        tla.make_coord(k_l1, c0),
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

                    k_l0_count = (l1_a.origin_shape[1] + cfg.l0_tk - 1) // cfg.l0_tk
                    k_l0_range = tla.range(c0, k_l0_count, c1)

                    for k_l0 in k_l0_range:
                        l1_a_l0 = tla.tile_view(
                            l1_a,
                            tla.make_shape(cfg.l0_tm, cfg.l0_tk),
                            tla.make_coord(c0, k_l0),
                        )
                        l1_b_l0 = tla.tile_view(
                            l1_b,
                            tla.make_shape(cfg.l0_tk, cfg.l0_tn),
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
                        if tla.const_expr(cfg.ENABLE_UNIT_FLAG):
                            if (k_l1_i == k_l1_count - 1) and (
                                k_l0 == k_l0_count - 1
                            ):
                                unit_flag = 0b11
                            else:
                                unit_flag = 0b10
                        init_c = True if k_l1_i == 0 and k_l0 == 0 else False
                        tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                        if l0_buf_idx == c0:
                            tla.set_flag(l0a0_available)
                            tla.set_flag(l0b0_available)
                        else:
                            tla.set_flag(l0a1_available)
                            tla.set_flag(l0b1_available)
                        l0_buf_idx = c1 - l0_buf_idx
                    l1_buf_idx = c1 - l1_buf_idx
                
                if is_sk_now:
                    if not tla.const_expr(cfg.ENABLE_UNIT_FLAG):
                        tla.set_flag(l0c_data_ready)
                        tla.wait_flag(l0c_data_ready)
                        tla.copy(gm_ws_by_core, l0_c)
                        tla.set_flag(l0c_available)
                    else:
                        tla.copy(
                            gm_ws_by_core,
                            l0_c,
                            tla.params.CopyL0C2DstParams(unit_flag=0b11),
                        )
                else:
                    if not tla.const_expr(cfg.ENABLE_UNIT_FLAG):
                        tla.set_flag(l0c_data_ready)
                        tla.wait_flag(l0c_data_ready)
                        tla.copy(gm_c_by_core, l0_c)
                        tla.set_flag(l0c_available)
                    else:
                        tla.copy(
                            gm_c_by_core,
                            l0_c,
                            tla.params.CopyL0C2DstParams(unit_flag=0b11),
                        )

            # Match C++: CrossCoreSetFlag on PIPE_FIX after SK (swap slot
            # or every SK when normal_blocks==0). No PipeBarrier before the
            # flag so AIV can overlap the deferred last normal.
            if is_sk_now:
                if normal_blocks > 0:
                    if block_idx < streamk_cores:
                        if loop_idx == normal_blocks - tla.arch.block_num() + block_idx:
                            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
                if normal_blocks == 0:
                    tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

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
        # Wait for the paired AIC, then for every AIC through the all-AIV barrier.
        tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(aiv_ibarrier, tla.arch.MTE2)
        # In a mix kernel block_idx() is the AIC id and sub_block_idx() the AIV half.
        aiv_id = tla.arch.block_idx()
        aiv_sub = tla.arch.sub_block_idx()
        aiv_global = aiv_id * cfg.AIV_SUB_BLOCK_NUM + aiv_sub

        # tla.range keeps the loop dynamic; a Python range would unroll it.
        for aiv_sk_id in tla.range(c0, streamk_blocks, c1):
            # GetCoreIdx / IsCross (block_swizzle.hpp) — same as C++ AIV reduce.
            # Fold kTileNumPerCore==0 at Python level to avoid %/÷ 0 in IR.
            aiv_start_core = c0
            aiv_end_core = c0
            aiv_head_cross = False
            aiv_tail_cross = False
            if k_tile_num_per_core == 0:
                aiv_start_core = aiv_sk_id * loops_k
                aiv_end_core = (aiv_sk_id + 1) * loops_k
                aiv_head_cross = False
                aiv_tail_cross = False
            else:
                aiv_threshold = k_tile_remain * (k_tile_num_per_core + 1)
                aiv_start_core = aiv_sk_id * loops_k // (k_tile_num_per_core + 1)
                if aiv_sk_id * loops_k > aiv_threshold:
                    aiv_start_core = k_tile_remain + (
                        aiv_sk_id * loops_k - aiv_threshold
                    ) // k_tile_num_per_core
                aiv_end_core = (aiv_sk_id + 1) * loops_k // (
                    k_tile_num_per_core + 1
                )
                if (aiv_sk_id + 1) * loops_k > aiv_threshold:
                    aiv_end_core = k_tile_remain + (
                        (aiv_sk_id + 1) * loops_k - aiv_threshold
                    ) // k_tile_num_per_core
                aiv_head_cross = (
                    aiv_sk_id * loops_k
                ) % (k_tile_num_per_core + 1) != 0
                if aiv_sk_id * loops_k > aiv_threshold:
                    aiv_head_numer = aiv_sk_id * loops_k - aiv_threshold
                    aiv_head_cross = aiv_head_numer % k_tile_num_per_core != 0
                aiv_tail_cross = (
                    (aiv_sk_id + 1) * loops_k
                ) % (k_tile_num_per_core + 1) != 0
                if (aiv_sk_id + 1) * loops_k > aiv_threshold:
                    aiv_tail_numer = (aiv_sk_id + 1) * loops_k - aiv_threshold
                    aiv_tail_cross = aiv_tail_numer % k_tile_num_per_core != 0
            # Workers are the cores that own a K slice of this tile. A trailing
            # cross core contributes one more slice to reduce but no worker.
            aiv_end_core_raw = aiv_end_core
            aiv_labor = (
                aiv_end_core_raw - aiv_start_core
            ) * cfg.AIV_SUB_BLOCK_NUM
            if aiv_tail_cross:
                aiv_end_core = aiv_end_core + 1

            aiv_linear = normal_blocks + aiv_sk_id
            aiv_span = cfg.SWIZZLE_OFFSET * loops_n
            aiv_tb_loop = (loops_m + cfg.SWIZZLE_OFFSET - 1) // cfg.SWIZZLE_OFFSET
            aiv_tb_idx = aiv_linear // aiv_span
            aiv_in_tile = aiv_linear % aiv_span
            aiv_n_row = cfg.SWIZZLE_OFFSET
            if aiv_tb_idx == (aiv_tb_loop - 1):
                aiv_n_row = loops_m - cfg.SWIZZLE_OFFSET * aiv_tb_idx
            aiv_block_row = (
                aiv_tb_idx * cfg.SWIZZLE_OFFSET + aiv_in_tile % aiv_n_row
            )
            aiv_block_col = aiv_in_tile // aiv_n_row
            if (aiv_tb_idx % 2) == 1:
                aiv_block_col = loops_n - aiv_block_col - 1

            aiv_tile_m = cfg.l1_tm
            if aiv_block_row == (loops_m - 1):
                aiv_tile_m = m - aiv_block_row * cfg.l1_tm

            aiv_slice_count = aiv_end_core - aiv_start_core
            aiv_m_loops = (aiv_tile_m + cfg.AIV_TILE_M - 1) // cfg.AIV_TILE_M
            aiv_rows_per_slot = cfg.l1_tm // cfg.AIV_TILE_M

            # Only AIVs paired with the producing AICs reduce this tile; they
            # split its row chunks between them.
            if aiv_id >= aiv_start_core:
                if aiv_id < aiv_end_core_raw:
                    aiv_loop_start = (
                        aiv_global - aiv_start_core * cfg.AIV_SUB_BLOCK_NUM
                    )
                    # tla.range needs a constexpr step, so hand each worker a
                    # contiguous [lo, hi) instead of striding by worker count.
                    aiv_chunk_per = (
                        aiv_m_loops + aiv_labor - 1
                    ) // aiv_labor
                    aiv_chunk_lo = aiv_loop_start * aiv_chunk_per
                    aiv_chunk_hi = aiv_chunk_lo + aiv_chunk_per
                    if aiv_chunk_lo > aiv_m_loops:
                        aiv_chunk_lo = aiv_m_loops
                    if aiv_chunk_hi > aiv_m_loops:
                        aiv_chunk_hi = aiv_m_loops
                    for aiv_m_idx in tla.range(aiv_chunk_lo, aiv_chunk_hi, c1):
                        aiv_c_row = aiv_block_row * aiv_rows_per_slot + aiv_m_idx
                        aiv_c_col = aiv_block_col
                        aiv_gm_c_tile = tla.tile_view(
                            mem_c,
                            tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
                            tla.make_coord(aiv_c_row, aiv_c_col),
                        )

                        aiv_init_pingpong = 0
                        if aiv_head_cross:
                            aiv_init_pingpong = 1
                        aiv_init_row = (
                            (aiv_start_core * 2 + aiv_init_pingpong) * aiv_rows_per_slot
                            + aiv_m_idx
                        )
                        aiv_ws_init = tla.tile_view(
                            mem_workspace,
                            tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
                            tla.make_coord(aiv_init_row, c0),
                        )
                        tla.copy(aiv_acc_ub, aiv_ws_init)
                        tla.set_flag(aiv_loaded)
                        tla.wait_flag(aiv_loaded)

                        aiv_slice_range = tla.range(c1, aiv_slice_count, c1)
                        for aiv_slice_idx in aiv_slice_range:
                            aiv_core = aiv_start_core + aiv_slice_idx
                            aiv_ws_row = (
                                (aiv_core * 2) * aiv_rows_per_slot + aiv_m_idx
                            )
                            aiv_ws_tile = tla.tile_view(
                                mem_workspace,
                                tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
                                tla.make_coord(aiv_ws_row, c0),
                            )
                            tla.copy(aiv_temp_ub, aiv_ws_tile)
                            tla.set_flag(aiv_loaded)
                            tla.wait_flag(aiv_loaded)
                            with tla.vec.func(mode="simd"):
                                for _aiv_rm in tla.range(0, cfg.AIV_M_CHUNKS, 1):
                                    for _aiv_rn in tla.range(
                                        0, cfg.AIV_N_CHUNKS, 1
                                    ):
                                        aiv_acc_chunk = tla.tile_view(
                                            aiv_acc_ub,
                                            tla.make_shape(cfg.AIV_REG_M, cfg.AIV_REG_N),
                                            tla.make_coord(_aiv_rm, _aiv_rn),
                                        )
                                        aiv_temp_chunk = tla.tile_view(
                                            aiv_temp_ub,
                                            tla.make_shape(cfg.AIV_REG_M, cfg.AIV_REG_N),
                                            tla.make_coord(_aiv_rm, _aiv_rn),
                                        )
                                        aiv_acc_chunk.store(
                                            tla.add(
                                                aiv_acc_chunk.load(),
                                                aiv_temp_chunk.load(),
                                                mask=tla.create_mask(pattern=tla.mask.ALL, dtype=cfg.DTYPE_C)
                                            ), 
                                            mask=tla.create_mask(pattern=tla.mask.ALL, dtype=cfg.DTYPE_C)
                                        )
                            tla.set_flag(aiv_vec_to_mte2)
                            tla.wait_flag(aiv_vec_to_mte2)

                        aiv_store_ub = aiv_acc_ub
                        if tla.const_expr(cfg.DTYPE_GM_C != cfg.DTYPE_C):
                            # multi_core_splitk pattern: 1D VL chunks, even-cast
                            # with f32 mask, then DIST_PACK_B32 densify.
                            if tla.const_expr(cfg.DTYPE_GM_C == tla.Float16):
                                aiv_cast_even = tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                    round_mode=tla.params.RoundMode.CAST_FLOOR,
                                )
                            else:
                                aiv_cast_even = tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                    round_mode=tla.params.RoundMode.CAST_ROUND,
                                )
                            aiv_pack_store = NormalStoreParams(
                                store_dist=StoreDist.DIST_PACK_B32
                            )
                            aiv_cast_vl_loops = (
                                cfg.AIV_M_CHUNKS * cfg.AIV_N_CHUNKS
                            )
                            aiv_acc_1d = tla.make_tensor(
                                aiv_acc_ptr,
                                tla.make_layout(
                                    tla.make_shape(aiv_ub_tile_elems),
                                    tla.make_stride(c1),
                                ),
                            )
                            aiv_out_1d = tla.make_tensor(
                                aiv_out_ptr,
                                tla.make_layout(
                                    tla.make_shape(aiv_ub_tile_elems),
                                    tla.make_stride(c1),
                                ),
                            )
                            with tla.vec.func(mode="simd"):
                                aiv_cast_mask = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=cfg.DTYPE_C
                                )
                                aiv_store_mask = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=cfg.DTYPE_GM_C
                                )
                                for aiv_cast_vl in tla.range(
                                    0, aiv_cast_vl_loops, 1
                                ):
                                    aiv_cast_src = tla.tile_view(
                                        aiv_acc_1d,
                                        tla.make_shape(cfg.AIV_REG_N),
                                        tla.make_coord(aiv_cast_vl),
                                    )
                                    aiv_cast_dst = tla.tile_view(
                                        aiv_out_1d,
                                        tla.make_shape(cfg.AIV_REG_N),
                                        tla.make_coord(aiv_cast_vl),
                                    )
                                    aiv_cast_h = aiv_cast_src.load().to(
                                        cfg.DTYPE_GM_C,
                                        aiv_cast_even,
                                        aiv_cast_mask,
                                    )
                                    aiv_cast_dst.store(
                                        aiv_cast_h,
                                        aiv_pack_store,
                                        mask=aiv_store_mask,
                                    )
                            aiv_store_ub = aiv_out_ub

                        tla.set_flag(aiv_done)
                        tla.wait_flag(aiv_done)
                        tla.copy(aiv_gm_c_tile, aiv_store_ub)
                        tla.pipe_barrier(tla.pipes.ALL)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"

LayoutChoice = Literal["row", "col"]
ElemDType = Literal["f16", "bf16", "f32"]

# Relative tolerance for result compare: tighter when K is below this threshold.
_COMPARE_RTOL_K_THRESHOLD = 2048
_COMPARE_RTOL_NUMERATOR = 1.0
_COMPARE_RTOL_DENOM_SMALL_K = 256
_COMPARE_RTOL_DENOM_LARGE_K = 128
# Extra pass gate: allow a small fraction of elements outside atol/rtol.
_COMPARE_MISMATCH_RATIO_NARROW = 0.001  # f16 / bf16: <= 0.1%
_COMPARE_MISMATCH_RATIO_F32 = 0.0001  # f32: <= 0.01%


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


def _validate_mmad_dtype_triple(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    if dtype_a != dtype_b:
        raise ValueError(
            "dtype-a and dtype-b must match (tla.mmad requires lhs and rhs element types equal)."
        )
    allowed = {
        ("f16", "f16", "f32"),
        ("f16", "f16", "f16"),
        ("bf16", "bf16", "f32"),
        ("bf16", "bf16", "bf16"),
        ("f32", "f32", "f32"),
    }
    triple = (dtype_a, dtype_b, dtype_c)
    if triple not in allowed:
        raise ValueError(
            "unsupported (dtype-a, dtype-b, dtype-c); allowed: "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32 "
            "(L0C is fp32; dtype-c is GM C element type, including narrowed f16/bf16)."
        )


def _apply_kernel_dtypes(
    dtype_a: ElemDType, dtype_b: ElemDType, dtype_c: ElemDType
) -> None:
    cfg.DTYPE_A = _tla_elem_dtype(dtype_a)
    cfg.DTYPE_B = _tla_elem_dtype(dtype_b)
    cfg.DTYPE_GM_C = _tla_elem_dtype(dtype_c)
    cfg.DTYPE_C = tla.Float32


def _apply_problem_size(m_val: int, n_val: int, k_val: int, block: int) -> None:
    if m_val <= 0 or n_val <= 0 or k_val <= 0:
        raise ValueError(f"m, n, k must be positive; got m={m_val}, n={n_val}, k={k_val}")
    if block <= 0:
        raise ValueError(f"block must be positive; got {block}")
    cfg.m = m_val
    cfg.n = n_val
    cfg.k = k_val
    cfg.BLOCK_DIM = block


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Host-side tensors in this example require PyTorch. "
            "Install it with ``pip install torch``."
        ) from exc
    return torch


def _torch_dtype(token: ElemDType) -> Any:
    torch = _require_torch()
    if token == "f16":
        return torch.float16
    if token == "bf16":
        return torch.bfloat16
    return torch.float32


def _require_torch_npu(device_id: int) -> Any:
    torch = _require_torch()
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit(
            "This example requires torch_npu for device DLPack bindings."
        ) from exc
    torch.npu.set_device(device_id)
    return torch


def _default_aic_block_dim(device_id: int) -> int:
    """Host launch block count matching runtime ``tla.arch.block_num()``.

    Uses ``tla.get_aicore_num`` after ``tla.initialize``. Falls back to
    ``BLOCK_DIM`` only if the runtime query is unavailable.
    """
    try:
        return max(1, int(tla.get_aicore_num(device_id)))
    except Exception:
        return max(1, int(cfg.BLOCK_DIM))


def _l1_padded_mn(m_val: int, n_val: int) -> tuple[int, int]:
    """Round MN up to L1 tile multiples so full-tile AIC GM access stays in bounds.

    The AIC addresses MN with SSA coords and always moves whole ``l1_tm x l1_tn``
    tiles, so a residual problem size needs zero-padded GM behind it.
    """
    pm = (m_val + cfg.l1_tm - 1) // cfg.l1_tm * cfg.l1_tm
    pn = (n_val + cfg.l1_tn - 1) // cfg.l1_tn * cfg.l1_tn
    return pm, pn


def _device_buffer_for_layout(dense: Any, choice: LayoutChoice) -> Any:
    if choice == "row":
        return dense.contiguous()
    return dense.permute(1, 0).contiguous()


def _create_tla_tensor(dev_buf: Any, layout: LayoutChoice) -> Any:
    return from_dlpack(
        _device_buffer_for_layout(dev_buf, layout),
        layout_tag=_gm_layout_tag(layout),
    ).mark_layout_dynamic()


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "arch_scope": "aic.c310",
        "cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "force_recompile": args.force_recompile,
    }


def _comparison_atol(dtype_c: ElemDType, args: argparse.Namespace) -> float:
    if dtype_c in ("f16", "bf16"):
        return max(float(args.atol), 5e-3)
    return float(args.atol)


def _comparison_rtol(k_val: int) -> float:
    """Pick relative tolerance from K: ``1/256`` if ``k < 2048``, else ``1/128``."""
    if k_val < _COMPARE_RTOL_K_THRESHOLD:
        return _COMPARE_RTOL_NUMERATOR / _COMPARE_RTOL_DENOM_SMALL_K
    return _COMPARE_RTOL_NUMERATOR / _COMPARE_RTOL_DENOM_LARGE_K


def _mismatch_ratio_budget(dtype_c: ElemDType) -> float:
    """Max fraction of out-of-tolerance elements still counted as pass."""
    if dtype_c in ("f16", "bf16"):
        return _COMPARE_MISMATCH_RATIO_NARROW
    return _COMPARE_MISMATCH_RATIO_F32


def _compare_expected_torch(
    actual: Any, expected: Any, *, rtol: float, atol: float, dtype_c: ElemDType
) -> dict[str, Any]:
    """Compare against golden with atol/rtol plus a small mismatch-ratio budget."""
    torch = _require_torch()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    total = int(actual.numel())
    mismatch_count = int((~close).sum().item())
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    budget = _mismatch_ratio_budget(dtype_c)
    all_close = mismatch_count == 0
    within_budget = mismatch_ratio <= budget
    ok = all_close or within_budget
    first_mismatch: dict[str, Any] | None = None
    if mismatch_count > 0:
        row, col = (
            int(value)
            for value in close.logical_not().nonzero(as_tuple=False)[0]
        )
        first_mismatch = {
            "index": [row, col],
            "actual": float(actual[row, col].item()),
            "expected": float(expected[row, col].item()),
        }
    return {
        "ok": ok,
        "all_close": all_close,
        "within_budget": within_budget,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_ratio,
        "mismatch_budget": budget,
        "total": total,
        "first_mismatch": first_mismatch,
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
    unchanged_all: bool | None = None,
    expected_match_all: bool | None = None,
    changed_count: int | None = None,
    first_mismatch: dict[str, Any] | None = None,
    mismatch_count: int | None = None,
    mismatch_ratio: float | None = None,
    mismatch_budget: float | None = None,
    verify: bool = True,
) -> None:
    print(
        "compile_ok=True "
        f"host={host} layout_a={layout_a} layout_b={layout_b} "
        f"dtype_a={dtype_a} dtype_b={dtype_b} dtype_c={dtype_c}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    if verify:
        print(f"C unchanged? {unchanged_all}")
        print(f"C equals expected matmul? {expected_match_all}")
        print(f"C changed count={changed_count}")
        if mismatch_count is not None and mismatch_ratio is not None:
            budget_s = (
                f"{mismatch_budget:.6f}" if mismatch_budget is not None else "n/a"
            )
            print(
                f"mismatch count={mismatch_count} "
                f"ratio={mismatch_ratio:.8f} budget={budget_s}"
            )
        print(f"first mismatch={first_mismatch}")


def run_single_case(
    args: argparse.Namespace,
    layout_a: LayoutChoice,
    layout_b: LayoutChoice,
    dtype_a: ElemDType,
    dtype_b: ElemDType,
    dtype_c: ElemDType,
) -> int:
    _apply_kernel_dtypes(dtype_a, dtype_b, dtype_c)
    torch = _require_torch_npu(args.device)
    device = "npu"
    print("m: ", cfg.m)
    print("n: ", cfg.n)
    print("k: ", cfg.k)
    torch_dtype_a = _torch_dtype(dtype_a)
    torch_dtype_b = _torch_dtype(dtype_b)
    torch_dtype_c = _torch_dtype(dtype_c)
    # Pad MN to L1 multiples so StreamK AIC full-tile GM DMA stays in-bounds.
    # Schedule / AIV still use the logical (m, n); only GM backing is padded.
    pad_m, pad_n = _l1_padded_mn(cfg.m, cfg.n)
    torch_tensor_a = torch.zeros((pad_m, cfg.k), dtype=torch_dtype_a, device=device)
    torch_tensor_b = torch.zeros((cfg.k, pad_n), dtype=torch_dtype_b, device=device)
    torch_tensor_a[:cfg.m, :] = (
        torch.empty((cfg.m, cfg.k), dtype=torch.float32, device=device).uniform_(-5.0, 5.0)
    ).to(torch_dtype_a)
    torch_tensor_b[:, :cfg.n] = (
        torch.empty((cfg.k, cfg.n), dtype=torch.float32, device=device).uniform_(-5.0, 5.0)
    ).to(torch_dtype_b)
    torch_tensor_c = torch.full(
        (pad_m, pad_n), args.sentinel, dtype=torch_dtype_c, device=device
    )
    ws_rows = workspace_rows(args.block)
    torch_workspace = torch.zeros(
        (ws_rows, cfg.l1_tn), dtype=torch.float32, device=device
    )
    verify = not args.no_verify
    expected = None
    atol = _comparison_atol(dtype_c, args)
    if verify:
        expected_f32 = torch_tensor_a[:cfg.m, :].to(torch.float32) @ torch_tensor_b[
            :, :cfg.n
        ].to(torch.float32)
        if dtype_c in ("f16", "bf16"):
            expected = expected_f32.to(torch_dtype_c).to(torch.float32)
        else:
            expected = expected_f32

    tla_tensor_a = _create_tla_tensor(torch_tensor_a, layout_a)
    tla_tensor_b = _create_tla_tensor(torch_tensor_b, layout_b)
    tla_tensor_c = _create_tla_tensor(torch_tensor_c, "row")
    tla_workspace = _create_tla_tensor(torch_workspace, "row")

    artifact = tla.compile(
        streamk_mmad_kernel,
        tla_tensor_a,
        tla_tensor_b,
        tla_tensor_c,
        tla_workspace,
        **_runtime_kwargs(args),
    )

    artifact(
        tla_tensor_a,
        tla_tensor_b,
        tla_tensor_c,
        tla_workspace,
        block_dim=args.block,
    )
    torch.npu.synchronize()

    if verify:
        rtol = _comparison_rtol(cfg.k)
        print("rtol: ", rtol)
        actual = torch_tensor_c[:cfg.m, :cfg.n].to(torch.float32)
        sentinel_f32 = torch.full_like(actual, args.sentinel)
        unchanged = torch.isclose(actual, sentinel_f32, rtol=rtol, atol=atol)
        cmp = _compare_expected_torch(
            actual, expected, rtol=rtol, atol=atol, dtype_c=dtype_c
        )
        _print_case_result(
            host="torch_npu",
            layout_a=layout_a,
            layout_b=layout_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_c=dtype_c,
            artifact=artifact,
            unchanged_all=bool(unchanged.all()),
            expected_match_all=bool(cmp["ok"]),
            changed_count=int((~unchanged).sum().item()),
            first_mismatch=cmp["first_mismatch"],
            mismatch_count=int(cmp["mismatch_count"]),
            mismatch_ratio=float(cmp["mismatch_ratio"]),
            mismatch_budget=float(cmp["mismatch_budget"]),
            verify=True,
        )
        return 0 if cmp["ok"] else 1

    _print_case_result(
        host="torch_npu",
        layout_a=layout_a,
        layout_b=layout_b,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        artifact=artifact,
        verify=False,
    )
    return 0


MMAD_DTYPE_TRIPLES: tuple[tuple[ElemDType, ElemDType, ElemDType], ...] = (
    ("f16", "f16", "f32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "f32"),
    ("bf16", "bf16", "bf16"),
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
        # Resolve auto block after ACL init so schedule BLOCK_DIM matches launch.
        if getattr(args, "auto_block", False):
            args.block = _default_aic_block_dim(args.device)
            _apply_problem_size(args.m, args.n, args.k, args.block)
            print(
                f"auto_block=True block={args.block} "
                "(tla.get_aicore_num / tla.arch.block_num)"
            )
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
            "Compile, launch, and validate StreamK MMAD (separate AIC cube + AIV vector). "
            "Tail-round StreamK on AIC + AIV workspace reduce. GM layouts for A/B are "
            "selectable; A/B must match; allowed (dtype-a, dtype-b, dtype-c): "
            "f16,f16,f32 | f16,f16,f16 | bf16,bf16,f32 | bf16,bf16,bf16 | f32,f32,f32. "
            "dtype-c is GM C element type; L0C stays fp32. Output C is GM row_major."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Compile, launch, and compare the full output matrix. This is the default.",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id.")
    parser.add_argument(
        "--m",
        type=int,
        default=cfg.m,
        help=f"GEMM M dimension (default: {cfg.m}).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=cfg.n,
        help=f"GEMM N dimension (default: {cfg.n}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=cfg.k,
        help=f"GEMM K dimension (default: {cfg.k}).",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help=(
            "Launch block count (AIC cores); also used as StreamK BLOCK_DIM. "
            "Default: tla.get_aicore_num (matches kernel tla.arch.block_num), else "
            f"{cfg.BLOCK_DIM}."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip golden matmul and accuracy checks after launch "
            "(compile/launch only; useful when measuring kernel runtime)."
        ),
    )
    parser.add_argument("--sentinel", type=float, default=-7.0, help="Initial C value.")
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Comparison tolerance."
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
        default="f16",
        help="GM element type for A (M×K); must equal --dtype-b for tla.mmad.",
    )
    parser.add_argument(
        "--dtype-b",
        type=_parse_elem_dtype,
        default="f16",
        help="GM element type for B (K×N); must equal --dtype-a.",
    )
    parser.add_argument(
        "--dtype-c",
        type=_parse_elem_dtype,
        default="f32",
        help="GM element type for C (M×N): f32, or narrowed f16/bf16 with f16/f16 or bf16/bf16 inputs.",
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
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    # None / -1 means take the block count from the device once ACL is up.
    args.auto_block = args.block is None or args.block < 0
    if args.auto_block:
        args.block = cfg.BLOCK_DIM
    _apply_problem_size(args.m, args.n, args.k, args.block)
    if not args.all_mmad_dtypes:
        _validate_mmad_dtype_triple(args.dtype_a, args.dtype_b, args.dtype_c)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
