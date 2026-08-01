"""StreamK MMAD: AIC cube and AIV vector in one ``streamk_mmad_kernel``.

- Only the tail round uses StreamK: ``streamk_blocks = total_mn % BLOCK_DIM``.
  The leading ``normal_blocks`` MN tiles run full-K straight to GM C; the tail
  tiles have their K tiles spread over the AIC cores, each core writing a partial
  sum to workspace.
- A K slice may straddle two MN tiles (cross-block), so each core owns two
  workspace slots of ``l1_tm x l1_tn``.
- MN task coordinates use the Zn swizzle (``SWIZZLE_OFFSET``).
- The cube signals ``aic_finish`` once its workspace writes have landed; the
  vector section waits on it, joins the all-AIV barrier, then reduces the
  workspace slices of each StreamK tile into GM C.

Config and schedule helpers live in ``streamk_config`` / ``streamk_schedule``.
SSA ``if`` / ``tla.range`` must stay in the ``@tla.kernel`` body (AST lowering).
"""

from __future__ import annotations

import catlass as tla

import streamk_config as cfg
import streamk_schedule as sched

# Schedule helpers re-exported for the host module.
_ceil_div = sched.ceil_div
_schedule_constants = sched.schedule_constants
workspace_rows = sched.workspace_rows

# Tile geometry aliases; the host mutates ``streamk_config`` instead.
l1_tm = cfg.l1_tm
l1_tn = cfg.l1_tn
l1_tk = cfg.l1_tk
l0_tm = cfg.l0_tm
l0_tn = cfg.l0_tn
l0_tk = cfg.l0_tk
SWIZZLE_OFFSET = cfg.SWIZZLE_OFFSET
AIV_TILE_M = cfg.AIV_TILE_M
AIV_REG_M = cfg.AIV_REG_M
AIV_REG_N = cfg.AIV_REG_N
AIV_M_CHUNKS = cfg.AIV_M_CHUNKS
AIV_N_CHUNKS = cfg.AIV_N_CHUNKS
AIV_DENSE_N = cfg.AIV_DENSE_N
AIV_N_DENSE_CHUNKS = cfg.AIV_N_DENSE_CHUNKS


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

    sched_consts = _schedule_constants()
    grid_m = sched_consts["loops_m"]
    grid_n = sched_consts["loops_n"]
    loops_k = sched_consts["loops_k"]
    normal_blocks = sched_consts["normal_blocks"]
    streamk_blocks = sched_consts["streamk_blocks"]
    streamk_cores = sched_consts["streamk_cores"]
    k_tile_num_per_core = sched_consts["k_tile_num_per_core"]
    k_tile_remain = sched_consts["k_tile_remain"]

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
    aiv_ws_dummy = tla.tile_view(
        mem_workspace, tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn), tla.make_coord(c0, c0)
    )
    aiv_c_dummy = tla.tile_view(
        mem_c, tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn), tla.make_coord(c0, c0)
    )
    aiv_acc_ub = tla.make_tensor_like(aiv_acc_ptr, aiv_ws_dummy, tla.arch.RowMajor)
    aiv_temp_ub = tla.make_tensor_like(aiv_temp_ptr, aiv_ws_dummy, tla.arch.RowMajor)
    aiv_out_ub = tla.make_tensor_like(aiv_out_ptr, aiv_c_dummy, tla.arch.RowMajor)

    aiv_loaded = tla.flag("aiv_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    aiv_vec_to_mte2 = tla.flag("aiv_vec_to_mte2", tla.arch.VECTOR, tla.arch.MTE2)
    aiv_done = tla.flag("aiv_done", tla.arch.VECTOR, tla.arch.MTE3)


    with tla.cube():
        # Drain pipe state left behind by a previous launch.
        tla.pipe_barrier(tla.pipes.ALL)

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
        block_idx = tla.arch.block_idx()

        # Cores without a StreamK task write only GM C, so nothing in the vector
        # section depends on them: release the paired AIV up front.
        if block_idx >= streamk_cores:
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        # Normal MN tasks: full-K MMAD straight to GM C, strided by block dim.
        normal_range = tla.range(block_idx, normal_blocks, tla.arch.block_dim())
        for block_linear in normal_range:
            _span = cfg.SWIZZLE_OFFSET * grid_n
            _tb_loop = (grid_m + cfg.SWIZZLE_OFFSET - 1) // cfg.SWIZZLE_OFFSET
            tile_block_idx = block_linear // _span
            in_tile = block_linear % _span
            n_row = cfg.SWIZZLE_OFFSET
            if tile_block_idx == (_tb_loop - 1):
                n_row = grid_m - cfg.SWIZZLE_OFFSET * tile_block_idx
            block_row = tile_block_idx * cfg.SWIZZLE_OFFSET + in_tile % n_row
            block_col = in_tile // n_row
            if (tile_block_idx % 2) == 1:
                block_col = grid_n - block_col - 1
            gm_a_by_core = tla.tile_view(
                mem_a, tla.make_shape(cfg.l1_tm, cfg.k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                mem_b, tla.make_shape(cfg.k, cfg.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                mem_c,
                tla.make_shape(cfg.l1_tm, cfg.l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + cfg.l1_tk - 1) // cfg.l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            if not cfg.ENABLE_UNIT_FLAG:
                tla.wait_flag(fix_done)
            for k_l1 in k_l1_range:
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
                    if cfg.ENABLE_UNIT_FLAG:
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

            if not cfg.ENABLE_UNIT_FLAG:
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)
                tla.copy(gm_c_by_core, l0_c)
                tla.set_flag(fix_done)
            else:
                tla.copy(
                    gm_c_by_core, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11)
                )

        # StreamK tasks: one per core while block_idx < streamk_cores. The primary
        # tile and the cross-block tile share the MMAD body via the slot loop.
        sk_range = tla.range(block_idx, streamk_cores, tla.arch.block_dim())
        for sk_task_id in sk_range:
            rel = sk_task_id
            cur_k_tile_num = k_tile_num_per_core
            k_tile_idx = rel * k_tile_num_per_core + k_tile_remain
            if rel < k_tile_remain:
                cur_k_tile_num = k_tile_num_per_core + 1
                k_tile_idx = rel * cur_k_tile_num

            streamk_block_idx = k_tile_idx // loops_k
            # Primary MN tile of this K slice (slot 0).
            main_linear = normal_blocks + streamk_block_idx
            _sk_span = cfg.SWIZZLE_OFFSET * grid_n
            _sk_tb_loop = (grid_m + cfg.SWIZZLE_OFFSET - 1) // cfg.SWIZZLE_OFFSET
            main_tb_idx = main_linear // _sk_span
            main_in_tile = main_linear % _sk_span
            main_n_row = cfg.SWIZZLE_OFFSET
            if main_tb_idx == (_sk_tb_loop - 1):
                main_n_row = grid_m - cfg.SWIZZLE_OFFSET * main_tb_idx
            main_row = main_tb_idx * cfg.SWIZZLE_OFFSET + main_in_tile % main_n_row
            main_col = main_in_tile // main_n_row
            if (main_tb_idx % 2) == 1:
                main_col = grid_n - main_col - 1
            main_block_k = k_tile_idx % loops_k
            main_actual_k = cur_k_tile_num * cfg.l1_tk
            if (k_tile_idx % loops_k + cur_k_tile_num) * cfg.l1_tk > cfg.k:
                main_actual_k = cfg.k - (k_tile_idx % loops_k) * cfg.l1_tk

            sk_is_cross = k_tile_idx % loops_k + cur_k_tile_num > loops_k
            # Cross-block second MN tile (slot 1); unused unless sk_is_cross.
            sk_row = main_row
            sk_col = main_col
            sk_block_k = 0
            sk_actual_k = 0
            sk_slot_count = 1
            if sk_is_cross:
                sk_slot_count = 2
                next_sk_block_idx = (k_tile_idx + cur_k_tile_num) // loops_k
                sk_linear = normal_blocks + next_sk_block_idx
                sk_tb_idx = sk_linear // _sk_span
                sk_in_tile = sk_linear % _sk_span
                sk_n_row = cfg.SWIZZLE_OFFSET
                if sk_tb_idx == (_sk_tb_loop - 1):
                    sk_n_row = grid_m - cfg.SWIZZLE_OFFSET * sk_tb_idx
                sk_row = sk_tb_idx * cfg.SWIZZLE_OFFSET + sk_in_tile % sk_n_row
                sk_col = sk_in_tile // sk_n_row
                if (sk_tb_idx % 2) == 1:
                    sk_col = grid_n - sk_col - 1
                sk_block_k = 0
                sk_actual_k = ((k_tile_idx + cur_k_tile_num) % loops_k) * cfg.l1_tk

            # One slot per MN tile this K slice touches: 1 normally, 2 when crossing.
            sk_slots = tla.range(c0, sk_slot_count, c1)
            for sk_slot in sk_slots:
                sk_slot_row = main_row if sk_slot == 0 else sk_row
                sk_slot_col = main_col if sk_slot == 0 else sk_col
                sk_slot_block_k = main_block_k if sk_slot == 0 else sk_block_k
                sk_slot_actual_k = main_actual_k if sk_slot == 0 else sk_actual_k
                sk_slot_ws_row = sk_task_id * 2 + sk_slot

                sk_gm_a_by_core = tla.tile_view(
                    mem_a,
                    tla.make_shape(cfg.l1_tm, cfg.k),
                    tla.make_coord(sk_slot_row, c0),
                )
                sk_gm_b_by_core = tla.tile_view(
                    mem_b,
                    tla.make_shape(cfg.k, cfg.l1_tn),
                    tla.make_coord(c0, sk_slot_col),
                )
                sk_gm_ws = tla.tile_view(
                    mem_workspace,
                    tla.make_shape(cfg.l1_tm, cfg.l1_tn),
                    tla.make_coord(sk_slot_ws_row, c0),
                )
                sk_l0c_like = tla.tile_view(
                    mem_c,
                    tla.make_shape(cfg.l1_tm, cfg.l1_tn),
                    tla.make_coord(sk_slot_row, sk_slot_col),
                )
                sk_l0_c = tla.make_tensor_like(l0c_ptr, sk_l0c_like)

                if not cfg.ENABLE_UNIT_FLAG:
                    tla.wait_flag(fix_done)

                sk_k_l1_count = (sk_slot_actual_k + cfg.l1_tk - 1) // cfg.l1_tk
                sk_k_l1_range = tla.range(c0, sk_k_l1_count, c1)
                sk_l1_buf_idx = c0
                sk_l0_buf_idx = c0

                for sk_k_l1_i in sk_k_l1_range:
                    sk_k_l1 = sk_slot_block_k + sk_k_l1_i
                    sk_gm_a_l1 = tla.tile_view(
                        sk_gm_a_by_core,
                        tla.make_shape(cfg.l1_tm, cfg.l1_tk),
                        tla.make_coord(c0, sk_k_l1),
                    )
                    sk_gm_b_l1 = tla.tile_view(
                        sk_gm_b_by_core,
                        tla.make_shape(cfg.l1_tk, cfg.l1_tn),
                        tla.make_coord(sk_k_l1, c0),
                    )

                    sk_l1_a = tla.make_tensor_like(
                        l1a0_ptr if (sk_l1_buf_idx == c0) else l1a1_ptr, sk_gm_a_l1
                    )
                    sk_l1_b = tla.make_tensor_like(
                        l1b0_ptr if (sk_l1_buf_idx == c0) else l1b1_ptr, sk_gm_b_l1
                    )
                    if sk_l1_buf_idx == c0:
                        tla.wait_flag(l1a0_copy_start)
                    else:
                        tla.wait_flag(l1a1_copy_start)
                    tla.copy(sk_l1_a, sk_gm_a_l1)
                    if sk_l1_buf_idx == c0:
                        tla.set_flag(l1a0_copy_end)
                    else:
                        tla.set_flag(l1a1_copy_end)

                    if sk_l1_buf_idx == c0:
                        tla.wait_flag(l1b0_copy_start)
                    else:
                        tla.wait_flag(l1b1_copy_start)
                    tla.copy(sk_l1_b, sk_gm_b_l1)
                    if sk_l1_buf_idx == c0:
                        tla.set_flag(l1b0_copy_end)
                    else:
                        tla.set_flag(l1b1_copy_end)

                    sk_k_l0_count = (sk_l1_a.origin_shape[1] + cfg.l0_tk - 1) // cfg.l0_tk
                    sk_k_l0_range = tla.range(c0, sk_k_l0_count, c1)

                    for sk_k_l0_i in sk_k_l0_range:
                        sk_l1_a_l0 = tla.tile_view(
                            sk_l1_a,
                            tla.make_shape(cfg.l0_tm, cfg.l0_tk),
                            tla.make_coord(c0, sk_k_l0_i),
                        )
                        sk_l1_b_l0 = tla.tile_view(
                            sk_l1_b,
                            tla.make_shape(cfg.l0_tk, cfg.l0_tn),
                            tla.make_coord(sk_k_l0_i, c0),
                        )

                        sk_l0_a = tla.make_tensor_like(
                            l0a0_ptr if (sk_l0_buf_idx == c0) else l0a1_ptr, sk_l1_a_l0
                        )
                        sk_l0_b = tla.make_tensor_like(
                            l0b0_ptr if (sk_l0_buf_idx == c0) else l0b1_ptr, sk_l1_b_l0
                        )
                        if sk_k_l0_i == 0:
                            if sk_l1_buf_idx == c0:
                                tla.wait_flag(l1a0_copy_end)
                            else:
                                tla.wait_flag(l1a1_copy_end)

                        if sk_l0_buf_idx == c0:
                            tla.wait_flag(l0a0_copy_start)
                        else:
                            tla.wait_flag(l0a1_copy_start)
                        tla.copy(sk_l0_a, sk_l1_a_l0)
                        if sk_k_l0_i == sk_k_l0_count - 1:
                            if sk_l1_buf_idx == c0:
                                tla.set_flag(l1a0_copy_start)
                            else:
                                tla.set_flag(l1a1_copy_start)

                        if sk_k_l0_i == 0:
                            if sk_l1_buf_idx == c0:
                                tla.wait_flag(l1b0_copy_end)
                            else:
                                tla.wait_flag(l1b1_copy_end)
                        if sk_l0_buf_idx == c0:
                            tla.wait_flag(l0b0_copy_start)
                        else:
                            tla.wait_flag(l0b1_copy_start)
                        tla.copy(sk_l0_b, sk_l1_b_l0)
                        if sk_k_l0_i == sk_k_l0_count - 1:
                            if sk_l1_buf_idx == c0:
                                tla.set_flag(l1b0_copy_start)
                            else:
                                tla.set_flag(l1b1_copy_start)

                        tla.set_flag(l0_copy_end)
                        tla.wait_flag(l0_copy_end)

                        sk_unit_flag = 0
                        if cfg.ENABLE_UNIT_FLAG:
                            if (sk_k_l1_i == sk_k_l1_count - 1) and (
                                sk_k_l0_i == sk_k_l0_count - 1
                            ):
                                sk_unit_flag = 0b11
                            else:
                                sk_unit_flag = 0b10
                        sk_init_c = True if sk_k_l1_i == 0 and sk_k_l0_i == 0 else False
                        tla.mmad(
                            sk_l0_c,
                            sk_l0_a,
                            sk_l0_b,
                            init_c=sk_init_c,
                            unit_flag=sk_unit_flag,
                        )
                        if sk_l0_buf_idx == c0:
                            tla.set_flag(l0a0_copy_start)
                            tla.set_flag(l0b0_copy_start)
                        else:
                            tla.set_flag(l0a1_copy_start)
                            tla.set_flag(l0b1_copy_start)
                        sk_l0_buf_idx = c1 - sk_l0_buf_idx
                    sk_l1_buf_idx = c1 - sk_l1_buf_idx

                if not cfg.ENABLE_UNIT_FLAG:
                    tla.set_flag(mmad_done)
                    tla.wait_flag(mmad_done)
                    tla.copy(sk_gm_ws, sk_l0_c)
                    tla.set_flag(fix_done)
                else:
                    tla.copy(
                        sk_gm_ws,
                        sk_l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )

            # No normal tasks: signal after each task stores its workspace slot.
            if tla.const_expr(normal_blocks == 0):
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        # With normal tasks present, a StreamK core signals once all its slots land.
        if tla.const_expr(normal_blocks > 0):
            if block_idx < streamk_cores:
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

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

    with tla.vector():
        # Wait for the paired AIC, then for every AIC through the all-AIV barrier.
        tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(aiv_ibarrier, tla.arch.MTE2)
        # In a mix kernel block_idx() is the AIC id and sub_block_idx() the AIV half.
        aiv_id = tla.arch.block_idx()
        aiv_sub = tla.arch.sub_block_idx()
        aiv_global = aiv_id * cfg.AIV_SUB_BLOCK_NUM + aiv_sub
        loops_k = sched_consts["loops_k"]
        k_tile_num_per_core = sched_consts["k_tile_num_per_core"]
        k_tile_remain = sched_consts["k_tile_remain"]
        # tla.range keeps the loop dynamic; a Python range would unroll it.
        for aiv_sk_id in tla.range(c0, streamk_blocks, c1):
            # Core range and cross flags of this tile; pre-init so the dynamic
            # ifs below only reassign already-bound names.
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

            # Zn swizzle: tile index -> (row, col).
            aiv_linear = normal_blocks + aiv_sk_id
            aiv_span = cfg.SWIZZLE_OFFSET * grid_n
            aiv_tb_loop = (grid_m + cfg.SWIZZLE_OFFSET - 1) // cfg.SWIZZLE_OFFSET
            aiv_tb_idx = aiv_linear // aiv_span
            aiv_in_tile = aiv_linear % aiv_span
            aiv_n_row = cfg.SWIZZLE_OFFSET
            if aiv_tb_idx == (aiv_tb_loop - 1):
                aiv_n_row = grid_m - cfg.SWIZZLE_OFFSET * aiv_tb_idx
            aiv_block_row = (
                aiv_tb_idx * cfg.SWIZZLE_OFFSET + aiv_in_tile % aiv_n_row
            )
            aiv_block_col = aiv_in_tile // aiv_n_row
            if (aiv_tb_idx % 2) == 1:
                aiv_block_col = grid_n - aiv_block_col - 1

            aiv_tile_m = cfg.l1_tm
            if aiv_block_row == (grid_m - 1):
                aiv_tile_m = cfg.m - aiv_block_row * cfg.l1_tm

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
                        aiv_acc_view = tla.tile_view(
                            aiv_acc_ub,
                            tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
                            tla.make_coord(c0, c0),
                        )
                        aiv_temp_view = tla.tile_view(
                            aiv_temp_ub,
                            tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
                            tla.make_coord(c0, c0),
                        )
                        aiv_out_view = tla.tile_view(
                            aiv_out_ub,
                            tla.make_shape(cfg.AIV_TILE_M, cfg.l1_tn),
                            tla.make_coord(c0, c0),
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
                        tla.copy(aiv_acc_view, aiv_ws_init)
                        tla.set_flag(aiv_loaded)
                        tla.wait_flag(aiv_loaded)

                        # Pre-init the loop indices reused after the slice loop.
                        aiv_rm = 0
                        aiv_rn = 0
                        aiv_rp = 0
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
                            tla.copy(aiv_temp_view, aiv_ws_tile)
                            tla.set_flag(aiv_loaded)
                            tla.wait_flag(aiv_loaded)
                            with tla.vec.func(mode="simd"):
                                for aiv_rm in tla.range_constexpr(0, cfg.AIV_M_CHUNKS, 1):
                                    for aiv_rn in tla.range_constexpr(
                                        0, cfg.AIV_N_CHUNKS, 1
                                    ):
                                        aiv_acc_chunk = tla.tile_view(
                                            aiv_acc_ub,
                                            tla.make_shape(cfg.AIV_REG_M, cfg.AIV_REG_N),
                                            tla.make_coord(aiv_rm, aiv_rn),
                                        )
                                        aiv_temp_chunk = tla.tile_view(
                                            aiv_temp_ub,
                                            tla.make_shape(cfg.AIV_REG_M, cfg.AIV_REG_N),
                                            tla.make_coord(aiv_rm, aiv_rn),
                                        )
                                        aiv_acc_chunk.store(
                                            tla.add(
                                                aiv_acc_chunk.load(),
                                                aiv_temp_chunk.load(),
                                            )
                                        )
                            tla.set_flag(aiv_vec_to_mte2)
                            tla.wait_flag(aiv_vec_to_mte2)

                        aiv_store_ub = aiv_acc_view
                        if cfg.DTYPE_GM_C != cfg.DTYPE_C:
                            aiv_cast_even = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            with tla.vec.func(mode="simd"):
                                for aiv_rm in tla.range_constexpr(0, cfg.AIV_M_CHUNKS, 1):
                                    for aiv_rp in tla.range_constexpr(
                                        0, cfg.AIV_N_DENSE_CHUNKS, 1
                                    ):
                                        aiv_acc0 = tla.tile_view(
                                            aiv_acc_ub,
                                            tla.make_shape(cfg.AIV_REG_M, cfg.AIV_REG_N),
                                            tla.make_coord(aiv_rm, aiv_rp * 2),
                                        )
                                        aiv_acc1 = tla.tile_view(
                                            aiv_acc_ub,
                                            tla.make_shape(cfg.AIV_REG_M, cfg.AIV_REG_N),
                                            tla.make_coord(aiv_rm, aiv_rp * 2 + 1),
                                        )
                                        aiv_out_dense = tla.tile_view(
                                            aiv_out_ub,
                                            tla.make_shape(
                                                cfg.AIV_REG_M, cfg.AIV_DENSE_N
                                            ),
                                            tla.make_coord(aiv_rm, aiv_rp),
                                        )
                                        aiv_h0 = aiv_acc0.load().to(
                                            cfg.DTYPE_GM_C, aiv_cast_even
                                        )
                                        aiv_h1 = aiv_acc1.load().to(
                                            cfg.DTYPE_GM_C, aiv_cast_even
                                        )
                                        aiv_dense, _aiv_odd = tla.deinterleave(
                                            aiv_h0, aiv_h1
                                        )
                                        aiv_out_dense.store(aiv_dense)
                            aiv_store_ub = aiv_out_view

                        tla.set_flag(aiv_done)
                        tla.wait_flag(aiv_done)
                        tla.copy(aiv_gm_c_tile, aiv_store_ub)
                        tla.pipe_barrier(tla.pipes.ALL)

        tla.pipe_barrier(tla.pipes.ALL)
