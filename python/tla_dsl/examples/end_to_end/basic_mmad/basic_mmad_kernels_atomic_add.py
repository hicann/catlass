"""MMAD variant that accumulates independent K-tile products atomically in GM.

Each Cube MMAD starts from a cleared L0C accumulator and is copied to GM
immediately.  GM C is intentionally restricted to Float32 until the atomic
type is carried explicitly by the lowering pass rather than inferred from the
destination tensor type.
"""

from __future__ import annotations

from typing import Any

import catlass as tla
from catlass.types import dtype_size_bytes


DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float32

ENABLE_UNIT_FLAG = True

def _elem_bytes(num_elems: int, dtype_tla: Any) -> int:
    return num_elems * dtype_size_bytes(dtype_tla.dtype)


m = 333
n = 444
k = 555
l1_tm = 256
l1_tn = 256
l1_tk = 128
l0_tm = 256
l0_tn = 256
l0_tk = 32


@tla.kernel
def basic_mmad_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    c0 = 0
    c1 = 1

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
    partial_mmad_done = tla.flag(
        "partial_mmad_done", tla.arch.CUBE, tla.arch.FIX
    )

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
    total_blocks = grid_m * grid_n

    with tla.cube():
        tla.set_flag(l1a0_copy_start)
        tla.set_flag(l1a1_copy_start)
        tla.set_flag(l1b0_copy_start)
        tla.set_flag(l1b1_copy_start)
        tla.set_flag(l0a0_copy_start)
        tla.set_flag(l0a1_copy_start)
        tla.set_flag(l0b0_copy_start)
        tla.set_flag(l0b1_copy_start)

        l1_buf_idx = c0
        l0_buf_idx = c0

        block_range = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_dim())
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
            gm_a_by_core = tla.tile_view(
                mem_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                mem_b, tla.make_shape(k, l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                mem_c,
                tla.make_shape(l1_tm, l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk - 1) // l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)
            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            for k_l1 in k_l1_range:
                is_first_l1_tile = k_l1 == 0

                gm_a_l1 = tla.tile_view(
                    gm_a_by_core,
                    tla.make_shape(l1_tm, l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_b_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(l1_tk, l1_tn),
                    tla.make_coord(k_l1, c0),
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if l1_buf_idx == c0 else l1a1_ptr, gm_a_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if l1_buf_idx == c0 else l1b1_ptr, gm_b_l1
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
                        l1_a,
                        tla.make_shape(l0_tm, l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_b_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(l0_tk, l0_tn),
                        tla.make_coord(k_l0, c0),
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if l0_buf_idx == c0 else l0a1_ptr, l1_a_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if l0_buf_idx == c0 else l0b1_ptr, l1_b_l0
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

                    unit_flag=0b00
                    if (k_l0 == k_l0_count - 1):
                        unit_flag=0b11
                    else:
                        unit_flag=0b10

                    # This MMAD produces one independent K-tile partial.  It
                    # must not consume the previous contents of L0C.
                    init_c = (k_l0 == 0)
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_copy_start)
                        tla.set_flag(l0b0_copy_start)
                    else:
                        tla.set_flag(l0a1_copy_start)
                        tla.set_flag(l0b1_copy_start)
                    l0_buf_idx = c1 - l0_buf_idx

                # FIX consumes this partial immediately.  The FIX barrier
                # keeps L0C reuse and the atomic control state behind the
                if not is_first_l1_tile:
                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(
                            unit_flag=0b11,
                            atomic_mode=tla.params.AtomicMode.ADD
                        )
                    )
                else:
                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11)
                    )

                l1_buf_idx = c1 - l1_buf_idx

        tla.wait_flag(l1a0_copy_start)
        tla.wait_flag(l1a1_copy_start)
        tla.wait_flag(l1b0_copy_start)
        tla.wait_flag(l1b1_copy_start)
        tla.wait_flag(l0a0_copy_start)
        tla.wait_flag(l0a1_copy_start)
        tla.wait_flag(l0b0_copy_start)
        tla.wait_flag(l0b1_copy_start)
