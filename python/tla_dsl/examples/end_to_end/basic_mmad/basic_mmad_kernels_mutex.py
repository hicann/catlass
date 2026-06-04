"""L1 staging variants for ``basic_matmul`` (real ``.py`` source for lowering transforms).

Kernels defined via ``exec`` are not rewritten by ``maybe_transform_for_lowering``, so
``for x in tla.range(...)`` fails at compile time. These four entry points duplicate the
same body with literal ``tla.arch.zN`` / ``tla.arch.nZ`` for L1 ``make_tensor_like`` only.
"""

from __future__ import annotations

from typing import Any

import catlass as tla
from catlass.types import dtype_size_bytes

# GM / tile element types for this kernel. ``basic_matmul.py`` mutates DTYPE_A/B/GM_C before
# ``tla.compile`` so host type_args and on-device recast_ptr agree with ``tla.mmad``.
# Cube MMAD always accumulates in fp32 on L0C: ``DTYPE_C`` is always Float32 for ``l0c_ptr``
# and for ``make_tensor_like(..., dst_dtype=DTYPE_C)``. ``DTYPE_GM_C`` is the GM C matrix
# element type (f32, or narrowed f16/bf16); ``tla.copy(gm_c, l0_c)`` lowers to
# copy_cc_to_gm_row_major_float | _half | _bf16 accordingly.
DTYPE_A = tla.Float16
DTYPE_B = tla.Float16
DTYPE_C = tla.Float32
DTYPE_GM_C = tla.Float32


def _elem_bytes(num_elems: int, dtype_tla: Any) -> int:
    return num_elems * dtype_size_bytes(dtype_tla.dtype)


# Problem size + tiling for this demo. ``basic_matmul.py`` imports ``m,n,k`` from here
# so host tensors and kernels stay consistent. MNK are logical GEMM extents.
m = 333
n = 444
k = 555
l1_tm = 256
l1_tn = 256
l1_tk = 256
l0_tm = 256
l0_tn = 256
l0_tk = 64


@tla.kernel
def basic_mmad_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    c0 = 0
    c1 = 1

    mutex_l1a0  = tla.mutex(resource="l1a0", id=0)
    mutex_l1a1  = tla.mutex(resource="l1a1", id=1)
    mutex_l1b0  = tla.mutex(resource="l1b0", id=2)
    mutex_l1b1  = tla.mutex(resource="l1b1", id=3)

    mutex_l0a0  = tla.mutex(resource="l0a0", id=4)
    mutex_l0a1  = tla.mutex(resource="l0a1", id=5)
    mutex_l0b0  = tla.mutex(resource="l0b0", id=6)
    mutex_l0b1  = tla.mutex(resource="l0b1", id=7)

    mutex_l0c  = tla.mutex(resource="l0c", id=8)

    mem_allocator = tla.utils.LocalmemAllocator()
    l1a0_ptr = mem_allocator.allocate(_elem_bytes(l1_tm * l1_tk, DTYPE_A), 512, tla.AddressSpace.l1)
    l1a0_ptr = tla.recast_ptr(l1a0_ptr, dtype=DTYPE_A)
    l1a1_ptr = mem_allocator.allocate(_elem_bytes(l1_tm * l1_tk, DTYPE_A), 512, tla.AddressSpace.l1)
    l1a1_ptr = tla.recast_ptr(l1a1_ptr, dtype=DTYPE_A)
    l1b0_ptr = mem_allocator.allocate(_elem_bytes(l1_tk * l1_tn, DTYPE_B), 512, tla.AddressSpace.l1)
    l1b0_ptr = tla.recast_ptr(l1b0_ptr, dtype=DTYPE_B)
    l1b1_ptr = mem_allocator.allocate(_elem_bytes(l1_tk * l1_tn, DTYPE_B), 512, tla.AddressSpace.l1)
    l1b1_ptr = tla.recast_ptr(l1b1_ptr, dtype=DTYPE_B)

    l0a0_ptr = mem_allocator.allocate(_elem_bytes(l0_tm * l0_tk, DTYPE_A), 512, tla.AddressSpace.l0a)
    l0a0_ptr = tla.recast_ptr(l0a0_ptr, dtype=DTYPE_A)
    l0a1_ptr = mem_allocator.allocate(_elem_bytes(l0_tm * l0_tk, DTYPE_A), 512, tla.AddressSpace.l0a)
    l0a1_ptr = tla.recast_ptr(l0a1_ptr, dtype=DTYPE_A)
    l0b0_ptr = mem_allocator.allocate(_elem_bytes(l0_tk * l0_tn, DTYPE_B), 512, tla.AddressSpace.l0b)
    l0b0_ptr = tla.recast_ptr(l0b0_ptr, dtype=DTYPE_B)
    l0b1_ptr = mem_allocator.allocate(_elem_bytes(l0_tk * l0_tn, DTYPE_B), 512, tla.AddressSpace.l0b)
    l0b1_ptr = tla.recast_ptr(l0b1_ptr, dtype=DTYPE_B)

    l0c_ptr = mem_allocator.allocate(_elem_bytes(l0_tm * l0_tn, DTYPE_C), 512, tla.AddressSpace.l0c)
    l0c_ptr = tla.recast_ptr(l0c_ptr, dtype=DTYPE_C)

    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    total_blocks = grid_m * grid_n


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
            mem_c, tla.make_shape(l1_tm, l1_tn), tla.make_coord(block_row, block_col)
        )

        k_block = gm_a_by_core.origin_shape[1]
        k_l1_count = (k_block + l1_tk - 1) // l1_tk
        k_l1_range = tla.range(c0, k_l1_count, c1)

        l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core, dst_dtype=DTYPE_C)

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
            
            mutex_l1a = mutex_l1a0 if (l1_buf_idx == c0) else mutex_l1a1
            mutex_l1a.lock(pipe=tla.arch.MTE2)
            tla.copy(l1_a, gm_a_l1)
            mutex_l1a.unlock(pipe=tla.arch.MTE2)

            mutex_l1b = mutex_l1b0 if (l1_buf_idx == c0) else mutex_l1b1
            mutex_l1b.lock(pipe=tla.arch.MTE2)
            tla.copy(l1_b, gm_b_l1)
            mutex_l1b.unlock(pipe=tla.arch.MTE2)

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

                mutex_l0a = mutex_l0a0 if (l0_buf_idx == c0) else mutex_l0a1
                mutex_l1a.lock(pipe=tla.arch.MTE1)
                mutex_l0a.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_a, l1_a_l0)
                mutex_l0a.unlock(pipe=tla.arch.MTE1)
                mutex_l1a.unlock(pipe=tla.arch.MTE1)


                mutex_l0b = mutex_l0b0 if (l0_buf_idx == c0) else mutex_l0b1
                mutex_l1b.lock(pipe=tla.arch.MTE1)
                mutex_l0b.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_b, l1_b_l0)
                mutex_l0b.unlock(pipe=tla.arch.MTE1)
                mutex_l1b.unlock(pipe=tla.arch.MTE1)

                mutex_l0a.lock(pipe=tla.arch.CUBE)
                mutex_l0b.lock(pipe=tla.arch.CUBE)
                mutex_l0c.lock(pipe=tla.arch.CUBE)
                tla.mmad(
                    l0_c, l0_a, l0_b, init_c=True if k_l1 == 0 and k_l0 == 0 else False
                )
                mutex_l0c.unlock(pipe=tla.arch.CUBE)
                mutex_l0b.unlock(pipe=tla.arch.CUBE)
                mutex_l0a.unlock(pipe=tla.arch.CUBE)

                l0_buf_idx = c1 - l0_buf_idx
            l1_buf_idx = c1 - l1_buf_idx

        mutex_l0c.lock(pipe=tla.arch.FIX)
        tla.copy(gm_c_by_core, l0_c)
        mutex_l0c.unlock(pipe=tla.arch.FIX)



