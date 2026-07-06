from __future__ import annotations

import catlass as tla
import catlass.runtime as runtime_mod
from catlass._mlir_bindings import tla_ops_gen


@tla.kernel
def _ops_surface_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    shape = tla.make_shape(1, 16)
    coord = tla.make_coord(0, 0)
    src_tile = tla.tile_view(src, shape, coord)
    dst_tile = tla.tile_view(dst, shape, coord)
    allocator = tla.utils.LocalmemAllocator()
    local_ptr = allocator.allocate(1 * 16 * 2, 32, tla.AddressSpace.l1)
    _ = local_ptr

    tla.copy(src_tile, dst_tile)

    with tla.vector():
        ready = tla.flag("ready")
        tla.set_flag(ready)
        tla.wait_flag(ready)
        cross = tla.cross_flag("x", tla.pipes.MTE3, tla.pipes.SCALAR)
        tla.cross_core_set_flag(cross)
        tla.cross_core_wait_flag(cross)
        tla.pipe_barrier(tla.pipes.MTE3)
        mutex_ping = tla.mutex(resource="l0a_ping", id=0)
        mutex_pong = tla.mutex(resource="l0a_pong", id=1)
        block_range = tla.range(0, 10, 1)
        for idx in block_range:
            mutex = mutex_ping if idx % 2 == 0 else mutex_pong
            mutex.lock(pipe=tla.arch.MTE2)
            mutex.unlock(pipe=tla.arch.MTE2)


def test_generated_binding_symbols_exist_for_wrapped_ops() -> None:
    required = (
        "tile_view",
        "copy",
        "flag",
        "cross_flag",
        "cross_core_set_flag",
        "cross_core_wait_flag",
        "set_flag",
        "wait_flag",
        "pipe_barrier",
        "mutex",
        "mutex_lock",
        "mutex_unlock",
        "make_shape",
        "make_coord",
        "make_stride",
        "make_layout",
        "mmad",
        "arch_block_idx",
        "arch_block_dim",
        "recast_ptr",
        "hivm_memref_as_ptr",
        "adds",
        "subs",
        "muls",
        "maxs",
        "mins",
        "divs",
        "neg",
    )
    for symbol in required:
        assert hasattr(tla_ops_gen, symbol)


def test_public_api_exports_representative_helpers() -> None:
    assert callable(tla.tile_view)
    assert callable(tla.copy)
    assert callable(tla.flag)
    assert callable(tla.cross_flag)
    assert callable(tla.mutex)
    assert callable(tla.mutex_guard)
    assert callable(tla.make_tensor_like)
    assert callable(tla.range_constexpr)
    assert callable(tla.arch.block_idx)
    assert callable(tla.utils.LocalmemAllocator)
    assert callable(tla.recast_ptr)
    assert tla.arch.FIX is tla.pipes.FIX
    assert tla.pipes.ALL is not None


def test_ops_surface_kernel_lowers_key_op_families() -> None:
    with runtime_mod._eager_capture():
        src = tla.Tensor(
            tla.make_shape(1, 16), tla.Float16, origin_shape=tla.make_shape(1, 16)
        )
        dst = tla.Tensor(
            tla.make_shape(1, 16), tla.Float16, origin_shape=tla.make_shape(1, 16)
        )
    mlir = _ops_surface_kernel.dump_mlir(type_args=(src, dst))
    for op_name in (
        "tla.alloc_ptr",
        "tla.tile_view",
        "tla.copy",
        "tla.flag",
        "tla.set_flag",
        "tla.wait_flag",
        "tla.cross_flag",
        "tla.cross_core_set_flag",
        "tla.cross_core_wait_flag",
        "tla.pipe_barrier",
        "tla.mutex",
        "tla.mutex_lock",
        "tla.mutex_unlock",
        "tla.make_shape",
        "tla.make_coord",
    ):
        assert op_name in mlir
