from __future__ import annotations

from typing import Any

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _ub_tensor(dtype: type[tla.Numeric] = tla.Float32) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(64),
            dtype,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(64),
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def gather_vector_kernel(src: tla.Tensor, idx_mem: tla.Tensor, dst: tla.Tensor) -> None:
    """Kernel exercising tla.gather: per-lane indexed load from a UB tile."""
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    idx_tile = tla.tile_view(idx_mem, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            indices = idx_tile.load()
            gathered = tla.gather(src_tile, indices)
            dst_tile.store(gathered)


def test_gather_op_emits_mlir() -> None:
    """tla.gather produces expected MLIR with proper operands and region nesting."""
    src = _ub_tensor()
    idx_mem = _ub_tensor(tla.Int32)
    dst = _ub_tensor()
    mlir = gather_vector_kernel.dump_mlir(type_args=(src, idx_mem, dst))

    assert "tla.gather" in mlir
    assert "tla.store" in mlir
    assert "tla.vector" in mlir
    assert mlir.index("tla.vector") < mlir.index("tla.gather")
    gather_lines = [line for line in mlir.splitlines() if "tla.gather" in line]
    assert len(gather_lines) == 1


def test_gather_rejects_non_ub_src() -> None:
    """tla.gather raises when the source tile is not in ub address space."""

    @tla.kernel
    def bad_src(l0c_mem: tla.Tensor, idx_mem: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(l0c_mem, tla.make_shape(64), tla.make_coord(0))
        idx_tile = tla.tile_view(idx_mem, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                indices = idx_tile.load()
                dst_tile.store(tla.gather(src_tile, indices))

    with runtime_mod._eager_capture():
        l0c_mem = tla.Tensor(
            tla.make_shape(64),
            tla.Float32,
            addrspace=tla.AddressSpace.l0c,
            origin_shape=tla.make_shape(64),
            layout_tag=tla.arch.RowMajor,
        )
        idx_mem = _ub_tensor(tla.Int32)
        dst = _ub_tensor()

    with pytest.raises(
        tla.TlaCoreAPIError, match="invalid argument 'x'.*expected addrspace ub"
    ):
        bad_src.dump_mlir(type_args=(l0c_mem, idx_mem, dst))
