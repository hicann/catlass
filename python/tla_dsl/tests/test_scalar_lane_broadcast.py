from __future__ import annotations

from typing import Any

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape = tla.make_shape(extent)
        return tla.Tensor(
            shape,
            dtype,
            addrspace=tla.AddressSpace.ub,
            origin_shape=shape,
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def scalar_lane_full(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        reg = src_tile.load()
        out = reg + tla.full(0.0, tla.Float32)
        dst_tile.store(out)


@tla.kernel
def full_int_literal(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        dst_tile.store(src_tile.load() + tla.full(1, tla.Int32))


@tla.kernel
def full_on_lhs(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        dst_tile.store(tla.full(0.0, tla.Float32) + src_tile.load())


@tla.kernel
def full_index_ssa_is_rejected(src: tla.Tensor, dst: tla.Tensor) -> None:
    idx = tla.arch.block_idx()
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        dst_tile.store(src_tile.load() + tla.full(idx, tla.Float32))


@tla.kernel
def full_string_dtype_is_rejected(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        dst_tile.store(src_tile.load() + tla.full(0.0, "f32"))


def test_scalar_lane_full_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(scalar_lane_full, type_args=(_ub_tensor(), _ub_tensor()))

    assert "tla.full" in mlir
    assert "tla.add" in mlir


def test_full_int_literal_uses_vector_dtype(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        full_int_literal,
        type_args=(_ub_tensor(tla.Int32), _ub_tensor(tla.Int32)),
    )

    assert "value = 1 : i32" in mlir


def test_full_on_lhs_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(full_on_lhs, type_args=(_ub_tensor(), _ub_tensor()))

    assert "tla.full" in mlir
    assert "tla.add" in mlir


def test_full_rejects_index_ssa_operand(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="Python scalar literal"):
        compiler_tlair(
            full_index_ssa_is_rejected,
            type_args=(_ub_tensor(), _ub_tensor()),
        )


def test_full_rejects_invalid_dtype(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="not a str"):
        compiler_tlair(
            full_string_dtype_is_rejected,
            type_args=(_ub_tensor(), _ub_tensor()),
        )
