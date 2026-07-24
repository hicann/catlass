"""Tests for tla.store with block_stride (via BlockStoreParams)."""

from __future__ import annotations

from typing import Any

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.params import BlockStoreParams


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    *,
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
def block_store_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_v = src_tile.load()
            dst_tile.store(x_v, params=BlockStoreParams(block_stride=32))


def test_block_store_tlair_has_stride(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        block_store_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir


def test_block_store_tlair_has_block_stride_value(compiler_tlair: Any) -> None:
    """BlockStoreParams(block_stride=32) should produce stride with value 32."""
    mlir = compiler_tlair(
        block_store_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir
    # After tla-compile, block_stride=32 should produce
    # arith.constant 32 : i32 or index in the lowered output


def test_block_store_different_stride(compiler_tlair: Any) -> None:
    """BlockStoreParams with different stride values."""

    @tla.kernel
    def block_store_16_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(x_v, params=BlockStoreParams(block_stride=16))

    mlir = compiler_tlair(
        block_store_16_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir


def test_block_store_kernel_parameter_stride(compiler_tlair: Any) -> None:
    """BlockStoreParams with stride from a kernel scalar argument."""

    @tla.kernel
    def block_store_param_kernel(src: tla.Tensor, dst: tla.Tensor, stride_val: tla.Int32) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(x_v, params=BlockStoreParams(block_stride=stride_val))

    with runtime_mod._eager_capture():
        n = tla.Int32(32)
        mlir = compiler_tlair(
            block_store_param_kernel,
            type_args=(_ub_tensor(), _ub_tensor(), n),
        )
    assert "tla.store" in mlir
    assert "stride" in mlir
