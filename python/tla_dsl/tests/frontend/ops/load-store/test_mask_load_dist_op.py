# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
from catlass.core_api import MaskSSA
from catlass.execution_lowering import UnsupportedExecutionLowering
from catlass.params import MaskLoadDist, MaskLoadParams
from catlass.tla.runtime import make_fake_tensor


def _ub(dtype: type[tla.Numeric], *shape: int) -> tla.Tensor:
    if len(shape) == 1:
        extent = shape[0]
        return make_fake_tensor(
            dtype,
            (extent,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(extent,),
            layout_tag=tla.arch.RowMajor,
        )
    rows, cols = shape
    return make_fake_tensor(
        dtype,
        (rows, cols),
        (cols, 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.RowMajor,
    )


def _mask_load_kernel_1d(load_dist: str, mask_extent: int):
    """Load-only kernel: emit MaskSSA IR without companion where lane matching."""
    params = MaskLoadParams(load_dist=load_dist)

    @tla.kernel
    def _kernel(mask_ub: tla.Tensor) -> None:
        mask_tile = tla.tile_view(
            mask_ub, tla.make_shape(mask_extent), tla.make_coord(0)
        )
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask_reg = mask_tile.load(params)
                assert isinstance(mask_reg, MaskSSA)

    return _kernel


def _mask_select_kernel_1d(load_dist: str, mask_extent: int, data_extent: int):
    params = MaskLoadParams(load_dist=load_dist)

    @tla.kernel
    def _kernel(mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor) -> None:
        mask_tile = tla.tile_view(
            mask_ub, tla.make_shape(mask_extent), tla.make_coord(0)
        )
        data_tile = tla.tile_view(data, tla.make_shape(data_extent), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(data_extent), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask_reg = mask_tile.load(params)
                assert isinstance(mask_reg, MaskSSA)
                src = data_tile.load()
                dst_tile.store(tla.where(mask_reg, src, tla.sub(src, src)))

    return _kernel


def _mask_select_kernel_2d(
    load_dist: str, rows: int, cols: int, data_extent: int
):
    params = MaskLoadParams(load_dist=load_dist)

    @tla.kernel
    def _kernel(mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor) -> None:
        mask_tile = tla.tile_view(
            mask_ub, tla.make_shape(rows, cols), tla.make_coord(0, 0)
        )
        data_tile = tla.tile_view(data, tla.make_shape(data_extent), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(data_extent), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask_reg = mask_tile.load(params)
                assert isinstance(mask_reg, MaskSSA)
                src = data_tile.load()
                dst_tile.store(tla.where(mask_reg, src, tla.sub(src, src)))

    return _kernel


@pytest.mark.parametrize(
    ("load_dist", "mask_dtype", "mask_extent", "mask_n", "expect_attr"),
    (
        (MaskLoadDist.DIST_DS, tla.Int8, 64, 64, "ds"),
        (MaskLoadDist.DIST_DS, tla.Int8, 32, 32, "ds"),
        (MaskLoadDist.DIST_DS, tla.Int8, 128, 128, "ds"),
        (MaskLoadDist.DIST_DS, tla.Int8, 256, 256, "ds"),
        (MaskLoadDist.DIST_DS, tla.UInt8, 64, 64, "ds"),
        (MaskLoadDist.DIST_DS, tla.UInt16, 64, 64, "ds"),
        (MaskLoadDist.DIST_DS, tla.UInt32, 64, 64, "ds"),
        (MaskLoadDist.DIST_DS, tla.UInt64, 64, 64, "ds"),
        (MaskLoadDist.DIST_DS, tla.Float32, 64, 64, "ds"),
        (MaskLoadDist.DIST_US, tla.Int8, 64, 64, "us"),
        (MaskLoadDist.DIST_US, tla.Int8, 32, 32, "us"),
        (MaskLoadDist.DIST_US, tla.UInt8, 64, 64, "us"),
        (MaskLoadDist.DIST_US, tla.UInt32, 64, 64, "us"),
        (MaskLoadDist.DIST_NORM, tla.Float32, 2, 64, None),
    ),
)
def test_mask_load_dist_emits_tlair(
    compiler_tlair: Any,
    load_dist: str,
    mask_dtype: type[tla.Numeric],
    mask_extent: int,
    mask_n: int,
    expect_attr: str | None,
) -> None:
    kernel = _mask_load_kernel_1d(load_dist, mask_extent)
    mlir = compiler_tlair(kernel, type_args=(_ub(mask_dtype, mask_extent),))
    assert "tla.load" in mlir
    assert f"!tla.mask<{mask_n}>" in mlir
    assert f"-> !tla.mask<{mask_n}>" in mlir
    if expect_attr is None:
        assert "#tla.load_dist<us>" not in mlir
        assert "#tla.load_dist<ds>" not in mlir
    else:
        assert f"#tla.load_dist<{expect_attr}>" in mlir


@pytest.mark.parametrize(
    "load_dist",
    (MaskLoadDist.DIST_DS, MaskLoadDist.DIST_US),
)
def test_mask_load_dist_select_with_f32_companion(
    compiler_tlair: Any, load_dist: str
) -> None:
    """VL=64 i8 mask + f32 companion Select (matches e2e scripts)."""
    kernel = _mask_select_kernel_1d(load_dist, 64, 64)
    mlir = compiler_tlair(
        kernel,
        type_args=(
            _ub(tla.Int8, 64),
            _ub(tla.Float32, 64),
            _ub(tla.Float32, 64),
        ),
    )
    assert f"#tla.load_dist<{load_dist}>" in mlir
    assert "tla.where" in mlir
    assert "!tla.mask<64>" in mlir


def test_mask_load_dist_ds_static_2d_tile_product_lanes(compiler_tlair: Any) -> None:
    """DIST_DS with static tile (1,64) prefers product lanes → mask<64>."""
    kernel = _mask_select_kernel_2d(MaskLoadDist.DIST_DS, 1, 64, 64)
    mlir = compiler_tlair(
        kernel,
        type_args=(
            _ub(tla.Int8, 1, 64),
            _ub(tla.Float32, 64),
            _ub(tla.Float32, 64),
        ),
    )
    assert "#tla.load_dist<ds>" in mlir
    assert "!tla.mask<64>" in mlir
    assert "-> !tla.mask<64>" in mlir


def test_mask_load_dist_us_static_2d_tile_product_lanes(compiler_tlair: Any) -> None:
    """DIST_US shares product-lane inference with DIST_DS."""
    kernel = _mask_select_kernel_2d(MaskLoadDist.DIST_US, 1, 64, 64)
    mlir = compiler_tlair(
        kernel,
        type_args=(
            _ub(tla.Int8, 1, 64),
            _ub(tla.Float32, 64),
            _ub(tla.Float32, 64),
        ),
    )
    assert "#tla.load_dist<us>" in mlir
    assert "!tla.mask<64>" in mlir


def test_mask_load_dist_rejects_unknown_mode() -> None:
    params = MaskLoadParams(load_dist="not_a_mask_dist")

    @tla.kernel
    def kernel(mask_ub: tla.Tensor) -> None:
        mask_tile = tla.tile_view(mask_ub, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                _ = mask_tile.load(params)

    with pytest.raises(
        (NotImplementedError, UnsupportedExecutionLowering),
        match="unsupported MaskLoadDist",
    ):
        kernel.dump_mlir(type_args=(_ub(tla.Int8, 64),))


def test_mask_load_dist_us_accepts_f32_ub(compiler_tlair: Any) -> None:
    """DIST_US/DS align AscendC b32: f32 UB is address typing only."""
    kernel = _mask_load_kernel_1d(MaskLoadDist.DIST_US, 64)
    mlir = compiler_tlair(kernel, type_args=(_ub(tla.Float32, 64),))
    assert "#tla.load_dist<us>" in mlir
    assert "!tla.mask<64>" in mlir


def test_mask_load_dist_ds_accepts_u32_ub(compiler_tlair: Any) -> None:
    """Matches XA Softmax LoadAlign<uint32_t, MaskDist::DIST_DS> typing."""
    kernel = _mask_load_kernel_1d(MaskLoadDist.DIST_DS, 64)
    mlir = compiler_tlair(kernel, type_args=(_ub(tla.UInt32, 64),))
    assert "#tla.load_dist<ds>" in mlir
    assert "!tla.mask<64>" in mlir
