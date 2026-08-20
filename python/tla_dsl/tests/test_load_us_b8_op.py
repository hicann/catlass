from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError
from catlass.params import LoadDist, NormalLoadParams


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def load_us_b8_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_US_B8)))


def test_load_us_b8_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_us_b8_kernel,
        type_args=(
            _ub_tensor(dtype=tla.Int8, extent=128),
            _ub_tensor(dtype=tla.Int8, extent=256),
        ),
    )

    assert "#tla.load_dist<us_b8>" in mlir
    load_lines = [line for line in mlir.splitlines() if "tla.load" in line]
    assert len(load_lines) == 1
    # Up-sample fills a full VL b8 register (i8 -> 256 lanes).
    assert "!tla.vector<256xi8>" in load_lines[0]


@tla.kernel
def load_us_b8_wrong_dtype_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_US_B8)))


@pytest.mark.parametrize("dtype", [tla.Float32, tla.Float16, tla.Int32])
def test_load_us_b8_rejects_non_b8(
    compiler_tlair: Any, dtype: type[tla.Numeric]
) -> None:
    # DIST_US_B8 is a b8-only up-sample mode; non-b8 element types must be
    # rejected at the frontend before any IR is emitted.
    with pytest.raises(TlaLoweringError, match="b8|US_B8|1-byte"):
        compiler_tlair(
            load_us_b8_wrong_dtype_kernel,
            type_args=(
                _ub_tensor(dtype=dtype, extent=64),
                _ub_tensor(dtype=dtype, extent=64),
            ),
        )
