from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.params import NormalStoreParams, StoreDist


def _ub_tensor(
    dtype: type[tla.Numeric],
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
def store_first_element_kernel(src: tla.Tensor, dst: tla.Tensor, dist: tla.Constexpr[str]) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            dst_tile.store(src_reg, NormalStoreParams(store_dist=dist))


@pytest.mark.parametrize(
    ("dist", "dtype"),
    (
        (StoreDist.DIST_FIRST_ELEMENT_B32, tla.Float32),
        (StoreDist.DIST_FIRST_ELEMENT_B32, tla.Int32),
        (StoreDist.DIST_FIRST_ELEMENT_B16, tla.Float16),
        (StoreDist.DIST_FIRST_ELEMENT_B16, tla.Int16),
        (StoreDist.DIST_FIRST_ELEMENT_B8, tla.Int8),
    ),
)
def test_store_first_element_emits_tlair(
    compiler_tlair: Any, dist: StoreDist, dtype: type[tla.Numeric]
) -> None:
    mlir = compiler_tlair(
        store_first_element_kernel,
        type_args=(
            _ub_tensor(dtype),
            _ub_tensor(dtype),
            dist,
        ),
    )

    assert "tla.store" in mlir
    assert f"#tla.store_dist<{dist}>" in mlir


@tla.kernel
def store_first_element_wrong_dtype_kernel(
    src: tla.Tensor, dst: tla.Tensor, dist: tla.Constexpr[str]
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            dst_tile.store(src_reg, NormalStoreParams(store_dist=dist))


@pytest.mark.parametrize(
    ("dist", "dtype"),
    (
        (StoreDist.DIST_FIRST_ELEMENT_B32, tla.Float16),
        (StoreDist.DIST_FIRST_ELEMENT_B16, tla.Int32),
        (StoreDist.DIST_FIRST_ELEMENT_B8, tla.Float32),
    ),
)
def test_store_first_element_rejects_width_mismatch(
    compiler_tlair: Any, dist: StoreDist, dtype: type[tla.Numeric]
) -> None:
    # The _b8/_b16/_b32 suffix fixes the stored element width; a mismatched
    # source/dest element type would silently truncate or widen the stored
    # bytes and must be rejected at the frontend.
    with pytest.raises(TlaLoweringError, match="FIRST_ELEMENT|byte"):
        compiler_tlair(
            store_first_element_wrong_dtype_kernel,
            type_args=(
                _ub_tensor(dtype),
                _ub_tensor(dtype),
                dist,
            ),
        )
