from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.params import (
    NormalStoreParams, 
    StoreDist,
    )

# DIST mode now supports:
# - StoreDist.DIST_NORM: Normal mode, move out a VL-width from reg tensor to ub tensor with datatype b8/b16/b32
# - StoreDist.DIST_PACK_B32: Pack mode (b32), the lower half bits of valid elements in src are stored in dst according to the mask.
# - StoreDist.DIST_PACK_B16: Pack mode (b16).

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
def store_dist(src: tla.Tensor, dst: tla.Tensor, dist: tla.Constexpr[str]) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            dst_tile.store(src_reg, NormalStoreParams(store_dist=dist))


@pytest.mark.parametrize(
    ("dist", "dtype"),
    (
        (StoreDist.DIST_NORM, tla.Float32),
        (StoreDist.DIST_PACK_B32, tla.Float32),
        (StoreDist.DIST_PACK_B32, tla.Int32),
        (StoreDist.DIST_PACK_B16, tla.Float16),
        (StoreDist.DIST_PACK_B16, tla.Int16),
    )
)
def test_store_dist_emits_tlair(compiler_tlair: Any, dist: StoreDist, dtype: type[tla.Numeric]):
    mlir = compiler_tlair(
        store_dist,
        type_args=(
            _ub_tensor(dtype),
            _ub_tensor(dtype),
            dist
        ),
    )

    assert "tla.store" in mlir
    if dist != StoreDist.DIST_NORM:
        assert f"#tla.store_dist<{dist}>" in mlir
