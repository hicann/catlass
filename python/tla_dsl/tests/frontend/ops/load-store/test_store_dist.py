from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass.params import NormalStoreParams, StoreDist
from catlass.tla.runtime import make_fake_tensor


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
    ("dist", "source_dtype", "dest_dtype"),
    (
        (StoreDist.DIST_NORM, tla.Float32, tla.Float32),
        (StoreDist.DIST_PACK_B32, tla.Float32, tla.Float16),
        (StoreDist.DIST_PACK_B32, tla.Int32, tla.Int16),
        # A packed 16-bit vector has two source predicate lanes per compacted
        # 16-bit destination element.
        (StoreDist.DIST_PACK_B32, tla.Float16, tla.Float16),
        (StoreDist.DIST_PACK_B16, tla.Int16, tla.Int8),
    ),
)
def test_store_dist_emits_tlair(
    compiler_tlair: Any,
    dist: StoreDist,
    source_dtype: type[tla.Numeric],
    dest_dtype: type[tla.Numeric],
) -> None:
    mlir = compiler_tlair(
        store_dist,
        type_args=(
            _ub_tensor(source_dtype),
            _ub_tensor(dest_dtype),
            dist,
        ),
    )

    assert "tla.store" in mlir
    if dist != StoreDist.DIST_NORM:
        assert f"#tla.store_dist<{dist}>" in mlir
