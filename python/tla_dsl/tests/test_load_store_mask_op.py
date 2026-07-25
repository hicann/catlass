from __future__ import annotations

from typing import Any

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.core_api import MaskSSA, TlaCoreAPIError
from catlass.params import MaskLoadParams, MaskStoreParams


def _ub_tensor(
    dtype: type[tla.Numeric],
    extent: int,
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


def _gm_tensor(
    dtype: type[tla.Numeric],
    extent: int,
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape = tla.make_shape(extent)
        return tla.Tensor(
            shape,
            dtype,
            addrspace=tla.AddressSpace.gm,
            origin_shape=shape,
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def load_store_mask_roundtrip(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(8), tla.make_coord(0))
    data_tile = tla.tile_view(data, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(tla.add(v, v, mask=loaded), mask=loaded)


@tla.kernel
def load_mask_wrong_elem(mask_ub: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(8), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = mask_tile.load(MaskLoadParams())


@tla.kernel
def load_mask_from_gm(mask_gm: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_gm, tla.make_shape(8), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = mask_tile.load(MaskLoadParams())


@tla.kernel
def load_mask_wrong_bytes(mask_ub: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(3), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = mask_tile.load(MaskLoadParams())


@tla.kernel
def load_mask_outside_vec_func(mask_ub: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(8), tla.make_coord(0))
    with tla.vector():
        _ = mask_tile.load(MaskLoadParams())


def test_load_store_mask_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_store_mask_roundtrip,
        type_args=(
            _ub_tensor(tla.Int8, 8),
            _ub_tensor(tla.Float32, 64),
            _ub_tensor(tla.Float32, 64),
        ),
    )
    assert "tla.load" in mlir
    assert "tla.store" in mlir
    assert "!tla.mask<64>" in mlir
    assert "tla.load_mask" not in mlir
    assert "tla.store_mask" not in mlir
    # Mask load result type carries N; no dtype attr on tla.load.
    assert "tla.load" in mlir and "-> !tla.mask<64>" in mlir


def test_load_mask_rejects_non_byte_elem() -> None:
    mask_ub = _ub_tensor(tla.Float32, 8)
    with pytest.raises(TlaCoreAPIError, match="i8/u8"):
        load_mask_wrong_elem(mask_ub)


def test_load_mask_rejects_gm() -> None:
    mask_gm = _gm_tensor(tla.Int8, 8)
    with pytest.raises(TlaCoreAPIError, match="addrspace ub"):
        load_mask_from_gm(mask_gm)


def test_load_mask_rejects_wrong_byte_count() -> None:
    mask_ub = _ub_tensor(tla.Int8, 3)
    with pytest.raises(TlaCoreAPIError, match="packed bytes"):
        load_mask_wrong_bytes(mask_ub)


def test_load_mask_requires_vec_func() -> None:
    mask_ub = _ub_tensor(tla.Int8, 8)
    with pytest.raises(Exception, match="vec.func"):
        load_mask_outside_vec_func(mask_ub)
