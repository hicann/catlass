from __future__ import annotations

from typing import Any, Callable

import pytest

import catlass as tla
import catlass.runtime as runtime_mod

from catlass.execution_lowering import TlaLoweringError


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Int32,
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

def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Int32,
    shape: tuple[int, ...] = (64,),
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape = tla.make_shape(*shape)
        return tla.Tensor(
            shape,
            dtype,
            addrspace=tla.AddressSpace.gm,
            origin_shape=shape,
            layout_tag=tla.arch.RowMajor,
        )

def _l0c_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    m: int = 32,
    n: int = 64
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        shape = tla.make_shape(m, n)
        return tla.Tensor(
            shape,
            dtype,
            addrspace=tla.AddressSpace.l0c,
            origin_shape=shape,
            layout_tag=tla.arch.RowMajor,
        )

@tla.kernel
def atomic_add_l0c2gm_fake_atomic_mode(l0c_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Test atomic add mode, data path from L0C to GM"""

    _m, _n = 32, 32
    l0c_tile = tla.tile_view(l0c_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    gm_tile = tla.tile_view(gm_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c_tile,
            tla.params.CopyL0C2DstParams(
                unit_flag=0b11,
                atomic_mode=2
            )
        )

@tla.kernel
def atomic_add_ub2gm(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Test atomic add mode, data path from UB to GM"""

    _vec_len = 64
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(x_gm, x_ub, tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD))

@tla.kernel
def atomic_add_l0c2gm(l0c_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Test atomic add mode, data path from L0C to GM"""

    _m, _n = 32, 32
    l0c_tile = tla.tile_view(l0c_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    gm_tile = tla.tile_view(gm_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c_tile,
            tla.params.CopyL0C2DstParams(
                unit_flag=0b11,
                atomic_mode=tla.params.AtomicMode.ADD
            )
        )

@tla.kernel
def atomic_none_ub2gm(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Test common data copy, without atomic operation"""

    _vec_len = 64
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(x_gm, x_ub)

@tla.kernel
def gm2ub_atomic_false(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    _vec_len = 64
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(x_ub, x_gm, tla.params.CopyL0C2DstParams(atomic_mode=tla.params.AtomicMode.ADD))

def test_atomic_add_ub2gm(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(atomic_add_ub2gm, type_args=(_ub_tensor(), _gm_tensor()))

    assert "tla.copy" in mlir
    assert "atomic_mode = #tla.atomic_mode<add>" in mlir

def test_atomic_add_l0c2gm(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        atomic_add_l0c2gm,
        type_args=(_l0c_tensor(m=32, n=32), _gm_tensor(dtype=tla.Float32, shape=(32, 32))),
    )

    assert "tla.copy" in mlir
    assert "atomic_mode = #tla.atomic_mode<add>" in mlir

def test_atomic_none_ub2gm(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(atomic_none_ub2gm, type_args=(_ub_tensor(), _gm_tensor()))

    assert "tla.copy" in mlir
    assert "atomic_mode" not in mlir

def test_atomic_add_gm2ub_rejected() -> None:
    """Atomic add with dst=UB should fail because atomic add requires dst=GM."""

    with pytest.raises(TlaLoweringError, match="the dst location should only be GM"):
        gm2ub_atomic_false.dump_mlir(type_args=(_gm_tensor(), _ub_tensor()))

def test_atomic_illega_f64_dtype() -> None:
    with pytest.raises(TlaLoweringError):
        atomic_add_ub2gm.dump_mlir(type_args=(_ub_tensor(dtype=tla.Int64), _gm_tensor(dtype=tla.Int64)))