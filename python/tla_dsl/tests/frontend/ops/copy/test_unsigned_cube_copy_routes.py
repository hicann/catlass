"""Frontend acceptance for unsigned cube / vector staging copy routes."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _gm_row(dtype, rows=128, cols=128):
    return make_fake_tensor(
        dtype,
        (rows, cols),
        (cols, 1),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.RowMajor,
    )


def _l1_zn(dtype, rows=128, cols=128):
    return make_fake_tensor(
        dtype,
        (rows, cols),
        (cols, 1),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.zN,
    )


def _ub_row(dtype, rows=128, cols=128):
    return make_fake_tensor(
        dtype,
        (rows, cols),
        (cols, 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def gm_to_l1_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def gm_to_ub_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)


@pytest.mark.parametrize("dtype", [tla.UInt16, tla.UInt32])
def test_gm_to_l1_accepts_uint16_uint32(dtype) -> None:
    src = _gm_row(dtype)
    dst = _l1_zn(dtype)
    mlir = gm_to_l1_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir


@pytest.mark.parametrize("dtype", [tla.UInt8, tla.UInt16, tla.UInt32])
def test_gm_to_ub_accepts_unsigned(dtype) -> None:
    src = _gm_row(dtype)
    dst = _ub_row(dtype)
    mlir = gm_to_ub_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.copy" in mlir
