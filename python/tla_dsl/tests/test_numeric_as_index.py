"""Numeric values used as tensor indices and make_ptr addresses."""

from __future__ import annotations

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _gm_tensor_1d(length: int, *, dtype: type) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(length),
            dtype,
            addrspace=tla.AddressSpace.gm,
            origin_shape=tla.make_shape(length),
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def _numeric_index_store_kernel(index_buf: tla.Tensor, out: tla.Tensor) -> None:
    index = index_buf[0]
    out[index] = 7


@tla.kernel
def _numeric_make_ptr_kernel(address_buf: tla.Tensor) -> None:
    address = address_buf[0]
    _ = tla.make_ptr(tla.Int32, address, mem_space=tla.AddressSpace.ub)


def test_numeric_as_store_index_emits_index_cast() -> None:
    index_buf = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _numeric_index_store_kernel.dump_mlir(type_args=(index_buf, out))
    assert "tla.scalar_load" in mlir
    assert "arith.index_cast" in mlir
    assert "tla.scalar_store" in mlir
    assert "7" in mlir or "value = 7" in mlir


def test_uint_numeric_as_store_index_rejected() -> None:
    """Index path is Int*/Bool only; UInt* must be cast explicitly first."""
    index_buf = _gm_tensor_1d(8, dtype=tla.UInt32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match=r"signed integer Numeric index|UInt32"):
        _numeric_index_store_kernel.dump_mlir(type_args=(index_buf, out))


def test_numeric_as_make_ptr_address() -> None:
    address_buf = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _numeric_make_ptr_kernel.dump_mlir(type_args=(address_buf,))
    assert "tla.scalar_load" in mlir
    assert "tla.inttoptr" in mlir
