"""``tla.range`` bounds accept signed integer Numeric (e.g. Int32 scalar load)."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod


def _gm_tensor_1d(length: int, *, dtype: type) -> tla.Tensor:
    return make_fake_tensor(
               dtype,
               (length,),
               (1,),
               origin_shape=(length,),
               layout_tag=tla.arch.RowMajor,
           )


@tla.kernel
def _numeric_range_end_kernel(limit_buf: tla.Tensor, out: tla.Tensor) -> None:
    # end is Int32 Numeric from scalar_load; scf.for is i32 (no index_cast for bounds).
    end = limit_buf[0]
    for i in tla.range(0, end, 1):
        out[i] = 1


@tla.kernel
def _numeric_range_start_end_kernel(bounds: tla.Tensor, out: tla.Tensor) -> None:
    start = bounds[0]
    end = bounds[1]
    for i in tla.range(start, end, 1):
        out[i] = 1


def test_numeric_range_end_emits_scf_for_i32() -> None:
    limit_buf = _gm_tensor_1d(1, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _numeric_range_end_kernel.dump_mlir(type_args=(limit_buf, out))
    assert "tla.scalar_load" in mlir
    assert "scf.for" in mlir
    assert "i32, i32, i32" in mlir.replace(" ", "") or "%arg" in mlir
    assert "tla.scalar_store" in mlir


def test_numeric_range_start_end_emits_scf_for() -> None:
    bounds = _gm_tensor_1d(2, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _numeric_range_start_end_kernel.dump_mlir(type_args=(bounds, out))
    assert "tla.scalar_load" in mlir
    assert "scf.for" in mlir


def test_uint_numeric_range_end_rejected() -> None:
    limit_buf = _gm_tensor_1d(1, dtype=tla.UInt32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match=r"signed integer Numeric|UInt32"):
        _numeric_range_end_kernel.dump_mlir(type_args=(limit_buf, out))
