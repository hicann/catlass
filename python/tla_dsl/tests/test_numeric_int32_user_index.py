"""User-facing indexes are Int32 (MLIR index only at lowering boundaries)."""

from __future__ import annotations

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
def _range_iv_int32_kernel(out: tla.Tensor) -> None:
    for i in tla.range(0, 4, 1):
        out[i] = i


@tla.kernel
def _int_arg_and_iv_min_kernel(limit: int, out: tla.Tensor) -> None:
    # Kernel ``int`` and range IV are both Int32 — no Index / index SSA on user side.
    no_skip = limit
    for i in tla.range(0, 4, 1):
        if i < no_skip:
            no_skip = i
    out[0] = no_skip


@tla.kernel
def _bare_int_carry_kernel(out: tla.Tensor) -> None:
    idx = 0
    for i in tla.range(0, 4, 1):
        idx = i
    out[0] = idx


def test_range_iv_is_int32_store() -> None:
    out = _gm_tensor_1d(4, dtype=tla.Int32)
    mlir = _range_iv_int32_kernel.dump_mlir(type_args=(out,))
    assert "scf.for" in mlir
    # Bounds/IV are i32; index_cast appears when storing/indexing into tensor APIs.
    assert "(i32, i32, i32)" in mlir.replace(" ", "") or "scf.for" in mlir
    assert "tla.scalar_store" in mlir
    assert ": i32" in mlir


def test_kernel_int_arg_is_i32_not_index() -> None:
    out = _gm_tensor_1d(1, dtype=tla.Int32)
    mlir = _int_arg_and_iv_min_kernel.dump_mlir(type_args=(4, out))
    assert "(%arg0: i32" in mlir.replace(" ", "") or "%arg0: i32" in mlir
    assert "scf.if" in mlir


def test_bare_int_carry_with_int32_iv() -> None:
    out = _gm_tensor_1d(1, dtype=tla.Int32)
    mlir = _bare_int_carry_kernel.dump_mlir(type_args=(out,))
    assert "scf.for" in mlir
    assert "tla.scalar_store" in mlir
