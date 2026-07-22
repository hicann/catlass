"""AST control-flow compares should use typed Numeric / index_cast paths."""

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
def _numeric_compare_if_kernel(src: tla.Tensor, out: tla.Tensor) -> None:
    value = src[0]
    if value < 0:
        value = value + 1
    else:
        value = value + 2
    out[0] = value


@tla.kernel
def _index_vs_numeric_compare_kernel(limit_buf: tla.Tensor, out: tla.Tensor) -> None:
    # Loop / tile index (index SSA) vs scalar load (Int32 Numeric), as in fag_81:
    # ``kv_token_idx >= tile_range[i]``.
    limit = limit_buf[0]
    idx = 0
    for i in tla.range(0, 4, 1):
        idx = i
    result = tla.Int32(0)
    if idx >= limit:
        result = tla.Int32(1)
    else:
        result = tla.Int32(0)
    out[0] = result


def test_numeric_compare_in_if_uses_element_type_not_index() -> None:
    src = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _numeric_compare_if_kernel.dump_mlir(type_args=(src, out))
    assert "scf.if" in mlir
    assert "arith.cmpi" in mlir
    assert ": i32" in mlir
    assert "(i32, index)" not in mlir.replace(" ", "")


def test_index_vs_numeric_compare_emits_index_cast() -> None:
    limit_buf = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _index_vs_numeric_compare_kernel.dump_mlir(type_args=(limit_buf, out))
    assert "arith.cmpi" in mlir
    assert "arith.index_cast" in mlir
    assert "(i32, index)" not in mlir.replace(" ", "")
