from __future__ import annotations

from pathlib import Path

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass._mlir_bindings import tla_ops_gen

_REG_LOGIC_DTYPE: type[tla.Numeric] = tla.Int32


@tla.kernel
def _mask_logic_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            zero = tla.sub(reg, reg)
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            h_mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            q_mask = tla.create_mask(pattern=tla.mask.Q, dtype=tla.Float32)
            m4_mask = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)

            not_mask = tla.not_(q_mask, mask=all_mask)
            and_mask = tla.and_(h_mask, m4_mask, all_mask)
            or_mask = tla.or_(q_mask, m4_mask, all_mask)
            xor_mask = tla.xor(h_mask, m4_mask, all_mask)

            tmp0 = tla.where(not_mask, reg, zero)
            tmp1 = tla.where(and_mask, tmp0, zero)
            tmp2 = tla.where(or_mask, tmp1, zero)
            dst_tile.store(tla.where(xor_mask, tmp2, zero), mask=all_mask)


@tla.kernel
def _regtensor_logic_unary_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=_REG_LOGIC_DTYPE)
            dst_tile.store(tla.not_(reg), mask=all_mask)


@tla.kernel
def _regtensor_logic_binary_kernel(src0: tla.Tensor, src1: tla.Tensor, dst: tla.Tensor) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg0 = src0_tile.load()
            reg1 = src1_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=_REG_LOGIC_DTYPE)
            and_reg = tla.and_(reg0, reg1, all_mask)
            or_reg = tla.or_(and_reg, reg0, all_mask)
            dst_tile.store(tla.xor(or_reg, reg1, all_mask), mask=all_mask)


def _tensor_args(dtype: type[tla.Numeric] = tla.Float32, count: int = 2) -> tuple[tla.Tensor, ...]:
    with runtime_mod._eager_capture():
        return tuple(
            tla.Tensor(
                tla.make_shape(64), dtype, origin_shape=tla.make_shape(64)
            )
            for _ in range(count)
        )


@pytest.mark.parametrize(
    ("binding_name", "op_name"),
    (
        ("mask_not", "tla.mask_not"),
        ("mask_and", "tla.mask_and"),
        ("mask_or", "tla.mask_or"),
        ("mask_xor", "tla.mask_xor"),
    ),
)
def test_mask_logic_bindings_and_public_ops_emit_mlir(
    binding_name: str, op_name: str
) -> None:
    assert hasattr(tla_ops_gen, binding_name)
    assert op_name in _mask_logic_kernel.dump_mlir(type_args=_tensor_args())


_REG_LOGIC_UNARY_DTYPES = (tla.Float16, tla.Float32, tla.Int32, tla.Int16, tla.Int8)
_REG_LOGIC_BINARY_DTYPES = (
    tla.Float16,
    tla.BFloat16,
    tla.Float32,
    tla.Int32,
    tla.Int16,
    tla.Int8,
)
_REGTENSOR_LOGIC_CASES = tuple(
    ("reg_not", "tla.reg_not", dtype, _regtensor_logic_unary_kernel, 2)
    for dtype in _REG_LOGIC_UNARY_DTYPES
) + tuple(
    (binding_name, op_name, dtype, _regtensor_logic_binary_kernel, 3)
    for binding_name, op_name in (
        ("reg_and", "tla.reg_and"),
        ("reg_or", "tla.reg_or"),
        ("reg_xor", "tla.reg_xor"),
    )
    for dtype in _REG_LOGIC_BINARY_DTYPES
)


@pytest.mark.parametrize(("binding_name", "op_name", "dtype", "kernel", "tensor_count"), _REGTENSOR_LOGIC_CASES)
def test_regtensor_logic_bindings_and_public_ops_emit_mlir(
    binding_name: str, op_name: str, dtype: type[tla.Numeric], kernel: object, tensor_count: int
) -> None:
    global _REG_LOGIC_DTYPE
    _REG_LOGIC_DTYPE = dtype
    assert hasattr(tla_ops_gen, binding_name)
    assert op_name in kernel.dump_mlir(type_args=_tensor_args(dtype, tensor_count))


def test_logic_ops_are_in_vector_lowering_info() -> None:
    pass_source = Path(
        __file__
    ).parents[1] / "csrc/mlir/lib/Passes/TlaVectorRegionPass.cpp"
    source = pass_source.read_text(encoding="utf-8")

    for op_name in ("MaskNotOp", "MaskAndOp", "MaskOrOp", "MaskXorOp"):
        assert f"::tla::{op_name}" in source
    for op_name in ("RegNotOp", "RegAndOp", "RegOrOp", "RegXorOp"):
        assert f"::tla::{op_name}" in source


def test_mask_logic_rejects_non_mask_predicate() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.not_.*mask"):
        _invalid_mask_logic_kernel.dump_mlir(type_args=_tensor_args())


@tla.kernel
def _invalid_mask_logic_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            dst_tile.store(reg, mask=tla.not_(all_mask, mask=reg))


def test_regtensor_logic_rejects_mixed_regtensor_and_maskreg() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="both be MaskReg values or both be RegTensor"):
        _invalid_mixed_logic_kernel.dump_mlir(type_args=_tensor_args(tla.Int32, 2))


@tla.kernel
def _invalid_mixed_logic_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
            bad = tla.and_(reg, all_mask, all_mask)
            dst_tile.store(bad, mask=all_mask)
