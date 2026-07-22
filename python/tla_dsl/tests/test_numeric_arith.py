# tests/test_numeric_arith.py — Numeric arithmetic operators

from __future__ import annotations

import operator

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _gm_tensor_1d(length: int, *, dtype: type = tla.Int32) -> tla.Tensor:
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
def _kernel_tiling_floordiv(tiling_int: tla.Tensor, out: tla.Tensor) -> None:
    embed = tiling_int[0]
    stages = embed // 128
    out[0] = stages


@tla.kernel
def _kernel_numeric_arith_i32(tiling_int: tla.Tensor, out: tla.Tensor) -> None:
    embed = tiling_int[0]
    a = embed + 1
    b = embed - 2
    c = embed * 3
    d = embed // 64
    e = embed % 64
    f = 256 // embed
    g = -embed
    out[0] = a
    out[1] = b
    out[2] = c
    out[3] = d
    out[4] = e
    out[5] = f
    out[6] = g


@tla.kernel
def _kernel_numeric_arith_f32(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    y = x + 1.5
    z = x * 2.0
    w = x / 4.0
    out[0] = y
    out[1] = z
    out[2] = w


@tla.kernel
def _kernel_numeric_binary(meta: tla.Tensor, out: tla.Tensor) -> None:
    a = meta[0]
    b = meta[1]
    out[0] = a + b
    out[1] = a * b
    out[2] = a // b


@tla.kernel
def _kernel_float_floordiv_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x // 2.0


@tla.kernel
def _kernel_numeric_bitwise(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x & 7
    out[1] = x | 1
    out[2] = x ^ 3
    out[3] = x << 2
    out[4] = x >> 1
    out[5] = ~x


@tla.kernel
def _kernel_numeric_int_pow_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x**2


@tla.kernel
def _kernel_numeric_float_pow(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x**2.0


def test_tiling_floordiv_returns_int32_and_divsi() -> None:
    tiling = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_tiling_floordiv.dump_mlir(type_args=(tiling, out))
    assert "tla.scalar_load" in mlir
    assert "arith.divsi" in mlir


def test_numeric_i32_arith_ops() -> None:
    tiling = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_numeric_arith_i32.dump_mlir(type_args=(tiling, out))
    assert "arith.addi" in mlir
    assert "arith.subi" in mlir
    assert "arith.muli" in mlir
    assert "arith.divsi" in mlir
    assert "arith.remsi" in mlir


def test_numeric_f32_arith_ops() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_arith_f32.dump_mlir(type_args=(meta, out))
    assert "arith.addf" in mlir
    assert "arith.mulf" in mlir
    assert "arith.divf" in mlir


def test_numeric_binary_with_numeric() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_numeric_binary.dump_mlir(type_args=(meta, out))
    assert "arith.addi" in mlir
    assert "arith.muli" in mlir
    assert "arith.divsi" in mlir


def test_numeric_float_floordiv_rejected() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    with pytest.raises(Exception, match="only supported for integer"):
        _kernel_float_floordiv_rejected.dump_mlir(type_args=(meta, out))


@tla.kernel
def _kernel_numeric_i32_truediv_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    a = meta[0]
    out[0] = a / 2


def test_numeric_i32_truediv_rejected() -> None:
    """Integer ``/`` is rejected; use ``//`` or float operands."""
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match=r"Numeric '/' is only supported for float"):
        _kernel_numeric_i32_truediv_rejected.dump_mlir(type_args=(meta, out))


def test_as_numeric_and_host_constructors() -> None:
    assert isinstance(tla.Int32(7), tla.Int32)
    assert isinstance(tla.Float32(1.5), tla.Float32)
    assert isinstance(tla.as_numeric(3), tla.Int32)
    assert isinstance(tla.as_numeric(1.25), tla.Float32)
    assert tla.Int32(2) + tla.Int32(3)  # host python path
    assert int(tla.Int32(2) + tla.Int32(3)) == 5


def test_numeric_same_type_required_and_div_rules() -> None:
    """Same-type only; integer ``/`` rejected; float ``/`` and int ``//`` ok."""
    with pytest.raises(TypeError, match=r"Numeric '/' is only supported for float"):
        _ = tla.Int32(4) / tla.Int32(2)
    with pytest.raises(TypeError, match="same type"):
        _ = tla.Int32(1) + tla.Float32(1.0)
    assert isinstance(tla.Float32(4.0) / tla.Float32(2.0), tla.Float32)
    assert (tla.Float32(4.0) / tla.Float32(2.0)).value == 2.0
    assert int(tla.Int32(4) // tla.Int32(2)) == 2


def test_mixed_numeric_comparison_requires_explicit_cast() -> None:
    """Rich comparisons reject mixed Numeric types (including ``==`` / ``!=``)."""
    left = tla.Int32(1)
    right = tla.Float32(1.0)
    for op in (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ):
        with pytest.raises(TypeError, match="same type"):
            _ = op(left, right)


def test_numeric_host_bitwise_and_float_pow() -> None:
    a = tla.Int32(0b1100)
    b = tla.Int32(0b1010)
    assert int(a & b) == 0b1000
    assert int(a | b) == 0b1110
    assert int(a ^ b) == 0b0110
    assert int(a << 1) == 0b11000
    assert int(a >> 2) == 0b11
    assert int(~tla.Int32(0)) == -1
    with pytest.raises(TypeError, match=r"Numeric '\*\*' is only supported for float"):
        _ = tla.Int32(3) ** 2
    assert (tla.Float32(3.0) ** tla.Float32(2.0)).value == 9.0
    assert bool(tla.Int32(1)) is True
    assert bool(tla.Int32(0)) is False


def test_numeric_bitwise_mlir() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_numeric_bitwise.dump_mlir(type_args=(meta, out))
    assert "arith.andi" in mlir
    assert "arith.ori" in mlir
    assert "arith.xori" in mlir
    assert "arith.shli" in mlir
    assert "arith.shrsi" in mlir
    assert "math.ipowi" not in mlir


def test_numeric_int_pow_rejected() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match=r"Numeric '\*\*' is only supported for float"):
        _kernel_numeric_int_pow_rejected.dump_mlir(type_args=(meta, out))


def test_numeric_float_pow_mlir() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_float_pow.dump_mlir(type_args=(meta, out))
    assert "math.powf" in mlir
    assert "math.ipowi" not in mlir


def test_cast_host_and_ssa_sitofp() -> None:
    """``cast`` / ``.to`` construct the target type; SSA cross-type emits arith ops."""
    from mlir import ir as mlir_ir

    # Host: no MLIR context required
    assert isinstance(tla.cast(5, tla.Float32), tla.Float32)
    host_f = tla.cast(tla.Int32(5), tla.Float32)
    assert isinstance(host_f, tla.Float32)
    assert host_f.value == 5.0
    assert tla.Int32(5).to(tla.Float32).value == 5.0

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown(), mlir_ir.InsertionPoint(
        mlir_ir.Module.create().body
    ):
        with runtime_mod._frontend_emission(module=None):
            i32_ty = mlir_ir.IntegerType.get_signless(32)
            i32_val = mlir_ir.Operation.create(
                "arith.constant",
                results=[i32_ty],
                attributes={"value": mlir_ir.IntegerAttr.get(i32_ty, 5)},
            ).results[0]
            result = tla.cast(tla.Int32(i32_val), tla.Float32)
            ir_v = result.ir_value()
            assert isinstance(result, tla.Float32)
            assert str(ir_v.type) == "f32"
            assert ir_v.owner.name == "arith.sitofp"


def test_numeric_add_captures_user_loc() -> None:
    """Numeric arithmetic attaches caller source location (via _capture_user_loc)."""
    from mlir import ir as mlir_ir

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown(), mlir_ir.InsertionPoint(
        mlir_ir.Module.create().body
    ):
        with runtime_mod._frontend_emission(module=None):
            i32_ty = mlir_ir.IntegerType.get_signless(32)
            i32_val = mlir_ir.Operation.create(
                "arith.constant",
                results=[i32_ty],
                attributes={"value": mlir_ir.IntegerAttr.get(i32_ty, 3)},
            ).results[0]
            result = tla.Int32(i32_val) + 1
            loc_str = str(result.ir_value().owner.location)
            assert "test_numeric_arith.py" in loc_str
            assert "unknown" not in loc_str.lower()
