from __future__ import annotations

from collections.abc import Callable

import pytest

import catlass as tla
import catlass.runtime as runtime_mod


def _vector_tensor(dtype: object = tla.Float32) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(64),
            dtype,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(64),
            layout_tag=tla.arch.RowMajor,
        )


@tla.kernel
def vector_scalar_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        reg = tile.load()
        mask = tla.create_mask(pattern=tla.mask.H)
        _ = tla.add(reg, 1.0)
        _ = tla.sub(reg, 2.0)
        _ = tla.mul(3.0, reg)
        _ = tla.max(reg, 5.0)
        _ = tla.min(6.0, reg)
        _ = tla.div(reg, 4.0, mask=mask)


@tla.kernel
def bf16_scalar_literal_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        _ = tla.add(tile.load(), 1.0)


@tla.kernel
def invalid_max_vector_rhs_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        reg = tile.load()
        _ = tla.max(reg, reg)


@tla.kernel
def invalid_scalar_lhs_sub_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        reg = tile.load()
        _ = tla.sub(3.0, reg)


@tla.kernel
def invalid_integer_scalar_fraction_kernel(lhs: tla.Tensor) -> None:
    tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vec.func(mode="simd"):
        reg = tile.load()
        _ = tla.add(reg, 1.9)


def test_vector_scalar_symbols_are_exported() -> None:
    for symbol in ("add", "sub", "mul", "max", "min", "div"):
        assert callable(getattr(tla, symbol))
    for symbol in (
        "adds",
        "subs",
        "muls",
        "maxs",
        "mins",
        "divs",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vadds",
        "vsubs",
        "vmuls",
        "vdivs",
    ):
        assert not hasattr(tla, symbol)


def test_vector_scalar_dispatch_and_masking() -> None:
    mlir = vector_scalar_kernel.dump_mlir(type_args=(_vector_tensor(),))

    for op_name in ("adds", "subs", "muls", "maxs", "mins", "divs"):
        assert mlir.count(f"tla.{op_name}") == 1
    divs_line = next(line for line in mlir.splitlines() if "tla.divs" in line)
    assert "mask" in divs_line
    assert "pass_thru" not in mlir


def test_bf16_scalar_literal_uses_bf16_constant(
    compiler_tlair: Callable[..., str],
) -> None:
    mlir = compiler_tlair(bf16_scalar_literal_kernel, type_args=(_vector_tensor(tla.BFloat16),))

    assert any("arith.constant" in line and "bf16" in line for line in mlir.splitlines())
    assert "tla.adds" in mlir


def test_scalar_only_vector_op_rejects_vector_rhs() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="expected vector-scalar operands"):
        invalid_max_vector_rhs_kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_noncommutative_vector_scalar_rejects_scalar_lhs() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="invalid argument 'lhs'"):
        invalid_scalar_lhs_sub_kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_integer_vector_scalar_rejects_fractional_literal() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="expected integer scalar"):
        invalid_integer_scalar_fraction_kernel.dump_mlir(
            type_args=(_vector_tensor(tla.Int32),)
        )
