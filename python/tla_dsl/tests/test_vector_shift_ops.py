from __future__ import annotations

import inspect

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _ub_tensor(dtype: type[tla.Numeric], lanes: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def _signed_shift_kernel(
    source: tla.Tensor, amounts: tla.Tensor, destination: tla.Tensor
) -> None:
    source_tile = tla.tile_view(source, tla.make_shape(128), tla.make_coord(0))
    amount_tile = tla.tile_view(amounts, tla.make_shape(128), tla.make_coord(0))
    destination_tile = tla.tile_view(
        destination, tla.make_shape(128), tla.make_coord(0)
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source_reg = source_tile.load()
            bit_shift_reg = amount_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            _ = tla.shift_left(source_reg, bit_shift_reg, mask=mask)
            _ = tla.shift_right(source_reg, bit_shift_reg, mask=mask)
            _ = tla.shift_left(source_reg, 3, mask=mask)
            result = tla.shift_right(source_reg, 4, mask=mask)
            destination_tile.store(result, mask=mask)


@tla.kernel
def _implicit_mask_shift_kernel(source: tla.Tensor, amounts: tla.Tensor) -> None:
    source_tile = tla.tile_view(source, tla.make_shape(128), tla.make_coord(0))
    amount_tile = tla.tile_view(amounts, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source_reg = source_tile.load()
            bit_shift_reg = amount_tile.load()
            _ = tla.shift_left(source_reg, bit_shift_reg)
            _ = tla.shift_right(source_reg, 3)


@tla.kernel
def _negative_scalar_shift_kernel(source: tla.Tensor) -> None:
    source_tile = tla.tile_view(source, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source_reg = source_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            _ = tla.shift_left(source_reg, -1, mask=mask)


@tla.kernel
def _invalid_shift_lanes_kernel(source: tla.Tensor, amounts: tla.Tensor) -> None:
    source_tile = tla.tile_view(source, tla.make_shape(128), tla.make_coord(0))
    amount_tile = tla.tile_view(amounts, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source_reg = source_tile.load()
            bit_shift_reg = amount_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            _ = tla.shift_left(source_reg, bit_shift_reg, mask=mask)


@tla.kernel
def _unsupported_i64_shift_kernel(source: tla.Tensor) -> None:
    source_tile = tla.tile_view(source, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source_reg = source_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int64)
            _ = tla.shift_left(source_reg, 1, mask=mask)


def test_shift_functions_expose_optional_mask() -> None:
    for name in ("shift_left", "shift_right"):
        parameters = inspect.signature(getattr(tla, name)).parameters
        assert parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["shift"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["mask"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["mask"].default is None


def test_shift_defaults_to_full_mask() -> None:
    source = _ub_tensor(tla.Int16, 128)
    amounts = _ub_tensor(tla.Int16, 128)
    mlir = _implicit_mask_shift_kernel.dump_mlir(type_args=(source, amounts))

    lines = mlir.splitlines()
    mask_lines = [line for line in lines if "tla.create_mask" in line]
    assert len(mask_lines) == 2
    assert all('pattern = "ALL"' in line for line in mask_lines)
    assert any("tla.shift_left " in line for line in lines)
    assert any("tla.shift_rights " in line for line in lines)


def test_shift_vector_and_scalar_forms_emit_distinct_tla_ops() -> None:
    source = _ub_tensor(tla.Int16, 128)
    amounts = _ub_tensor(tla.Int16, 128)
    mlir = _signed_shift_kernel.dump_mlir(type_args=(source, amounts, source))

    lines = mlir.splitlines()
    for op_name in (
        "tla.shift_left",
        "tla.shift_right",
        "tla.shift_lefts",
        "tla.shift_rights",
    ):
        assert sum(
            f'"{op_name}"' in line or f"{op_name} " in line for line in lines
        ) == 1
    scalar_lines = [
        line
        for line in lines
        if "tla.shift_lefts" in line or "tla.shift_rights" in line
    ]
    assert scalar_lines
    assert all("i16" in line for line in scalar_lines)


def test_shift_rejects_negative_scalar_amount() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="shift amount must be non-negative"):
        _negative_scalar_shift_kernel.dump_mlir(
            type_args=(_ub_tensor(tla.Int16, 128),)
        )


def test_shift_rejects_amount_vector_with_different_valid_lanes() -> None:
    source = _ub_tensor(tla.Int16, 128)
    amounts = _ub_tensor(tla.Int16, 64)
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="shift vector has 64 valid lanes, expected 128",
    ):
        _invalid_shift_lanes_kernel.dump_mlir(type_args=(source, amounts))


def test_shift_rejects_64_bit_source() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="unsupported source element type i64",
    ):
        _unsupported_i64_shift_kernel.dump_mlir(
            type_args=(_ub_tensor(tla.Int64, 32),)
        )
