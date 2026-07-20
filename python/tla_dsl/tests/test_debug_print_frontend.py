from __future__ import annotations

import inspect
import re

import pytest

import catlass as tla
import catlass.runtime as runtime_mod
from catlass.base_dsl import BaseDSL


@tla.kernel
def _scalar_kernel(i: object, j: object, f: object) -> None:
    with tla.cube():
        tla.debug_print(i)
        tla.debug_print(f)
    with tla.vector():
        tla.debug_print(f)
        tla.debug_print(j)


@tla.kernel
def _literal_kernel() -> None:
    with tla.vector():
        tla.debug_print(-(2**31))
        tla.debug_print(2**31 - 1)
        tla.debug_print(1.25)


@tla.kernel
def _f32_literal_location_kernel() -> None:
    with tla.vector():
        tla.debug_print(1.25)


@tla.kernel
def _regionless_kernel() -> None:
    tla.debug_print(tla.Int32(1))


@tla.kernel
def _typed_scalar_kernel(value: object) -> None:
    with tla.vector():
        tla.debug_print(value)


@tla.kernel
def _pointer_kernel() -> None:
    allocator = tla.utils.LocalmemAllocator()
    ptr = allocator.allocate(64, 32, tla.AddressSpace.ub)
    with tla.vector():
        tla.debug_print(ptr)


@tla.kernel
def _tensor_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.debug_print(value)


@tla.kernel
def _vector_value_kernel(value: tla.Tensor) -> None:
    tile = tla.tile_view(value, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.debug_print(tile.load())


def _host_vector_tensor() -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(64),
            tla.Float32,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(64),
            layout_tag=tla.arch.RowMajor,
        )


def test_debug_print_has_only_the_positional_unary_public_surface() -> None:
    assert str(inspect.signature(tla.debug_print)) == "(value, /)"


def test_debug_print_materializes_api_local_i32_and_f32_literals() -> None:
    mlir = _literal_kernel.dump_mlir()

    assert mlir.count("tla.debug_print") == 3
    assert f"{-(2**31)} : i32" in mlir
    assert f"{2**31 - 1} : i32" in mlir
    assert "1.250000e+00 : f32" in mlir or "1.25" in mlir


def test_debug_print_emits_direct_scalars_in_both_regions() -> None:
    mlir = _scalar_kernel.dump_mlir(
        type_args=(tla.Int32(0), tla.Int32(0), tla.Float32(0.0))
    )

    cube_start = mlir.index("tla.cube")
    vector_start = mlir.index("tla.vector")
    assert cube_start < vector_start
    for section in (mlir[cube_start:vector_start], mlir[vector_start:]):
        assert section.count("tla.debug_print") == 2
        assert "i32" in section and "f32" in section
    debug_lines = [line for line in mlir.splitlines() if "tla.debug_print" in line]
    assert any("%arg0" in line for line in debug_lines)
    assert any("%arg2" in line for line in debug_lines)


def test_debug_print_f32_literal_preserves_source_location() -> None:
    source_lines, first_lineno = inspect.getsourcelines(
        _f32_literal_location_kernel.fn
    )
    line = next(
        first_lineno + offset
        for offset, source in enumerate(source_lines)
        if "tla.debug_print(1.25)" in source
    )
    lowered = BaseDSL()._lower(
        _f32_literal_location_kernel.fn,
        kind=_f32_literal_location_kernel.kind,
        options=dict(_f32_literal_location_kernel.options),
        type_args=(),
        location=_f32_literal_location_kernel.decorator_location,
    )
    with lowered.context:
        mlir = lowered.module.operation.get_asm(
            print_generic_op_form=True,
            assume_verified=False,
            enable_debug_info=True,
        )

    constant_line = next(
        source
        for source in mlir.splitlines()
        if "arith.constant" in source and "f32" in source
    )
    location_alias = re.search(r"loc\((#loc\d+)\)", constant_line)
    assert location_alias is not None
    location_id = location_alias.group(1)
    for _ in range(8):
        location_line = next(
            source
            for source in mlir.splitlines()
            if source.startswith(f"{location_id} =")
        )
        if f'"{__file__}":{line}:' in location_line:
            break
        location_alias = re.search(r"\((#loc\d+)\)", location_line)
        assert location_alias is not None
        location_id = location_alias.group(1)
    else:
        pytest.fail("f32 debug-print constant did not retain its source location")


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((), {}, "exactly one positional argument; got 0"),
        (
            (tla.Int32(1), tla.Int32(2)),
            {},
            "exactly one positional argument; got 2",
        ),
        ((), {"value": tla.Int32(1)}, "does not accept keyword arguments"),
        ((2**31,), {}, "outside signless i32 range"),
        ((-(2**31) - 1,), {}, "outside signless i32 range"),
    ],
)
def test_debug_print_rejects_invalid_public_calls(
    args: tuple[object, ...], kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        tla.debug_print(*args, **kwargs)


_SCALAR_ERROR = "expected a signless i32 or f32 scalar"


@pytest.mark.parametrize(
    ("kernel", "type_args", "match"),
    [
        *[
            (_typed_scalar_kernel, (value,), _SCALAR_ERROR)
            for value in (True, tla.Int64(1), tla.Index(1), tla.Float16(1.0))
        ],
        (_pointer_kernel, (), _SCALAR_ERROR),
        (_tensor_kernel, (_host_vector_tensor(),), _SCALAR_ERROR),
        (_vector_value_kernel, (_host_vector_tensor(),), _SCALAR_ERROR),
        (
            _regionless_kernel,
            (),
            r"must be nested inside tla\.cube\(\) or tla\.vector\(\)",
        ),
    ],
)
def test_debug_print_rejects_invalid_values_and_placement(
    kernel: object, type_args: tuple[object, ...], match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        kernel.dump_mlir(type_args=type_args)
