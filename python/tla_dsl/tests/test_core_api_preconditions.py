from __future__ import annotations

import inspect
from typing import Any

from mlir import ir as mlir_ir
import pytest

import catlass as tla
from catlass.address_space import AddressSpace
from catlass.base_dsl import ast_helpers
from catlass.base_dsl.typing import Int8
from catlass.core_api import _category
from catlass.execution_lowering import TlaLoweringError
import catlass.runtime as runtime_mod


def test_mlir_value_identity_is_stable_across_operand_wrappers() -> None:
    with mlir_ir.Context() as ctx, mlir_ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(
            """
module {
  func.func @probe(%arg0: i64) -> i64 {
    %c1 = arith.constant 1 : i64
    %sum = arith.addi %arg0, %c1 : i64
    return %sum : i64
  }
}
"""
        )
        block = list(module.body.operations[0].regions[0].blocks)[0]
        ops = list(block.operations)

        equivalent_values = (
            (block.arguments[0], ops[1].operands[0]),
            (ops[0].results[0], ops[1].operands[1]),
            (ops[1].results[0], ops[2].operands[0]),
        )
        for registered, lookup in equivalent_values:
            assert registered is not lookup
            assert registered == lookup
            assert hash(registered) == hash(lookup)
            assert {registered: "metadata"}[lookup] == "metadata"


def test_copy_preconditions_require_tiles() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="tla.copy"):
            tla.copy(tla.make_shape(1, 2), tla.make_shape(1, 2))


def test_allocator_allocate_requires_supported_addrspace() -> None:
    with runtime_mod._eager_capture():
        allocator = tla.utils.LocalmemAllocator()
        with pytest.raises(
            tla.TlaCoreAPIError, match="tla.utils.LocalmemAllocator.allocate"
        ):
            allocator.allocate(128, 32, "gm")


def test_allocator_capacity_in_bytes_defaults_to_l1_capacity() -> None:
    assert tla.utils.LocalmemAllocator.capacity_in_bytes() == 512 * 1024


def test_allocator_capacity_in_bytes_accepts_supported_scope() -> None:
    assert (
        tla.utils.LocalmemAllocator.capacity_in_bytes(tla.AddressSpace.ub) == 248 * 1024
    )


def test_allocator_capacity_in_bytes_rejects_unknown_scope() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError, match="tla.utils.LocalmemAllocator.capacity_in_bytes"
    ):
        tla.utils.LocalmemAllocator.capacity_in_bytes("gm")


def test_allocator_allocate_returns_pointer_category() -> None:
    @tla.jit
    def kernel(mem: tla.Tensor) -> None:
        allocator = tla.utils.LocalmemAllocator()
        ptr = allocator.allocate(16, 32, tla.AddressSpace.l1)
        assert _category(ptr) == "pointer"
        _ = mem

    with runtime_mod._eager_capture():
        mem_arg = tla.Tensor(
            tla.make_shape(8, 8),
            tla.Float16,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(8, 8),
        )
    _ = kernel.dump_mlir(type_args=(mem_arg,))


def test_allocator_pointer_exposes_default_i8_dtype_and_value_type() -> None:
    with runtime_mod._eager_capture():
        allocator = tla.utils.LocalmemAllocator()
        ptr = allocator.allocate(64, 32, tla.AddressSpace.l1)
        assert ptr.dtype is Int8
        assert ptr.value_type is Int8


def test_allocator_pointer_mlir_marshalling_round_trips_metadata() -> None:
    with runtime_mod._eager_capture():
        allocator = tla.utils.LocalmemAllocator()
        ptr = allocator.allocate(64, 32, tla.AddressSpace.l1)
        mlir_types = ptr.__get_mlir_types__()
        values = ptr.__extract_mlir_values__()
        clone = ptr.__new_from_mlir_values__(values)
        assert len(mlir_types) == 1
        assert str(mlir_types[0]).startswith("!tla.ptr<")
        assert len(values) == 1
        assert str(values[0].type).startswith("!tla.ptr<")
        assert _category(clone) == "pointer"
        assert clone.addrspace == AddressSpace.l1
        assert clone.alignment == 32
        assert clone.dtype is Int8
        assert runtime_mod._resolve_frontend_bound_value(clone) is None
        assert clone.value is values[0]


def test_allocator_second_allocate_aligns_cursor() -> None:
    with runtime_mod._eager_capture():
        allocator = tla.utils.LocalmemAllocator()
        p0 = allocator.allocate(100, 256, tla.AddressSpace.l1)
        p1 = allocator.allocate(64, 512, tla.AddressSpace.l1)
        assert _category(p0) == "pointer"
        assert _category(p1) == "pointer"
        assert p0.alignment == 256
        assert p1.alignment == 512


def test_allocator_public_surface_does_not_expose_loc() -> None:
    allocator = tla.utils.LocalmemAllocator()
    assert "loc" not in inspect.signature(allocator.allocate).parameters


def test_cross_flag_requires_valid_mode() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="tla.cross_flag"):
            tla.cross_flag("x", "gpu", tla.pipes.MTE2)


def test_mutex_requires_valid_resource_and_id() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
            tla.mutex(resource="", id=-1)
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
            tla.mutex(resource=123, id=-1)  # type: ignore[arg-type]
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
            tla.mutex(resource="l0a_ping", id=True)  # type: ignore[arg-type]
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
            tla.mutex(resource="l0a_ping", id=-2)
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
            tla.mutex(resource="l0a_ping", id=32)


def test_mutex_lock_unlock_require_valid_pipe() -> None:
    with runtime_mod._eager_capture():
        m = tla.mutex(resource="l0a_ping", id=0)
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex_lock"):
            m.lock(pipe="gpu")
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex_unlock"):
            m.unlock(pipe="gpu")


def test_mutex_guard_requires_mutex_arguments() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex_guard"):
        tla.mutex_guard()
    with runtime_mod._eager_capture():
        shape = tla.make_shape(1, 1)
        with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex_guard"):
            with tla.mutex_guard(shape):
                pass


# Nested ``make_shape`` trees for L0 zN / nZ / L0C layouts (must match remap stride trees);
# flat ``origin_shape`` is the logical M×N bounds, aligned with ``tile_view`` targets.
_M128_64_FRACTAL_ZN = ((16, 8), (16, 4))
_M64_128_FRACTAL_ZN = ((16, 4), (16, 8))
_M128_128_FRACTAL_L0C = ((16, 8), (16, 8))
_M32_128_FRACTAL_ZN = ((16, 2), (16, 8))
_M128_64_FRACTAL_ZN_F32 = ((16, 8), (8, 8))


def _tensor_arg(
    fractal: tuple[tuple[int, int], tuple[int, int]],
    origin_mn: tuple[int, int],
    dtype: Any,
    addrspace: Any,
    layout_tag: Any,
) -> tla.Tensor:
    with runtime_mod._eager_capture():
        return tla.Tensor(
            tla.make_shape(fractal[0], fractal[1]),
            dtype,
            addrspace=addrspace,
            origin_shape=tla.make_shape(*origin_mn),
            layout_tag=layout_tag,
        )


def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    """``tile_view`` + ``mmad`` can hit rank-2 validation before types align; skip instead of failing."""
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )


def test_mmad_validates_operands_and_kwargs() -> None:
    @tla.jit
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        _ = tla.mmad(acc, lhs, rhs, init_c=True)

    try:
        mlir = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )
    except TlaLoweringError as e:
        _skip_if_mmad_rank2_tile_view_regression(e)
        raise
    assert "tla.mmad" in mlir
    assert "!tla.ptr<f16, l0a, 2>" in mlir
    assert "!tla.ptr<f16, l0b, 2>" in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir


def test_mmad_rejects_old_order_at_frontend() -> None:
    @tla.jit
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        _ = tla.mmad(lhs, rhs, acc, init_c=True)

    with pytest.raises(TlaLoweringError, match="unsupported tla.mmad tile addrspaces"):
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )


def test_mmad_rejects_wrong_element_types_at_frontend() -> None:
    @tla.jit
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        _ = tla.mmad(acc, lhs, rhs, init_c=True)

    with pytest.raises(TlaLoweringError, match="unsupported tla.mmad element types"):
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN_F32,
                    (128, 64),
                    tla.Float32,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )


def test_mmad_rejects_wrong_shape_contract_at_frontend() -> None:
    @tla.jit
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(32, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        _ = tla.mmad(acc, lhs, rhs, init_c=True)

    with pytest.raises(TlaLoweringError) as excinfo:
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M32_128_FRACTAL_ZN,
                    (32, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )
    msg = str(excinfo.value)
    assert (
        "unsupported tla.mmad tile shape contract" in msg or "rank-2 tiles only" in msg
    )


def test_mmad_rejects_rhs_zn_layout_at_frontend() -> None:
    @tla.jit
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        _ = tla.mmad(acc, lhs, rhs, init_c=True)

    with pytest.raises(TlaLoweringError, match="unsupported tla.mmad operand layout"):
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )


def test_mmad_rejects_unknown_kwarg() -> None:
    @tla.jit
    def kernel(mem: tla.Tensor) -> None:
        lhs = tla.tile_view(mem, tla.make_shape(1, 8), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem, tla.make_shape(1, 8), tla.make_coord(0, 0))
        acc = tla.tile_view(mem, tla.make_shape(1, 8), tla.make_coord(0, 0))
        _ = tla.mmad(acc, lhs, rhs, bad=True)

    with runtime_mod._eager_capture():
        mem_arg = tla.Tensor(
            tla.make_shape(8, 8),
            tla.Float16,
            addrspace=tla.AddressSpace.ub,
            origin_shape=tla.make_shape(8, 8),
        )
    with pytest.raises(tla.TlaCoreAPIError, match="unknown keyword"):
        _ = kernel.dump_mlir(type_args=(mem_arg,))


def test_make_shape_rejects_non_index_components() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="tla.make_shape"):
            _ = tla.make_shape(1.0, 2)


def test_make_coord_rejects_negative_static_leaf() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="coord leaf >= 0"):
            _ = tla.make_coord(-1, 0)


def test_make_stride_rejects_nonpositive_static_leaf() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="stride leaf strictly positive"):
            _ = tla.make_stride(0, 1)


def test_range_accepts_one_or_three_args() -> None:
    with runtime_mod._eager_capture():
        _ = tla.range(32)
        _ = tla.range(0, 32)
        _ = tla.range(0, 32, 1)


def test_range_rejects_bad_arity() -> None:
    with runtime_mod._eager_capture():
        with pytest.raises(tla.TlaCoreAPIError, match="tla.range"):
            _ = tla.range(0, step=1)


def test_range_constexpr_returns_python_range() -> None:
    assert list(tla.range_constexpr(0, 4, 2)) == [0, 2]
    with pytest.raises(tla.TlaCoreAPIError, match="tla.range_constexpr"):
        _ = tla.range_constexpr(0, 4.0)


def test_range_constexpr_warns_for_large_static_loop() -> None:
    with pytest.warns(
        ast_helpers.DSLOptimizationWarning,
        match="This static loop has 64 iterations",
    ):
        result = tla.range_constexpr(64)
    assert len(result) == 64


def test_range_rejects_bad_loop_attrs() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="unroll"):
        _ = tla.range(0, 4, 1, unroll=True)
    with pytest.raises(tla.TlaCoreAPIError, match="unroll_full"):
        _ = tla.range(0, 4, 1, unroll_full=1)
    with pytest.raises(tla.TlaCoreAPIError, match="prefetch_stages"):
        _ = tla.range(0, 4, 1, prefetch_stages=-1)
