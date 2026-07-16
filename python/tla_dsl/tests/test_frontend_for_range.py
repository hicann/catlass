import builtins
import inspect
from dataclasses import dataclass
from typing import Any

import pytest

import catlass as tla
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass.base_dsl import BaseDSL

tla_alias = tla
tla_range = tla.range
tla_range_constexpr = tla.range_constexpr
range_constexpr = tla.range_constexpr


@dataclass
class LoopState:
    coord: Any
    offset: Any


class CustomLoopState:
    def __init__(self, coord: Any, offset: Any) -> None:
        self.coord = coord
        self.offset = offset

    def __extract_mlir_values__(self) -> list[Any]:
        return [
            core_api_mod._as_branch_value(self.coord),
            core_api_mod._as_branch_value(self.offset),
        ]

    def __new_from_mlir_values__(self, values: list[Any]) -> "CustomLoopState":
        if len(values) != 2:
            raise ValueError(f"CustomLoopState expects 2 values, got {len(values)}")
        return CustomLoopState(
            core_api_mod._wrap_frontend_value(values[0]),
            core_api_mod._wrap_frontend_value(values[1]),
        )


@tla.kernel
def make_coord_from_for_index_ok(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        tla.make_coord(i, 0)


@tla.kernel
def make_coord_from_index_arith_ok(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        row = i // 2
        tla.make_coord(row, 0)


@tla.kernel
def range_alias_kernel(mem_a: tla.Tensor) -> None:
    loop_range = tla.range(0, 4, 1)
    for i in loop_range:
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_stop_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(4):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_start_stop_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(1, 4):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_module_alias_kernel(mem_a: tla.Tensor) -> None:
    for i in tla_alias.range(0, 4, 1):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_function_alias_kernel(mem_a: tla.Tensor) -> None:
    for i in tla_range(0, 4, 1):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bare_range_kernel(mem_a: tla.Tensor) -> None:
    for i in range(0, 4, 1):  # type: ignore[name-defined]
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def builtin_range_kernel(mem_a: tla.Tensor) -> None:
    for i in range(2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def local_shadowed_range_kernel(mem_a: tla.Tensor) -> None:
    range = builtins.range
    for i in range(2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_constexpr_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range_constexpr(0, 2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_constexpr_module_alias_kernel(mem_a: tla.Tensor) -> None:
    for i in tla_alias.range_constexpr(0, 2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_constexpr_function_alias_kernel(mem_a: tla.Tensor) -> None:
    for i in tla_range_constexpr(0, 2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bare_range_constexpr_kernel(mem_a: tla.Tensor) -> None:
    for i in range_constexpr(0, 2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_attrs_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(
        0,
        4,
        1,
        unroll=4,
        prefetch_stages=2,
    ):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_pipelining_alias_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(0, 4, 1, pipelining=2):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_unroll_full_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(0, 4, 1, unroll_full=True):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_negative_step_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(4, 0, -1):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_dynamic_step_kernel(
    mem_a: tla.Tensor, start: int, stop: int, step: int
) -> None:
    for i in tla.range(start, stop, step):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def range_tuple_carried_state_kernel(limit: int) -> None:
    state = (0, 1)
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
    tla.make_coord(state[0], state[1])


@tla.kernel
def range_list_carried_state_kernel(limit: int) -> None:
    state = [0, 1]
    for i in tla.range(0, limit, 1):
        state = [i, i + 1]
    tla.make_coord(state[0], state[1])


@tla.kernel
def range_dict_carried_state_kernel(limit: int) -> None:
    state = {"coord": 0, "offset": 1}
    for i in tla.range(0, limit, 1):
        state = {"coord": i, "offset": i + 1}
    tla.make_coord(state["coord"], state["offset"])


@tla.kernel
def range_dataclass_carried_state_kernel(limit: int) -> None:
    state = LoopState(0, 1)
    for i in tla.range(0, limit, 1):
        state = LoopState(i, i + 1)
    tla.make_coord(state.coord, state.offset)


@tla.kernel
def range_custom_class_carried_state_kernel(limit: int) -> None:
    state = CustomLoopState(0, 1)
    for i in tla.range(0, limit, 1):
        state = CustomLoopState(i, i + 1)
    tla.make_coord(state.coord, state.offset)


@tla.kernel
def range_active_object_method_call_kernel(limit: int) -> None:
    values = [0, 1]
    for i in tla.range(0, limit, 1):
        values.reverse()
        values = [i, i + 1]
    tla.make_coord(values[0], values[1])


@tla.kernel
def bad_range_active_object_method_structure_kernel(limit: int) -> None:
    values = [0]
    for i in tla.range(0, limit, 1):
        values.append(i)
    tla.make_coord(values[0], 0)


@tla.kernel
def bad_range_active_closure_call_kernel(limit: int) -> None:
    def helper(value: int) -> None:
        tla.make_coord(value, 0)

    for i in tla.range(0, limit, 1):
        helper(i)


@tla.kernel
def bad_range_keyword_bounds_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(start=0, end=4, step=1):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bad_range_alias_keyword_bounds_kernel(mem_a: tla.Tensor) -> None:
    loop_range = tla.range(start=0, end=4, step=1)
    for i in loop_range:
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bad_range_unknown_keyword_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(0, 4, 1, unknown=True):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bad_range_kwargs_kernel(mem_a: tla.Tensor) -> None:
    opts = {"unroll": 1}
    for i in tla.range(0, 4, 1, **opts):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bad_range_pipelining_prefetch_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(0, 4, 1, prefetch_stages=2, pipelining=3):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bad_range_no_bounds_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range():
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


@tla.kernel
def bad_range_too_many_bounds_kernel(mem_a: tla.Tensor) -> None:
    for i in tla.range(0, 1, 2, 3):
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))


def test_make_coord_accepts_for_loop_index_var() -> None:
    mlir = make_coord_from_for_index_ok.dump_mlir(type_args=(16,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.make_coord" in mlir


def test_make_coord_accepts_index_arith_var() -> None:
    mlir = make_coord_from_index_arith_ok.dump_mlir(type_args=(16,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.divui" in mlir
    assert "tla.make_coord" in mlir


def test_range_alias_lowers_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_range_arities_lower_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    for kernel in (range_stop_kernel, range_start_stop_kernel):
        mlir = kernel.dump_mlir(type_args=(mem,))
        assert "scf.for" in mlir
        assert "tla.range" not in mlir
        assert "tla.for" not in mlir
        assert "tla.tile_view" in mlir


def test_prefixed_range_module_alias_lowers_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_module_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_range_function_alias_lowers_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_function_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_imported_bare_range_lowers_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    old_range = globals().get("range", builtins.range)
    globals()["range"] = tla.range
    try:
        mlir = bare_range_kernel.dump_mlir(type_args=(mem,))
    finally:
        globals()["range"] = old_range
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_builtin_bare_range_remains_static_python_loop() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    old_range = globals().get("range", builtins.range)
    globals()["range"] = builtins.range
    try:
        mlir = builtin_range_kernel.dump_mlir(type_args=(mem,))
    finally:
        globals()["range"] = old_range
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_local_shadowed_range_remains_static_python_loop() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = local_shadowed_range_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_range_constexpr_unrolls_as_python_loop() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_constexpr_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_range_constexpr_aliases_unroll_as_python_loop() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    for kernel in (
        range_constexpr_module_alias_kernel,
        range_constexpr_function_alias_kernel,
    ):
        mlir = kernel.dump_mlir(type_args=(mem,))
        assert "scf.for" not in mlir
        assert "tla.range" not in mlir
        assert "tla.for" not in mlir
        assert mlir.count("tla.tile_view") == 2


def test_imported_bare_range_constexpr_unrolls_as_python_loop() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = bare_range_constexpr_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_range_loop_attrs_attach_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_attrs_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "loop_annotation" in mlir
    assert "count = 4 : i32" in mlir
    assert "cutlass.pipelining = 2 : i32" in mlir


def test_range_pipelining_alias_attaches_prefetch_attr() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_pipelining_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "cutlass.pipelining = 2 : i32" in mlir


def test_range_loop_unroll_full_attr_attaches_to_scf_for() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_unroll_full_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "loop_annotation" in mlir
    assert "full = true" in mlir


def test_range_negative_step_rewrites_at_ast_level() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_negative_step_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "arith.cmpi slt" not in mlir
    assert "arith.select" not in mlir
    assert "arith.subi" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_range_dynamic_step_rewrites_at_ast_level() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    mlir = range_dynamic_step_kernel.dump_mlir(type_args=(mem, 4, 0, -1))
    assert "scf.for" in mlir
    assert "scf.if" in mlir
    assert "arith.cmpi" in mlir
    assert "arith.subi" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_range_structured_carried_values_lower_to_scf_for_results() -> None:
    for kernel in (
        range_tuple_carried_state_kernel,
        range_list_carried_state_kernel,
        range_dict_carried_state_kernel,
        range_dataclass_carried_state_kernel,
        range_custom_class_carried_state_kernel,
    ):
        mlir = kernel.dump_mlir(type_args=(2,))
        assert "scf.for" in mlir
        assert "scf.yield" in mlir
        assert "tla.for" not in mlir
        assert "tla.make_coord" in mlir


def test_range_active_object_method_call_lowers_as_carried_value() -> None:
    mlir = range_active_object_method_call_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_range_rejects_active_object_method_structure_change() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'values'.*structure"):
        _ = bad_range_active_object_method_structure_kernel.dump_mlir(type_args=(2,))


def test_range_rejects_active_closure_call() -> None:
    with pytest.raises(SyntaxError, match="active local callable"):
        _ = bad_range_active_closure_call_kernel.dump_mlir(type_args=(2,))


def test_dynamic_tla_range_rejects_keyword_bounds() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    for kernel in (
        bad_range_keyword_bounds_kernel,
        bad_range_alias_keyword_bounds_kernel,
    ):
        with pytest.raises(SyntaxError, match="bounds must be positional"):
            _ = kernel.dump_mlir(type_args=(mem,))


def test_dynamic_tla_range_rejects_unknown_keyword() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    with pytest.raises(SyntaxError, match="unsupported keyword 'unknown'"):
        _ = bad_range_unknown_keyword_kernel.dump_mlir(type_args=(mem,))


def test_dynamic_tla_range_rejects_kwargs_unpacking() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    with pytest.raises(SyntaxError, match=r"\*\*kwargs"):
        _ = bad_range_kwargs_kernel.dump_mlir(type_args=(mem,))


def test_dynamic_tla_range_rejects_prefetch_and_pipelining_together() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    with pytest.raises(SyntaxError, match="prefetch_stages.*pipelining"):
        _ = bad_range_pipelining_prefetch_kernel.dump_mlir(type_args=(mem,))


def test_dynamic_tla_range_rejects_invalid_bound_arity() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 16),
            tla.Float16,
            origin_shape=tla.make_shape(16, 16),
        )
    for kernel in (bad_range_no_bounds_kernel, bad_range_too_many_bounds_kernel):
        with pytest.raises(SyntaxError, match="expects 1, 2, or 3 positional bounds"):
            _ = kernel.dump_mlir(type_args=(mem,))


def test_execution_only_mode_lowers_tla_range_loop() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    mlir = BaseDSL()._func(
        lowered,
        kind="kernel",
        options={},
        type_args=(mem_a,),
    )
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.make_coord" in mlir


def test_dynamic_tla_range_loop_rejects_return() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            return

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_break() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            break

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_continue() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            continue

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_raise() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            raise RuntimeError("unsupported")

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_for_else() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)
        else:
            tla.make_coord(0, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="for-else"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_non_name_target() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        pair = (0, 1)
        for pair[0] in tla.range(0, 16, 1):
            tla.make_coord(0, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="simple local name target"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_new_value_used_after() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for i in tla.range(0, 16, 1):
            coord = i
        tla.make_coord(coord, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="initialized before the loop"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_induction_value_used_after() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)
        tla.make_coord(i, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    with pytest.raises(SyntaxError, match="induction variables"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_execution_only_mode_handles_python_range_loop() -> None:
    def supported(mem_a: tla.Tensor) -> None:
        for _i in range(4):
            tla.make_coord(0, 0)

    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape(1, 2),
            tla.Float16,
            origin_shape=tla.make_shape(1, 2),
        )
    mlir = BaseDSL()._func(
        supported,
        kind="kernel",
        options={},
        type_args=(mem_a,),
    )
    assert mlir.count("tla.make_coord") == 4

@tla.kernel
def dynamic_for_bad_list_index_kernel(limit: int) -> None:
    values = [0, 1]
    for i in tla.range(0, limit, 1):
        values[i]


def _source_line(fn: Any, needle: str) -> int:
    source_lines, first_lineno = inspect.getsourcelines(fn.fn)
    for offset, line in enumerate(source_lines):
        if needle in line:
            return first_lineno + offset
    raise AssertionError(f"Unable to find source line containing {needle!r}")


def test_dynamic_for_body_operation_uses_original_source_location() -> None:
    line = _source_line(make_coord_from_for_index_ok, "tla.make_coord(i, 0)")
    lowered = BaseDSL()._lower(
        make_coord_from_for_index_ok.fn,
        kind=make_coord_from_for_index_ok.kind,
        options=dict(make_coord_from_for_index_ok.options),
        type_args=(4,),
        location=make_coord_from_for_index_ok.decorator_location,
    )
    with lowered.context:
        mlir = lowered.module.operation.get_asm(
            print_generic_op_form=True,
            assume_verified=False,
            enable_debug_info=True,
        )

    assert "__tladsl_loop_body_" in mlir
    assert f'"{__file__}":{line}:' in mlir


def test_dynamic_for_body_error_reports_original_source_location() -> None:
    line = _source_line(dynamic_for_bad_list_index_kernel, "values[i]")
    with pytest.raises(Exception) as excinfo:
        dynamic_for_bad_list_index_kernel.dump_mlir(type_args=(4,))
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in dynamic for body" in message
    assert f"{__file__}:{line}" in message
    assert "source: values[i]" in message
    assert "list indices" in message
    assert "_IndexExpr" in message

