from catlass.tla.runtime import make_fake_tensor

import ast
import builtins
import inspect
import re
from dataclasses import dataclass
from typing import Any

import pytest

import catlass.tla as tla
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass.base_dsl import BaseDSL
from catlass.base_dsl.ast_preprocessor import (
    _FrontendControlFlowTransformer,
    _FunctionAnalyzer,
    _scope_facts_for_transform,
)

tla_alias = tla
tla_range = tla.range
tla_range_constexpr = tla.range_constexpr
range_constexpr = tla.range_constexpr


def _return_in_tuple_loop(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            for _ in (0,):
                return


def _return_in_list_loop(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            for _ in [0]:
                return


def _python_loop_break_in_dynamic_if(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            for item in (0, 1):
                break
        tla.make_coord(i, 0)


def _python_loop_continue_in_dynamic_if(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            for item in (0, 1):
                continue
        tla.make_coord(i, 0)


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
def bad_range_active_closure_call_kernel(limit: int) -> None:
    def helper(value: int) -> None:
        tla.make_coord(value, 0)

    for i in tla.range(0, limit, 1):
        helper(i)


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
    assert ("arith.divui" in mlir) or ("arith.divsi" in mlir)
    assert "tla.make_coord" in mlir


def test_range_alias_lowers_to_scf_for() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = range_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_range_alias_is_preserved_in_nested_outlined_regions() -> None:
    source = (
        "def kernel(predicate, limit):\n"
        "    loop_range = tla.range(0, limit, 1)\n"
        "    if predicate:\n"
        "        for i in loop_range:\n"
        "            if i == 0:\n"
        "                tla.make_coord(i, 0)\n"
    )
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    scope_facts = _scope_facts_for_transform(
        source, "nested_range_alias.py", target
    )
    transformer = _FrontendControlFlowTransformer(
        {"tla": tla},
        filename="nested_range_alias.py",
        source_text=source,
        root_plan=_FunctionAnalyzer(
            global_symbols={"tla": tla}, scope_facts=scope_facts
        ).analyze(target),
    )

    transformed = transformer.visit(tree)

    assert sum(
        isinstance(node, ast.Name) and node.id == "__tladsl_internal_for__"
        for node in ast.walk(transformed)
    ) == 1


def test_range_alias_keeps_hidden_exit_owned_by_nested_dynamic_loop() -> None:
    source = (
        "def kernel(predicate, limit):\n"
        "    loop_range = tla.range(0, limit, 1)\n"
        "    if predicate:\n"
        "        for i in loop_range:\n"
        "            return\n"
    )
    tree = ast.parse(source)
    target = tree.body[0]
    assert isinstance(target, ast.FunctionDef)
    scope_facts = _scope_facts_for_transform(
        source, "nested_range_alias.py", target
    )
    transformer = _FrontendControlFlowTransformer(
        {"tla": tla},
        filename="nested_range_alias.py",
        source_text=source,
        root_plan=_FunctionAnalyzer(
            global_symbols={"tla": tla}, scope_facts=scope_facts
        ).analyze(target),
    )

    with pytest.raises(SyntaxError, match="dynamic Tla for does not support return"):
        transformer.visit(tree)


def test_range_arities_lower_to_scf_for() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    for kernel in (range_stop_kernel, range_start_stop_kernel):
        mlir = kernel.dump_mlir(type_args=(mem,))
        assert "scf.for" in mlir
        assert "tla.range" not in mlir
        assert "tla.for" not in mlir
        assert "tla.tile_view" in mlir


def test_prefixed_range_module_alias_lowers_to_scf_for() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = range_module_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_range_function_alias_lowers_to_scf_for() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = range_function_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir


def test_imported_bare_range_lowers_to_scf_for() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
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
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
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
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = local_shadowed_range_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_range_constexpr_unrolls_as_python_loop() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = range_constexpr_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_range_constexpr_aliases_unroll_as_python_loop() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
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
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = bare_range_constexpr_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" not in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert mlir.count("tla.tile_view") == 2


def test_range_negative_step_rewrites_at_ast_level() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
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
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
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


def test_range_rejects_active_closure_call() -> None:
    with pytest.raises(SyntaxError, match="nested function definition"):
        _ = bad_range_active_closure_call_kernel.dump_mlir(type_args=(2,))


def test_execution_only_mode_lowers_tla_range_loop() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
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

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_if_rejects_return_hidden_in_nested_static_for() -> None:
    def lowered(limit: int) -> None:
        for i in tla.range(0, limit, 1):
            if i == 0:
                for _ in range(1):
                    return
            tla.make_coord(i, 0)

    with pytest.raises(SyntaxError, match="dynamic Tla if does not support return"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(4,))


@pytest.mark.parametrize(
    "lowered",
    [_return_in_tuple_loop, _return_in_list_loop],
    ids=["tuple", "list"],
)
def test_dynamic_if_rejects_return_hidden_in_arbitrary_python_for(
    lowered,
) -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if does not support return"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(4,))


@pytest.mark.parametrize(
    "lowered",
    [_python_loop_break_in_dynamic_if, _python_loop_continue_in_dynamic_if],
    ids=["break", "continue"],
)
def test_dynamic_if_allows_exit_owned_by_nested_python_loop(lowered) -> None:
    mlir = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(4,))
    assert "scf.for" in mlir
    assert "scf.if" in mlir


def test_dynamic_tla_range_loop_rejects_break() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            break

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_continue() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            continue

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_raise() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            raise RuntimeError("unsupported")

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="dynamic Tla for"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_for_else() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for _i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)
        else:
            tla.make_coord(0, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="for-else"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_non_name_target() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        pair = (0, 1)
        for pair[0] in tla.range(0, 16, 1):
            tla.make_coord(0, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="simple local name target"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_new_value_used_after() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for i in tla.range(0, 16, 1):
            coord = i
        tla.make_coord(coord, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="initialized before the loop"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_dynamic_tla_range_loop_rejects_induction_value_used_after() -> None:
    def lowered(mem_a: tla.Tensor) -> None:
        for i in tla.range(0, 16, 1):
            tla.make_coord(0, 0)
        tla.make_coord(i, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
            )
    with pytest.raises(SyntaxError, match="induction variables"):
        _ = BaseDSL()._func(lowered, kind="kernel", options={}, type_args=(mem_a,))


def test_execution_only_mode_handles_python_range_loop() -> None:
    def supported(mem_a: tla.Tensor) -> None:
        for _i in range(4):
            tla.make_coord(0, 0)

    mem_a = make_fake_tensor(
                tla.Float16,
                (1, 2),
                (2, 1),
                origin_shape=(1, 2),
                layout_tag=tla.arch.RowMajor,
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
    assert "cannot be used as a Python index" in message or "list indices" in message
    assert "Int32" in message


# --- carriers are decided by control-flow semantics --------------------------
# A name the body *assigns to* is redefined each iteration and must be carried.
# A name the body only invokes a method on is mutated in place -- a side effect
# on an object reached from the enclosing scope, the same classification a
# subscript store already gets. Carrying the latter put a UB tile into scf.for's
# iter_args, where it has lost the allocation it came from and its loads no
# longer legalize.


def _ub_tile(n: int):
    gm = tla.make_tensor(
        tla.allocate(n, tla.Float32, tla.AddressSpace.ub, 256),
        tla.make_layout(
            tla.make_shape(n), tla.make_stride(1), origin_shape=tla.make_shape(n)
        ),
        tla.make_coord(0),
    )
    return tla.tile_view(gm, tla.make_shape(64), tla.make_coord(0))


@tla.kernel
def _tile_store_in_dynamic_loop_kernel(limit: int) -> None:
    with tla.vector():
        acc = _ub_tile(1024)
        src = _ub_tile(1024)
        with tla.vec.func(mode="simd"):
            for _ in tla.range(0, limit, 1):
                acc.store(acc.load() + src.load())


@tla.kernel
def _tile_rebound_in_dynamic_loop_kernel(limit: int) -> None:
    with tla.vector():
        first = _ub_tile(1024)
        second = _ub_tile(1024)
        selected = first
        for _ in tla.range(0, limit, 1):
            selected = second
        tla.make_shape(selected.shape[0])


def _scf_for_operand_count(mlir_text: str) -> int:
    """Number of values the generated scf.for carries."""
    match = re.search(r"scf\.for[^\n]*iter_args\(([^)]*)\)", mlir_text)
    return 0 if match is None else len(
        [p for p in match.group(1).split(",") if p.strip()]
    )


def test_stored_tile_is_not_carried_through_scf_for() -> None:
    mlir_text = _tile_store_in_dynamic_loop_kernel.dump_mlir(type_args=(4,))
    assert "scf.for" in mlir_text, mlir_text
    carrier_lines = [ln for ln in mlir_text.splitlines() if "scf.for" in ln]
    assert carrier_lines
    assert all("!tla.tensor" not in ln for ln in carrier_lines), mlir_text
    assert _scf_for_operand_count(mlir_text) == 0, mlir_text


def test_rebound_tile_is_carried_and_still_reads_metadata() -> None:
    """Rebinding a tensor across a runtime construct stays supported.

    The tensor is carried, and reading metadata off the merged value works --
    the shape is resolved at trace time. What does not survive the merge is the
    base address, which only matters to an op that needs it; see
    test_copy_through_a_selected_tensor_is_the_known_gap.
    """
    mlir_text = _tile_rebound_in_dynamic_loop_kernel.dump_mlir(type_args=(4,))
    assert "scf.for" in mlir_text, mlir_text


@dataclass
class _TileHolder:
    tile: Any


@tla.kernel
def _struct_tile_store_in_dynamic_loop_kernel(limit: int) -> None:
    with tla.vector():
        state = _TileHolder(_ub_tile(1024))
        src = _ub_tile(1024)
        with tla.vec.func(mode="simd"):
            for _ in tla.range(0, limit, 1):
                state.tile.store(state.tile.load() + src.load())


def test_structured_state_holding_a_tile_is_not_carried() -> None:
    """The handle check is at the leaves, not the top level.

    `state` is a dataclass, not a tensor, so a top-level check misses it -- and
    the pytree machinery then flattens it and puts the inner tile into
    iter_args regardless.
    """
    mlir_text = _struct_tile_store_in_dynamic_loop_kernel.dump_mlir(type_args=(4,))
    assert "scf.for" in mlir_text, mlir_text
    carrier_lines = [ln for ln in mlir_text.splitlines() if "scf.for" in ln]
    assert carrier_lines
    assert all("!tla.tensor" not in ln for ln in carrier_lines), mlir_text
    assert _scf_for_operand_count(mlir_text) == 0, mlir_text


@tla.kernel
def _struct_tile_rebound_in_dynamic_loop_kernel(limit: int) -> None:
    with tla.vector():
        first = _ub_tile(1024)
        second = _ub_tile(1024)
        state = _TileHolder(first)
        for _ in tla.range(0, limit, 1):
            state = _TileHolder(second)
        with tla.vec.func(mode="simd"):
            state.tile.store(state.tile.load())


def test_rebound_tile_inside_a_dataclass_is_carried() -> None:
    """A tensor nested in a dataclass is carried like a bare one.

    pytree flattens the tile into the scf carriers either way, so the dataclass
    is not a special case -- it just has to survive the round trip.
    """
    mlir_text = _struct_tile_rebound_in_dynamic_loop_kernel.dump_mlir(type_args=(4,))
    assert "scf.for" in mlir_text, mlir_text


@tla.kernel
def _host_list_mutated_in_dynamic_loop_kernel(limit: int) -> None:
    values = [0, 1]
    for _ in tla.range(0, limit, 1):
        values.reverse()
    tla.make_coord(values[0], values[1])


def test_method_call_on_a_host_object_is_rejected() -> None:
    """A method-only capture has to be a tla handle.

    The body traces once, so `values.reverse()` runs once at trace time and the
    emitted IR does not depend on the trip count -- the loop silently computes
    something else. Only a handle whose methods emit ops is safe here.
    """
    with pytest.raises(tla.TlaCoreAPIError, match="calls a method on it"):
        _host_list_mutated_in_dynamic_loop_kernel.dump_mlir(type_args=(4,))


def _analyze_for(source: str, *, active: set[str]):
    """Run the shared control-flow analysis over a single `for` statement."""
    from catlass.base_dsl.ast_preprocessor import _ControlFlowAnalyzer

    node = ast.parse(source).body[0]
    return _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="for",
        assigned_regions=[node.body],
        active_call_nodes=[node],
        active_symbols=set(active),
        active_callables=set(),
    )


def test_method_only_name_is_captured_not_carried() -> None:
    """The rule is lexical, so it holds whatever the object's runtime type is.

    `state.tile.store(...)` invokes on `state`; a check on the carried value
    would have to see through the dataclass to the tile inside it.
    """
    plan = _analyze_for(
        "for i in tla.range(0, limit, 1):\n    state.tile.store(value)\n",
        active={"state", "value", "limit"},
    )
    assert plan.carried_names == ()
    assert plan.captured_names == ("state",)


def test_assigned_name_is_carried() -> None:
    plan = _analyze_for(
        "for i in tla.range(0, limit, 1):\n    total = total + i\n",
        active={"total", "limit"},
    )
    assert plan.carried_names == ("total",)
    assert plan.captured_names == ()


def test_assigned_and_invoked_name_is_carried_once() -> None:
    plan = _analyze_for(
        "for i in tla.range(0, limit, 1):\n    acc.update(i)\n    acc = acc + i\n",
        active={"acc", "limit"},
    )
    assert plan.carried_names == ("acc",)
    assert plan.captured_names == ()
