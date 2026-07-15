from typing import Any

import pytest

import catlass as tla
import catlass.tla_ast_decorators as ast_decorators_mod
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod
from catlass.base_dsl.ast_helpers import while_executor

const_expr = tla.const_expr


class CustomWhileState:
    def __init__(self, coord: Any, offset: Any) -> None:
        self.coord = coord
        self.offset = offset

    def __extract_mlir_values__(self) -> list[Any]:
        return [
            core_api_mod._as_branch_value(self.coord),
            core_api_mod._as_branch_value(self.offset),
        ]

    def __new_from_mlir_values__(self, values: list[Any]) -> "CustomWhileState":
        if len(values) != 2:
            raise ValueError(f"CustomWhileState expects 2 values, got {len(values)}")
        return CustomWhileState(
            core_api_mod._wrap_frontend_value(values[0]),
            core_api_mod._wrap_frontend_value(values[1]),
        )


@tla.kernel
def statement_while_carried_index_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        i = i + 1
    tla.make_coord(i, 0)


@tla.kernel
def statement_while_compound_bool_kernel(limit: int) -> None:
    i = 0
    while (
        i < limit
        and i < 1024
        and i >= 0
        and i != 1025
        and i <= 1024
    ):
        i = i + 1
    tla.make_coord(i, 0)


@tla.kernel
def statement_while_structured_carried_kernel(limit: int) -> None:
    state = (0, 1)
    while state[0] < limit:
        state = (state[0] + 1, state[1] + 2)
    tla.make_coord(state[0], state[1])


@tla.kernel
def statement_while_nested_if_kernel(limit: int) -> None:
    i = 0
    offset = 0
    while i < limit:
        if i == 0:
            offset = i + 1
        else:
            offset = i + 2
        i = i + 1
    tla.make_coord(i, offset)


@tla.kernel
def statement_while_nested_for_kernel(limit: int) -> None:
    i = 0
    offset = 0
    while i < limit:
        for j in tla.range(0, 2, 1):
            offset = j
        i = i + 1
    tla.make_coord(i, offset)


@tla.kernel
def statement_while_contains_cube_region_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        with tla.cube():
            tla.make_coord(i, 0)
        i = i + 1


@tla.kernel
def static_false_while_kernel() -> None:
    while False:
        tla.make_coord(99, 0)
    tla.make_coord(1, 0)


@tla.kernel
def const_expr_while_kernel(limit: tla.Constexpr[int]) -> None:
    i = 0
    while tla.const_expr(i < limit):
        tla.make_coord(i, 0)
        i = i + 1


@tla.kernel
def imported_const_expr_while_kernel(limit: tla.Constexpr[int]) -> None:
    i = 0
    while const_expr(i < limit):
        tla.make_coord(i, 0)
        i = i + 1


@tla.kernel
def bad_statement_while_break_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        break


@tla.kernel
def bad_statement_while_continue_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        continue


@tla.kernel
def bad_statement_while_return_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        return


@tla.kernel
def bad_statement_while_raise_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        raise RuntimeError("bad")


@tla.kernel
def bad_statement_while_new_value_used_after_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        value = i
        i = i + 1
    tla.make_coord(value, 0)


@tla.kernel
def bad_statement_while_type_mismatch_kernel(limit: int) -> None:
    i = 0
    while i < limit:
        i = True
    tla.make_coord(i, 0)


@tla.kernel
def bad_statement_while_structure_mismatch_kernel(limit: int) -> None:
    state = (0, 1)
    while state[0] < limit:
        state = (state[0] + 1,)
    tla.make_coord(state[0], 0)


@tla.kernel
def bad_statement_while_custom_class_type_mismatch_kernel(limit: int) -> None:
    state = CustomWhileState(0, 1)
    while state.coord < limit:
        state = CustomWhileState(True, state.offset + 1)
    tla.make_coord(state.coord, state.offset)


@tla.kernel
def bad_statement_while_active_closure_call_kernel(limit: int) -> None:
    def helper(value: int) -> None:
        tla.make_coord(value, 0)

    i = 0
    while i < limit:
        helper(i)
        i = i + 1


@tla.kernel
def statement_while_active_object_method_call_kernel(limit: int) -> None:
    values = [0, 1]
    i = 0
    while i < limit:
        values.reverse()
        values = [i, i + 1]
        i = i + 1
    tla.make_coord(values[0], values[1])


@tla.kernel
def bad_statement_while_active_object_method_structure_kernel(limit: int) -> None:
    values = [0]
    i = 0
    while i < limit:
        values.append(i)
        i = i + 1
    tla.make_coord(values[0], 0)


def test_while_execute_dynamic_emits_scf_while() -> None:
    with runtime_mod._eager_capture() as state:

        def before(i: Any) -> list[Any]:
            return [i < 4, [i]]

        def after(i: Any) -> list[Any]:
            return [i + 1]

        result = ast_decorators_mod._while_execute_dynamic(
            before,
            after,
            0,
            carried_names=("i",),
        )
        tla.make_coord(result, 0)

    mlir = str(state.module)
    assert "scf.while" in mlir
    assert "scf.condition" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_while_carried_index_lowers_to_scf_while() -> None:
    mlir = statement_while_carried_index_kernel.dump_mlir(type_args=(4,))
    assert "scf.while" in mlir
    assert "scf.condition" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_while_compound_bool_condition_is_lowered_once() -> None:
    mlir = statement_while_compound_bool_kernel.dump_mlir(type_args=(4,))
    assert "scf.while" in mlir
    assert mlir.count("arith.cmpi") == 5
    assert mlir.count("arith.andi") == 4


def test_statement_while_structured_carried_value_lowers_to_scf_while() -> None:
    mlir = statement_while_structured_carried_kernel.dump_mlir(type_args=(4,))
    assert "scf.while" in mlir
    assert "scf.condition" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_while_supports_nested_dynamic_if() -> None:
    mlir = statement_while_nested_if_kernel.dump_mlir(type_args=(4,))
    assert "scf.while" in mlir
    assert "scf.if" in mlir
    assert "scf.condition" in mlir


def test_statement_while_supports_nested_dynamic_for() -> None:
    mlir = statement_while_nested_for_kernel.dump_mlir(type_args=(4,))
    assert "scf.while" in mlir
    assert "scf.for" in mlir
    assert "scf.condition" in mlir


def test_statement_while_contains_cube_region() -> None:
    mlir = statement_while_contains_cube_region_kernel.dump_mlir(type_args=(4,))
    assert "tla.cube" in mlir
    assert "scf.while" in mlir
    assert "scf.condition" in mlir


def test_static_false_while_stays_python_control_flow() -> None:
    mlir = static_false_while_kernel.dump_mlir()
    assert "scf.while" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<99,0>" not in mlir


def test_const_expr_while_stays_python_control_flow() -> None:
    mlir = const_expr_while_kernel.dump_mlir(type_args=(2,))
    assert "scf.while" not in mlir
    assert "!tla.coord<0,0>" in mlir
    assert "!tla.coord<1,0>" in mlir


def test_imported_const_expr_while_stays_python_control_flow() -> None:
    mlir = imported_const_expr_while_kernel.dump_mlir(type_args=(2,))
    assert "scf.while" not in mlir
    assert "!tla.coord<0,0>" in mlir
    assert "!tla.coord<1,0>" in mlir


def test_statement_while_rejects_break() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla while"):
        _ = bad_statement_while_break_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_continue() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla while"):
        _ = bad_statement_while_continue_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_return() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla while"):
        _ = bad_statement_while_return_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_raise() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla while"):
        _ = bad_statement_while_raise_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_new_value_used_after_loop() -> None:
    with pytest.raises(SyntaxError, match="initialized before the while"):
        _ = bad_statement_while_new_value_used_after_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_mismatched_carried_type() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="expected index"):
        _ = bad_statement_while_type_mismatch_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_mismatched_carried_structure() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'state'.*structure"):
        _ = bad_statement_while_structure_mismatch_kernel.dump_mlir(type_args=(4,))


def test_statement_while_rejects_custom_class_type_mismatch_by_leaf_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"state\[0\]"):
        _ = bad_statement_while_custom_class_type_mismatch_kernel.dump_mlir(
            type_args=(4,)
        )


def test_statement_while_rejects_active_closure_call() -> None:
    with pytest.raises(SyntaxError, match="active local callable"):
        _ = bad_statement_while_active_closure_call_kernel.dump_mlir(type_args=(4,))


def test_statement_while_active_object_method_call_lowers_as_carried_value() -> None:
    mlir = statement_while_active_object_method_call_kernel.dump_mlir(type_args=(4,))
    assert "scf.while" in mlir
    assert "scf.condition" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_while_rejects_active_object_method_structure_change() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'values'.*structure"):
        _ = bad_statement_while_active_object_method_structure_kernel.dump_mlir(
            type_args=(4,)
        )


def test_while_execute_dynamic_rejects_condition_region_structure_mismatch() -> None:
    with runtime_mod._eager_capture():

        def before(state: Any) -> list[Any]:
            return [state[0] < 4, [(state[0],)]]

        def after(state: Any) -> list[Any]:
            return [(state[0] + 1, state[1] + 2)]

        with pytest.raises(tla.TlaCoreAPIError, match="condition.*structure"):
            ast_decorators_mod._while_execute_dynamic(
                before,
                after,
                (0, 1),
                carried_names=("state",),
            )


def test_while_execute_dynamic_rejects_condition_region_type_mismatch() -> None:
    with runtime_mod._eager_capture():

        def before(state: Any) -> list[Any]:
            return [state[0] < 4, [(True, state[1])]]

        def after(state: Any) -> list[Any]:
            return [(state[0] + 1, state[1] + 2)]

        with pytest.raises(tla.TlaCoreAPIError, match="condition.*expected index"):
            ast_decorators_mod._while_execute_dynamic(
                before,
                after,
                (0, 1),
                carried_names=("state",),
            )


def test_while_executor_emits_scf_while_with_structured_carried_value() -> None:
    with runtime_mod._eager_capture() as state:

        def before(loop_state: Any) -> list[Any]:
            return [loop_state[0] < 4, [loop_state]]

        def after(loop_state: Any) -> list[Any]:
            return [(loop_state[0] + 1, loop_state[1] + 2)]

        result = while_executor(
            before,
            after,
            write_args=[(0, 1)],
            full_write_args_count=1,
            write_args_names=["loop_state"],
        )
        tla.make_coord(result[0], result[1])

    mlir = str(state.module)
    assert "scf.while" in mlir
    assert "scf.condition" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir
