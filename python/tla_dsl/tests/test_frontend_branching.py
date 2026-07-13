from dataclasses import dataclass
import inspect
from typing import Any

import pytest

import catlass as tla
import catlass.tla_ast_decorators as ast_decorators_mod
import catlass.core_api as core_api_mod
import catlass.runtime as runtime_mod

const_expr = tla.const_expr


@dataclass
class BranchState:
    coord: Any
    offset: Any


@dataclass(frozen=True)
class FrozenBranchState:
    coord: Any
    offset: Any


@dataclass
class OtherBranchState:
    coord: Any
    offset: Any


class CustomBranchState:
    def __init__(self, coord: Any, offset: Any) -> None:
        self.coord = coord
        self.offset = offset

    def __extract_mlir_values__(self) -> list[Any]:
        return [
            core_api_mod._as_branch_value(self.coord),
            core_api_mod._as_branch_value(self.offset),
        ]

    def __new_from_mlir_values__(self, values: list[Any]) -> "CustomBranchState":
        if len(values) != 2:
            raise ValueError(f"CustomBranchState expects 2 values, got {len(values)}")
        return CustomBranchState(
            core_api_mod._wrap_frontend_value(values[0]),
            core_api_mod._wrap_frontend_value(values[1]),
        )


def _explode() -> None:
    raise AssertionError("unreachable")


@tla.kernel
def const_expr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if tla.const_expr(flag):
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)


@tla.kernel
def imported_const_expr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if const_expr(flag):
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)


@tla.kernel
def bad_const_expr_dynamic_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if tla.const_expr(i == 0):
            tla.make_coord(i, 0)


@tla.kernel
def static_literal_if_kernel() -> None:
    if True:
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)


@tla.kernel
def inline_if_static_true_lazy_kernel() -> None:
    coord = 1 if True else _explode()
    tla.make_coord(coord, 0)


@tla.kernel
def pointer_conditional_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        loop_range = tla.range(0, 2, 1)
        for i in loop_range:
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            selected = ptr0 if i == 0 else ptr1
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.copy(local, tile)


@tla.kernel
def dynamic_mmad_init_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                tla.mmad(acc, lhs, rhs, init_c=True if outer == 0 and inner == 0 else False)

@tla.kernel
def dynamic_mmad_unit_flag_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                unit_flag = 0b11 if (outer == 1) and (inner == 1) else 0b10
                tla.mmad(acc, lhs, rhs, init_c=False, unit_flag=unit_flag)

@tla.kernel
def dynamic_mmad_initc_unit_flag_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                init_c = True if outer == 0 and inner == 0 else False
                unit_flag = 0b11 if (outer == 1) and (inner == 1) else 0b10
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=unit_flag)

@tla.kernel
def inline_if_tuple_result_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = (i, i + 1) if i == 0 else (i + 2, i + 3)
        tla.make_coord(coord[0], coord[1])


@tla.kernel
def inline_if_dataclass_result_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, i + 1) if i == 0 else BranchState(i + 2, i + 3)
        tla.make_coord(state.coord, state.offset)


@tla.kernel
def builtin_bool_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if bool(i == 0):
            tla.make_coord(i, 0)


@tla.kernel
def builtin_min_max_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        lower = min(i + 1, i + 2)
        upper = max([i, i + 3])
        tla.make_coord(lower, upper)


@tla.kernel
def builtin_name_redirect_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        choose = max
        pred = any
        coord = choose(i, i + 1)
        if pred((i == 0, i == 1)):
            tla.make_coord(coord, 0)


@tla.kernel
def bad_inline_if_structure_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = (i, i + 1) if i == 0 else i + 2
        tla.make_coord(coord[0], coord[1])


@tla.kernel
def statement_if_side_effect_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            tla.make_coord(i, 0)


@tla.kernel
def statement_if_carried_index_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        else:
            coord = i + 2
        tla.make_coord(coord, 0)


@tla.kernel
def statement_if_multiple_carried_values_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        offset = i + 1
        if i == 0:
            coord = i + 2
            offset = i + 3
        else:
            coord = i + 4
            offset = i + 5
        tla.make_coord(coord, offset)


@tla.kernel
def statement_if_tuple_assignment_carried_values_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord, offset = i, i + 1
        if i == 0:
            coord, offset = i + 2, i + 3
        else:
            coord, offset = i + 4, i + 5
        tla.make_coord(coord, offset)


@tla.kernel
def statement_if_tuple_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
        if i == 0:
            state = (i + 2, i + 3)
        else:
            state = (i + 4, i + 5)
        tla.make_coord(state[0], state[1])


@tla.kernel
def statement_if_list_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = [i, i + 1]
        if i == 0:
            state = [i + 2, i + 3]
        else:
            state = [i + 4, i + 5]
        tla.make_coord(state[0], state[1])


@tla.kernel
def statement_if_dict_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = {"coord": i, "offset": i + 1}
        if i == 0:
            state = {"coord": i + 2, "offset": i + 3}
        else:
            state = {"coord": i + 4, "offset": i + 5}
        tla.make_coord(state["coord"], state["offset"])


@tla.kernel
def statement_if_dataclass_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, (i + 1, i + 2))
        if i == 0:
            state = BranchState(i + 3, (i + 4, i + 5))
        else:
            state = BranchState(i + 6, (i + 7, i + 8))
        tla.make_coord(state.coord, state.offset[0])


@tla.kernel
def statement_if_frozen_dataclass_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = FrozenBranchState(i, i + 1)
        if i == 0:
            state = FrozenBranchState(i + 2, i + 3)
        else:
            state = FrozenBranchState(i + 4, i + 5)
        tla.make_coord(state.coord, state.offset)


@tla.kernel
def statement_if_custom_class_carried_state_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = CustomBranchState(i, i + 1)
        if i == 0:
            state = CustomBranchState(i + 2, i + 3)
        else:
            state = CustomBranchState(i + 4, i + 5)
        tla.make_coord(state.coord, state.offset)


@tla.kernel
def loop_carried_index_kernel(limit: int) -> None:
    acc = 0
    for i in tla.range(0, limit, 1):
        acc = acc + i
    tla.make_coord(acc, 0)


@tla.kernel
def loop_tuple_carried_state_kernel(limit: int) -> None:
    state = (0, 1)
    for i in tla.range(0, limit, 1):
        state = (state[0] + i, state[1] + i)
    tla.make_coord(state[0], state[1])


@tla.kernel
def loop_list_carried_state_kernel(limit: int) -> None:
    state = [0, 1]
    for i in tla.range(0, limit, 1):
        state = [state[0] + i, state[1] + i]
    tla.make_coord(state[0], state[1])


@tla.kernel
def loop_dict_carried_state_kernel(limit: int) -> None:
    state = {"coord": 0, "offset": 1}
    for i in tla.range(0, limit, 1):
        state = {"coord": state["coord"] + i, "offset": state["offset"] + i}
    tla.make_coord(state["coord"], state["offset"])


@tla.kernel
def loop_dataclass_carried_state_kernel(limit: int) -> None:
    state = BranchState(0, (1, 2))
    for i in tla.range(0, limit, 1):
        state = BranchState(state.coord + i, (state.offset[0] + i, state.offset[1]))
    tla.make_coord(state.coord, state.offset[0])


@tla.kernel
def loop_custom_class_carried_state_kernel(limit: int) -> None:
    state = CustomBranchState(0, 1)
    for i in tla.range(0, limit, 1):
        state = CustomBranchState(state.coord + i, state.offset + i)
    tla.make_coord(state.coord, state.offset)


@tla.kernel
def statement_if_implicit_else_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        tla.make_coord(coord, 0)


@tla.kernel
def statement_if_bool_op_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0 and i != 1:
            tla.make_coord(i, 0)


def _raise_if_evaluated() -> bool:
    raise AssertionError("short-circuited boolean RHS was evaluated")


@tla.kernel
def static_false_and_short_circuit_kernel() -> None:
    if False and _raise_if_evaluated():
        tla.make_coord(1, 0)


@tla.kernel
def static_true_or_short_circuit_kernel() -> None:
    if True or _raise_if_evaluated():
        tla.make_coord(1, 0)


@tla.kernel
def statement_if_ordered_comparison_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i < limit:
            tla.make_coord(i, 0)
        if i <= limit:
            tla.make_coord(i, 1)
        if i > 0:
            tla.make_coord(i, 2)
        if i >= 0:
            tla.make_coord(i, 3)


@tla.kernel
def statement_if_chained_comparison_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if 0 <= i < limit:
            tla.make_coord(i, 0)


@tla.kernel
def statement_if_any_all_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if any((i == 0, i == 1, i < limit)):
            tla.make_coord(i, 0)
        if all([i >= 0, i < limit]):
            tla.make_coord(i, 1)


@tla.kernel
def statement_if_any_all_generator_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if any(i == j for j in (0, 1)):
            tla.make_coord(i, 0)
        if all(i >= j for j in (0, 1)):
            tla.make_coord(i, 1)


@tla.kernel
def statement_if_inside_region_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        with tla.cube():
            if i == 0:
                tla.make_coord(i, 0)


@tla.kernel
def statement_if_carried_pointer_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        for i in tla.range(0, 2, 1):
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            selected = ptr0
            if i == 0:
                selected = ptr1
            else:
                selected = ptr0
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.copy(local, tile)


@tla.kernel
def statement_if_mixed_index_pointer_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        for i in tla.range(0, 2, 1):
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            coord = i
            selected = ptr0
            if i == 0:
                coord = i + 1
                selected = ptr1
            else:
                coord = i + 2
                selected = ptr0
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.make_coord(coord, 0)
            tla.copy(local, tile)


@tla.kernel
def statement_if_elif_carried_index_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        elif i == 1:
            coord = i + 2
        else:
            coord = i + 3
        tla.make_coord(coord, 0)


@tla.kernel
def statement_if_carried_tensor_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    for i in tla.range(0, 2, 1):
        left = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        right = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        selected = left
        if i == 0:
            selected = left
        else:
            selected = right
        tla.copy(selected, left)


@tla.kernel
def statement_if_mixed_index_tensor_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    for i in tla.range(0, 2, 1):
        left = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        right = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, 0))
        coord = i
        selected = left
        if i == 0:
            coord = i + 1
            selected = left
        else:
            coord = i + 2
            selected = right
        tla.make_coord(coord, 0)
        tla.copy(selected, left)


@tla.kernel
def statement_if_not_predicate_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if not (i == 0):
            tla.make_coord(i, 0)


@tla.kernel
def statement_if_augassign_carried_index_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord += 1
        tla.make_coord(coord, 0)


@tla.kernel
def statement_if_nested_function_local_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:

            def helper() -> None:
                local_only = i
                _ = local_only

            helper()
            coord = i + 1
        tla.make_coord(coord, 0)


@tla.kernel
def statement_if_nested_function_return_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:

            def helper() -> None:
                return

            helper()
            coord = i + 1
        tla.make_coord(coord, 0)


@tla.kernel
def bad_statement_if_return_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            return


@tla.kernel
def bad_statement_if_break_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            break


@tla.kernel
def bad_statement_if_continue_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            continue


@tla.kernel
def bad_statement_if_raise_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            raise ValueError("unsupported")


@tla.kernel
def bad_statement_if_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord = i + 1
        else:
            coord = True
        tla.make_coord(coord, 0)


@tla.kernel
def bad_statement_if_multiple_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        offset = i + 1
        if i == 0:
            coord = i + 2
            offset = i + 3
        else:
            coord = True
            offset = i + 4
        tla.make_coord(coord, offset)


@tla.kernel
def bad_statement_if_pytree_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
        if i == 0:
            state = (i + 2, i + 3)
        else:
            state = (True, i + 4)
        tla.make_coord(state[0], state[1])


@tla.kernel
def bad_statement_if_pytree_structure_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = (i, i + 1)
        if i == 0:
            state = (i + 2, i + 3)
        else:
            state = (i + 4,)
        tla.make_coord(state[0], 0)


@tla.kernel
def bad_statement_if_dataclass_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, i + 1)
        if i == 0:
            state = BranchState(i + 2, i + 3)
        else:
            state = BranchState(True, i + 4)
        tla.make_coord(state.coord, state.offset)


@tla.kernel
def bad_statement_if_dataclass_structure_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = BranchState(i, i + 1)
        if i == 0:
            state = BranchState(i + 2, i + 3)
        else:
            state = OtherBranchState(i + 4, i + 5)
        tla.make_coord(state.coord, state.offset)


@tla.kernel
def bad_statement_if_custom_class_type_mismatch_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        state = CustomBranchState(i, i + 1)
        if i == 0:
            state = CustomBranchState(i + 2, i + 3)
        else:
            state = CustomBranchState(True, i + 4)
        tla.make_coord(state.coord, state.offset)


@tla.kernel
def bad_statement_if_new_value_used_after_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1
        tla.make_coord(coord, 0)


@tla.kernel
def bad_statement_if_new_value_used_after_else_pass_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1
        else:
            pass
        tla.make_coord(coord, 0)


@tla.kernel
def bad_statement_if_new_value_used_after_elif_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1
        elif i == 1:
            coord = i + 2
        tla.make_coord(coord, 0)


@tla.kernel
def bad_statement_if_subscript_assignment_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        values = [i]
        if i == 0:
            values[0] = i + 1


@tla.kernel
def bad_statement_if_attribute_assignment_kernel(limit: int) -> None:
    box = object()
    for i in tla.range(0, limit, 1):
        if i == 0:
            box.value = i


@tla.kernel
def bad_statement_if_starred_assignment_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i
        if i == 0:
            coord, *rest = (i + 1, i + 2)
            _ = rest
        tla.make_coord(coord, 0)


@tla.kernel
def bad_statement_if_active_closure_call_kernel(limit: int) -> None:
    def helper(value: int) -> None:
        tla.make_coord(value, 0)

    for i in tla.range(0, limit, 1):
        if i == 0:
            helper(i)


@tla.kernel
def bad_statement_if_active_object_method_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        values = []
        if i == 0:
            values.append(i)


@tla.kernel
def statement_if_active_object_method_call_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        values = [i, i + 1]
        if i == 0:
            values.reverse()
        tla.make_coord(values[0], values[1])


def test_const_expr_true_branch_stays_static() -> None:
    mlir = const_expr_if_kernel.dump_mlir(type_args=(True,))
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<2,0>" not in mlir


def test_const_expr_false_branch_stays_static() -> None:
    mlir = const_expr_if_kernel.dump_mlir(type_args=(False,))
    assert "scf.if" not in mlir
    assert "!tla.coord<2,0>" in mlir
    assert "!tla.coord<1,0>" not in mlir


def test_imported_const_expr_if_stays_static() -> None:
    mlir = imported_const_expr_if_kernel.dump_mlir(type_args=(True,))
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<2,0>" not in mlir


def test_const_expr_rejects_dynamic_value() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.const_expr"):
        _ = bad_const_expr_dynamic_kernel.dump_mlir(type_args=(2,))


def test_static_literal_if_stays_static() -> None:
    mlir = static_literal_if_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "!tla.coord<1,0>" in mlir
    assert "!tla.coord<2,0>" not in mlir


def test_pointer_conditional_expression_lowers_to_scf_if() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 8),
            tla.Float16,
            origin_shape=tla.make_shape(16, 8),
        )
    mlir = pointer_conditional_kernel.dump_mlir(type_args=(mem,))
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_tensor_like" in mlir


def test_dynamic_mmad_init_expression_lowers_to_scf_if(compiler_tlair) -> None:
    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l0a,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.zN,
        )
        mem_b = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l0b,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.nZ,
        )
        mem_c = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float32,
            addrspace=tla.AddressSpace.l0c,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.L0Clayout,
        )
    mlir = compiler_tlair(dynamic_mmad_init_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "arith.andi" in mlir
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 2
    assert '"arith.constant"() <{value = true}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = false}> : () -> i1' in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_dynamic_mmad_unit_flag_expression_lowers_to_scf_if(compiler_tlair) -> None:
    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l0a,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.zN,
        )
        mem_b = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l0b,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.nZ,
        )
        mem_c = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float32,
            addrspace=tla.AddressSpace.l0c,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.L0Clayout,
        )
    mlir = compiler_tlair(dynamic_mmad_unit_flag_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "arith.andi" in mlir
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 2
    assert '"arith.constant"() <{value = 3 : index}> : () -> index' in mlir
    assert '"arith.constant"() <{value = 2 : index}> : () -> index' in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_dynamic_mmad_initc_unit_flag_expression_lowers_to_scf_if(compiler_tlair) -> None:
    with runtime_mod._eager_capture():
        mem_a = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l0a,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.zN,
        )
        mem_b = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float16,
            addrspace=tla.AddressSpace.l0b,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.nZ,
        )
        mem_c = tla.Tensor(
            tla.make_shape((16, 1), (16, 1)),
            tla.Float32,
            addrspace=tla.AddressSpace.l0c,
            origin_shape=tla.make_shape(16, 16),
            layout_tag=tla.arch.L0Clayout,
        )
    mlir = compiler_tlair(dynamic_mmad_initc_unit_flag_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "arith.andi" in mlir
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 4
    assert '"arith.constant"() <{value = true}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = false}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = 3 : index}> : () -> index' in mlir
    assert '"arith.constant"() <{value = 2 : index}> : () -> index' in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_inline_if_static_branch_is_lazy() -> None:
    mlir = inline_if_static_true_lazy_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "tla.make_coord" in mlir


def test_inline_if_tuple_result_lowers_to_scf_if_results() -> None:
    mlir = inline_if_tuple_result_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_inline_if_dataclass_result_lowers_to_scf_if_results() -> None:
    mlir = inline_if_dataclass_result_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_builtin_bool_predicate_lowers() -> None:
    mlir = builtin_bool_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir


def test_builtin_min_max_lower_dynamic_indexes() -> None:
    mlir = builtin_min_max_kernel.dump_mlir(type_args=(2,))
    assert "arith.cmpi" in mlir
    assert "arith.select" in mlir
    assert "tla.make_coord" in mlir


def test_builtin_name_redirects_to_dynamic_helpers() -> None:
    mlir = builtin_name_redirect_kernel.dump_mlir(type_args=(2,))
    assert "arith.ori" in mlir
    assert "arith.select" in mlir
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir


def test_inline_if_rejects_structure_mismatch() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="Conditional expression.*structure"):
        _ = bad_inline_if_structure_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_side_effect_lowers_to_scf_if() -> None:
    mlir = statement_if_side_effect_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_carried_index_lowers_to_scf_if_result() -> None:
    mlir = statement_if_carried_index_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_multiple_carried_values_lower_to_scf_if_results() -> None:
    mlir = statement_if_multiple_carried_values_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir
    assert "tla.make_coord" in mlir


def test_loop_carried_index_lowers_to_scf_for_result() -> None:
    mlir = loop_carried_index_kernel.dump_mlir(type_args=(2,))
    assert "scf.for" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir
    assert "tla.make_coord" in mlir


def test_loop_structured_carried_values_lower_to_scf_for_results() -> None:
    for kernel in (
        loop_tuple_carried_state_kernel,
        loop_list_carried_state_kernel,
        loop_dict_carried_state_kernel,
        loop_dataclass_carried_state_kernel,
        loop_custom_class_carried_state_kernel,
    ):
        mlir = kernel.dump_mlir(type_args=(2,))
        assert "scf.for" in mlir
        assert "scf.yield" in mlir
        assert "arith.addi" in mlir
        assert "tla.make_coord" in mlir


def test_statement_if_tuple_assignment_carried_values_lower() -> None:
    mlir = statement_if_tuple_assignment_carried_values_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_tuple_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_tuple_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_list_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_list_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_dict_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_dict_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_dataclass_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_dataclass_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_frozen_dataclass_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_frozen_dataclass_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_custom_class_carried_state_lowers_to_scf_if_results() -> None:
    mlir = statement_if_custom_class_carried_state_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_without_else_yields_original_carried_value() -> None:
    mlir = statement_if_implicit_else_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_predicate_bool_ops_lower() -> None:
    mlir = statement_if_bool_op_kernel.dump_mlir(type_args=(2,))
    assert "arith.andi" in mlir
    assert "scf.if" in mlir


def test_static_false_and_short_circuits_rhs() -> None:
    mlir = static_false_and_short_circuit_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "tla.make_coord" not in mlir


def test_static_true_or_short_circuits_rhs() -> None:
    mlir = static_true_or_short_circuit_kernel.dump_mlir()
    assert "scf.if" not in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_ordered_comparisons_lower() -> None:
    mlir = statement_if_ordered_comparison_kernel.dump_mlir(type_args=(2,))
    assert mlir.count("scf.if") == 4
    assert mlir.count("arith.cmpi") >= 4
    assert "tla.make_coord" in mlir


def test_statement_if_chained_comparison_lowers() -> None:
    mlir = statement_if_chained_comparison_kernel.dump_mlir(type_args=(2,))
    assert "arith.andi" in mlir
    assert mlir.count("arith.cmpi") >= 2
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_any_all_predicates_lower() -> None:
    mlir = statement_if_any_all_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.ori" in mlir
    assert "arith.andi" in mlir
    assert mlir.count("scf.if") == 2
    assert "tla.make_coord" in mlir


def test_statement_if_any_all_generator_predicates_lower() -> None:
    mlir = statement_if_any_all_generator_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.ori" in mlir
    assert "arith.andi" in mlir
    assert mlir.count("scf.if") == 2
    assert "tla.make_coord" in mlir


def test_statement_if_inside_region_lowers() -> None:
    mlir = statement_if_inside_region_kernel.dump_mlir(type_args=(2,))
    assert "tla.cube" in mlir
    assert "scf.if" in mlir


def test_statement_if_carried_pointer_lowers_to_scf_if_result() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 8),
            tla.Float16,
            origin_shape=tla.make_shape(16, 8),
        )
    mlir = statement_if_carried_pointer_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.ptr<f16, l1, 512>" in mlir
    assert "tla.make_tensor_like" in mlir
    assert "tla.copy" in mlir


def test_statement_if_mixed_index_pointer_lowers_to_scf_if_results() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 8),
            tla.Float16,
            origin_shape=tla.make_shape(16, 8),
        )
    mlir = statement_if_mixed_index_pointer_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.ptr<f16, l1, 512>" in mlir
    assert "index" in mlir
    assert "tla.make_tensor_like" in mlir
    assert "tla.copy" in mlir


def test_statement_if_elif_lowers_to_nested_scf_if() -> None:
    mlir = statement_if_elif_carried_index_kernel.dump_mlir(type_args=(2,))
    assert mlir.count("scf.if") >= 2
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_carried_tensor_lowers_to_scf_if_result() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 8),
            tla.Float16,
            origin_shape=tla.make_shape(16, 8),
        )
    mlir = statement_if_carried_tensor_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.tensor<!tla.layout<!tla.shape<16,4>" in mlir
    assert "tla.copy" in mlir


def test_statement_if_mixed_index_tensor_lowers_to_scf_if_results() -> None:
    with runtime_mod._eager_capture():
        mem = tla.Tensor(
            tla.make_shape(16, 8),
            tla.Float16,
            origin_shape=tla.make_shape(16, 8),
        )
    mlir = statement_if_mixed_index_tensor_kernel.dump_mlir(type_args=(mem,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "!tla.tensor<!tla.layout<!tla.shape<16,4>" in mlir
    assert "index" in mlir
    assert "tla.copy" in mlir


def test_statement_if_not_predicate_lowers() -> None:
    mlir = statement_if_not_predicate_kernel.dump_mlir(type_args=(2,))
    assert "arith.xori" in mlir
    assert "scf.if" in mlir


def test_statement_if_augassign_carried_index_lowers() -> None:
    mlir = statement_if_augassign_carried_index_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "arith.addi" in mlir


def test_statement_if_nested_function_local_does_not_become_carried() -> None:
    mlir = statement_if_nested_function_local_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_nested_function_return_is_not_branch_return() -> None:
    mlir = statement_if_nested_function_return_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_rejects_return() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_return_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_break() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_break_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_continue() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_continue_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_raise() -> None:
    with pytest.raises(SyntaxError, match="dynamic Tla if"):
        _ = bad_statement_if_raise_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_mismatched_carried_type() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'coord'"):
        _ = bad_statement_if_type_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_mismatched_multiple_carried_type_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'coord'"):
        _ = bad_statement_if_multiple_type_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_pytree_type_mismatch_by_leaf_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"state\[0\]"):
        _ = bad_statement_if_pytree_type_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_pytree_structure_mismatch_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'state'.*structure"):
        _ = bad_statement_if_pytree_structure_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_dataclass_type_mismatch_by_leaf_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="state\\.coord"):
        _ = bad_statement_if_dataclass_type_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_dataclass_structure_mismatch_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'state'.*structure"):
        _ = bad_statement_if_dataclass_structure_mismatch_kernel.dump_mlir(
            type_args=(2,)
        )


def test_statement_if_rejects_custom_class_type_mismatch_by_leaf_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"state\[0\]"):
        _ = bad_statement_if_custom_class_type_mismatch_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_wrong_carried_result_count_by_name() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="coord, offset"):
        ast_decorators_mod._internal_frontend_if(
            True,
            lambda coord, offset: coord,
            None,
            1,
            2,
            carried_names=("coord", "offset"),
        )


def test_statement_if_rejects_new_value_used_after_branch() -> None:
    with pytest.raises(SyntaxError, match="initialized before the if"):
        _ = bad_statement_if_new_value_used_after_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_new_value_used_after_else_pass() -> None:
    with pytest.raises(SyntaxError, match="initialized before the if"):
        _ = bad_statement_if_new_value_used_after_else_pass_kernel.dump_mlir(
            type_args=(2,)
        )


def test_statement_if_rejects_new_value_used_after_elif() -> None:
    with pytest.raises(SyntaxError, match="initialized before the if"):
        _ = bad_statement_if_new_value_used_after_elif_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_subscript_assignment() -> None:
    with pytest.raises(SyntaxError, match="assignments to local names"):
        _ = bad_statement_if_subscript_assignment_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_attribute_assignment() -> None:
    with pytest.raises(SyntaxError, match="assignments to local names"):
        _ = bad_statement_if_attribute_assignment_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_starred_assignment() -> None:
    with pytest.raises(SyntaxError, match="assignments to local names"):
        _ = bad_statement_if_starred_assignment_kernel.dump_mlir(type_args=(2,))


def test_statement_if_rejects_active_closure_call() -> None:
    with pytest.raises(SyntaxError, match="active local callable"):
        _ = bad_statement_if_active_closure_call_kernel.dump_mlir(type_args=(2,))


def test_statement_if_active_object_method_call_lowers_as_carried_value() -> None:
    mlir = statement_if_active_object_method_call_kernel.dump_mlir(type_args=(2,))
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_coord" in mlir


def test_statement_if_rejects_active_object_method_call_structure_change() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="'values'.*structure"):
        _ = bad_statement_if_active_object_method_call_kernel.dump_mlir(type_args=(2,))

@tla.kernel
def dynamic_if_bad_list_index_kernel() -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    if idx == 0:
        values[idx]


def _source_line(fn: Any, needle: str) -> int:
    source_lines, first_lineno = inspect.getsourcelines(fn.fn)
    for offset, line in enumerate(source_lines):
        if needle in line:
            return first_lineno + offset
    raise AssertionError(f"Unable to find source line containing {needle!r}")


def test_dynamic_if_then_error_reports_original_source_location() -> None:
    line = _source_line(dynamic_if_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        dynamic_if_bad_list_index_kernel.dump_mlir()
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in dynamic if then-region" in message
    assert f"{__file__}:{line}" in message
    assert "source: values[idx]" in message
    assert "list indices" in message
    assert "_IndexExpr" in message


@tla.kernel
def dynamic_if_constexpr_bool_bad_list_index_kernel(flag: tla.Constexpr[bool]) -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    if flag:
        values[idx]


def test_constexpr_if_body_error_reports_original_source_location() -> None:
    line = _source_line(dynamic_if_constexpr_bool_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        dynamic_if_constexpr_bool_bad_list_index_kernel.dump_mlir(type_args=(True,))
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in dynamic if then-region" in message
    assert f"{__file__}:{line}" in message
    assert "source: values[idx]" in message
    assert "list indices" in message
    assert "_IndexExpr" in message


@tla.kernel
def dynamic_if_expr_bad_list_index_kernel() -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    result = values[idx] if idx == 0 else 0
    tla.make_coord(result, 0)


def test_dynamic_if_expr_error_reports_original_source_location() -> None:
    line = _source_line(dynamic_if_expr_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        dynamic_if_expr_bad_list_index_kernel.dump_mlir()
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in conditional expression then-region" in message
    assert f"{__file__}:{line}" in message
    assert "source: result = values[idx] if idx == 0 else 0" in message
    assert "list indices" in message
    assert "_IndexExpr" in message

