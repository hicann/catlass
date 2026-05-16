"""Runtime executors for Tla AST-preprocessed frontend control flow."""

from __future__ import annotations

import builtins
from typing import Any, Callable

from . import runtime as _runtime
from .base_dsl.ast_helpers import FrontendRange
from .base_dsl.utils import tree_utils

TlaCoreAPIError = _runtime.TlaCoreAPIError
_BoolExpr = _runtime._BoolExpr
_IndexExpr = _runtime._IndexExpr
_capture_caller_location = _runtime._capture_caller_location
_coerce_bool_value = _runtime._coerce_bool_value
_coerce_index_value = _runtime._coerce_index_value
_const_i1 = _runtime._const_i1
_resolve_frontend_bound_value = _runtime._resolve_frontend_bound_value


def _loop_unroll_attr(**kwargs: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    def to_mlir_attr(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return f"{value} : i32"
        raise TlaCoreAPIError(f"Unsupported loop unroll value: {type(value).__name__}")

    cfg = {key: to_mlir_attr(kwargs[key]) for key in ("count", "full") if key in kwargs}
    if kwargs.get("count") == 1:
        cfg["disable"] = "true"
    unroll = "<" + ", ".join(f"{key} = {value}" for key, value in cfg.items()) + ">"
    return mlir_ir.Attribute.parse(f"#llvm.loop_annotation<unroll = {unroll}>")


def _attach_range_loop_attrs(for_op: Any, range_value: FrontendRange) -> None:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    if range_value.unroll_full:
        for_op.attributes["loop_annotation"] = _loop_unroll_attr(full=True)
    elif range_value.unroll != -1:
        for_op.attributes["loop_annotation"] = _loop_unroll_attr(
            count=range_value.unroll
        )
    if range_value.prefetch_stages is not None:
        i32 = mlir_ir.IntegerType.get_signless(32)
        for_op.attributes["cutlass.pipelining"] = mlir_ir.IntegerAttr.get(
            i32, range_value.prefetch_stages
        )


class ScfGenerator:
    """Shared SCF construction helper for AST-preprocessed Tla control flow."""

    @staticmethod
    def _normalize_region_result_to_list(region_result: Any) -> list[Any]:
        if region_result is None:
            return []
        if isinstance(region_result, list):
            return region_result
        return [region_result]

    def scf_execute_dynamic(
        self,
        *,
        op_type_name: str,
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
        mix_iter_arg_names: list[str] | tuple[str, ...],
        create_op_func: Callable[[list[Any]], Any],
        region_builders: list[
            Callable[
                [
                    Any,
                    list[Any],
                    list[Any],
                    tuple[list[Any], list[str]],
                    list[Any] | tuple[Any, ...],
                    int,
                ],
                list[Any] | tuple[Any, ...],
            ]
        ],
        initial_ir_values: list[Any] | None = None,
        initial_pytree_def: tuple[list[Any], list[str]] | None = None,
        initial_ir_types: list[Any] | None = None,
        block_term_op_builder: dict[Callable[..., Any], Callable[..., None]]
        | None = None,
    ) -> Any:
        from mlir import ir as mlir_ir  # type: ignore[assignment]
        from mlir.dialects import scf  # type: ignore[import-not-found]
        from . import core_api as _core_api

        if initial_ir_values is None or initial_pytree_def is None:
            ir_values, pytree_def = _core_api.unpack_to_irvalue(
                mix_iter_args,
                op_type_name,
                full_write_args_count,
                mix_iter_arg_names,
            )
        else:
            ir_values = initial_ir_values
            pytree_def = initial_pytree_def
        expected_types = (
            initial_ir_types
            if initial_ir_types is not None
            else [value.type for value in ir_values]
        )
        custom_terminators = block_term_op_builder or {}

        def unpack_and_validate_region_values(region_result: Any) -> list[Any]:
            region_result_list = self._normalize_region_result_to_list(region_result)
            region_values, yield_pytree_def = _core_api.unpack_to_irvalue(
                region_result_list,
                op_type_name,
                full_write_args_count,
                mix_iter_arg_names,
            )
            if pytree_def[0] != yield_pytree_def[0]:
                if len(region_values) != len(expected_types):
                    name = _dynamic_region_structure_name(
                        pytree_def, yield_pytree_def, mix_iter_arg_names
                    )
                    raise TlaCoreAPIError(
                        f"Dynamic {op_type_name} region result {name} has "
                        "incompatible carried value structure"
                    )
                for actual, expected_type in zip(
                    region_values, expected_types, strict=False
                ):
                    if str(actual.type) != str(expected_type):
                        name = _dynamic_region_leaf_name(yield_pytree_def, 0)
                        raise TlaCoreAPIError(
                            f"Dynamic {op_type_name} region result {name} has type "
                            f"{actual.type}, expected {expected_type}"
                        )
            else:
                for index, (actual, expected_type) in enumerate(
                    zip(region_values, expected_types, strict=False)
                ):
                    if str(actual.type) != str(expected_type):
                        name = _dynamic_region_leaf_name(yield_pytree_def, index)
                        raise TlaCoreAPIError(
                            f"Dynamic {op_type_name} region result {name} has type "
                            f"{actual.type}, expected {expected_type}"
                        )
            return region_values

        op = create_op_func(ir_values)
        for index, builder in enumerate(region_builders):
            region = op.regions[index]
            block = region.blocks[0]
            with mlir_ir.InsertionPoint(block):
                region_result = builder(
                    op,
                    list(block.arguments),
                    ir_values,
                    pytree_def,
                    mix_iter_args,
                    full_write_args_count,
                )
                custom_terminator = custom_terminators.get(builder)
                if custom_terminator is not None:
                    custom_terminator(
                        region_result,
                        pytree_def,
                        expected_types,
                        mix_iter_args,
                        full_write_args_count,
                    )
                else:
                    region_values = unpack_and_validate_region_values(region_result)
                    scf.YieldOp(region_values)
        final_results = _core_api.pack_from_irvalue(
            list(op.results), pytree_def, mix_iter_args, full_write_args_count
        )
        return tree_utils.return_carried_values(final_results)


def _internal_frontend_bool_and(*values: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    if not values:
        return True
    if all(isinstance(value, bool) for value in values):
        return all(values)
    current = _coerce_bool_value(values[0])
    for value in values[1:]:
        rhs = _coerce_bool_value(value)
        op = mlir_ir.Operation.create(
            "arith.andi", operands=[current, rhs], results=[current.type]
        )
        current = op.results[0]
    return _BoolExpr(current)


def _internal_frontend_for(
    range_value: Any,
    body_fn: Callable[..., Any],
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
) -> Any:
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    if not isinstance(range_value, FrontendRange):
        raise TlaCoreAPIError(
            "for loops over tla.range require frontend range metadata"
        )
    carried_names_tuple = tree_utils.normalize_frontend_if_carried_names(
        carried_names, len(carried_values)
    )
    carried_specs = [
        tree_utils.frontend_if_tree_spec(value) for value in carried_values
    ]
    _, carried_pytree_def = _core_api.unpack_to_irvalue(
        carried_values, "for", len(carried_values), carried_names_tuple
    )
    carried_leaf_names = carried_pytree_def[1]

    mlir_loc = _capture_caller_location()
    start = _coerce_index_value(range_value.start)
    end = _coerce_index_value(range_value.end)
    step = _coerce_index_value(range_value.step)

    generator = ScfGenerator()

    def create_for_op(ir_values: list[Any]) -> Any:
        for_op = scf.ForOp(start, end, step, ir_values, loc=mlir_loc)
        _attach_range_loop_attrs(for_op, range_value)
        return for_op

    def build_body(
        _op: Any,
        block_args: list[Any],
        _ir_values: list[Any],
        _pytree_def: tuple[list[Any], list[str]],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        _full_write_args_count: int,
    ) -> list[Any]:
        carried_args = _core_api.pack_from_irvalue(
            block_args[1:], carried_pytree_def, carried_values, len(carried_values)
        )
        body_result = body_fn(_IndexExpr(block_args[0]), *carried_args)
        return tree_utils.extract_frontend_if_yields(
            body_result,
            carried_values,
            carried_specs,
            carried_names_tuple,
            carried_leaf_names,
            "for",
        )

    return generator.scf_execute_dynamic(
        op_type_name="for",
        mix_iter_args=carried_values,
        full_write_args_count=len(carried_values),
        mix_iter_arg_names=carried_names_tuple,
        create_op_func=create_for_op,
        region_builders=[build_body],
    )


def _while_execute_dynamic(
    while_before_block: Callable[..., Any],
    while_after_block: Callable[..., Any],
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
    full_write_args_count: int | None = None,
) -> Any:
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    carried_count = (
        len(carried_values)
        if full_write_args_count is None or full_write_args_count == 0
        else full_write_args_count
    )
    carried_names_tuple = tree_utils.normalize_frontend_if_carried_names(
        carried_names, len(carried_values)
    )
    initial_ir_values, carried_pytree_def = _core_api.unpack_to_irvalue(
        carried_values, "while", carried_count, carried_names_tuple
    )
    expected_types = [value.type for value in initial_ir_values]

    def create_while_op(ir_values: list[Any]) -> Any:
        while_op = scf.WhileOp([value.type for value in ir_values], ir_values)
        while_op.before.blocks.append(*[value.type for value in ir_values])
        while_op.after.blocks.append(*[value.type for value in ir_values])
        return while_op

    def before_builder(
        _op: Any,
        block_args: list[Any],
        _ir_values: list[Any],
        pytree_def: tuple[list[Any], list[str]],
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> Any:
        before_args = _core_api.pack_from_irvalue(
            block_args, pytree_def, mix_iter_args, full_write_args_count
        )
        return while_before_block(*before_args)

    def before_terminator(
        cond_and_results: Any,
        pytree_def: tuple[list[Any], list[str]],
        expected_types: list[Any],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> None:
        cond, before_results = _normalize_while_before_result(cond_and_results)
        ir_cond = _coerce_bool_value(cond)
        result_values, result_pytree_def = _core_api.unpack_to_irvalue(
            before_results,
            "while",
            full_write_args_count,
            carried_names_tuple,
        )
        _validate_dynamic_while_results(
            result_values,
            result_pytree_def,
            pytree_def,
            expected_types,
            "condition",
        )
        scf.ConditionOp(ir_cond, result_values)

    def after_builder(
        _op: Any,
        block_args: list[Any],
        _ir_values: list[Any],
        pytree_def: tuple[list[Any], list[str]],
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> Any:
        after_args = _core_api.pack_from_irvalue(
            block_args, pytree_def, mix_iter_args, full_write_args_count
        )
        return while_after_block(*after_args)

    return ScfGenerator().scf_execute_dynamic(
        op_type_name="while",
        mix_iter_args=carried_values,
        full_write_args_count=carried_count,
        mix_iter_arg_names=carried_names_tuple,
        create_op_func=create_while_op,
        region_builders=[before_builder, after_builder],
        initial_ir_values=initial_ir_values,
        initial_pytree_def=carried_pytree_def,
        initial_ir_types=expected_types,
        block_term_op_builder={before_builder: before_terminator},
    )


def _normalize_while_before_result(result: Any) -> tuple[Any, list[Any]]:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise TlaCoreAPIError(
            "Dynamic while condition block must return (condition, carried_values)"
        )
    cond = result[0]
    carried = result[1]
    if carried is None:
        return cond, []
    if isinstance(carried, list):
        return cond, carried
    if isinstance(carried, tuple):
        return cond, list(carried)
    return cond, [carried]


def _validate_dynamic_while_results(
    actual_values: list[Any],
    actual_pytree_def: tuple[list[Any], list[str]],
    expected_pytree_def: tuple[list[Any], list[str]],
    expected_types: list[Any],
    region_name: str,
) -> None:
    if actual_pytree_def[0] != expected_pytree_def[0]:
        name = _dynamic_region_structure_name(
            expected_pytree_def, actual_pytree_def, tuple()
        )
        raise TlaCoreAPIError(
            f"Dynamic while {region_name} region result {name} has incompatible "
            "carried value structure"
        )
    if len(actual_values) != len(expected_types):
        raise TlaCoreAPIError(
            f"Dynamic while {region_name} region produced "
            f"{len(actual_values)} value(s), expected {len(expected_types)}"
        )
    for index, (actual, expected_type) in enumerate(
        zip(actual_values, expected_types, strict=False)
    ):
        if str(actual.type) != str(expected_type):
            name = _dynamic_region_leaf_name(actual_pytree_def, index)
            raise TlaCoreAPIError(
                f"Dynamic while {region_name} region result {name} has type "
                f"{actual.type}, expected {expected_type}"
            )


def _dynamic_region_structure_name(
    expected_pytree_def: tuple[list[Any], list[str]],
    actual_pytree_def: tuple[list[Any], list[str]],
    carried_names: list[str] | tuple[str, ...],
) -> str:
    expected_specs = expected_pytree_def[0]
    actual_specs = actual_pytree_def[0]
    for index, (expected, actual) in enumerate(
        zip(expected_specs, actual_specs, strict=False)
    ):
        if expected != actual:
            if index < len(carried_names):
                return repr(carried_names[index])
            expected_leaf_names = expected_pytree_def[1]
            if expected_leaf_names:
                return repr(expected_leaf_names[0].split("[", 1)[0].split(".", 1)[0])
            actual_leaf_names = actual_pytree_def[1]
            if actual_leaf_names:
                return repr(actual_leaf_names[0].split("[", 1)[0].split(".", 1)[0])
            return f"at index {index}"
    return "values"


def _dynamic_region_leaf_name(
    pytree_def: tuple[list[Any], list[str]], index: int
) -> str:
    leaf_names = pytree_def[1]
    if index < len(leaf_names):
        return repr(leaf_names[index])
    return f"at index {index}"


def _internal_frontend_bool_or(*values: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    if not values:
        return False
    if all(isinstance(value, bool) for value in values):
        return any(values)
    current = _coerce_bool_value(values[0])
    for value in values[1:]:
        rhs = _coerce_bool_value(value)
        op = mlir_ir.Operation.create(
            "arith.ori", operands=[current, rhs], results=[current.type]
        )
        current = op.results[0]
    return _BoolExpr(current)


def _internal_frontend_bool_not(value: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    if isinstance(value, bool):
        return not value
    operand = _coerce_bool_value(value)
    one = _const_i1(1)
    op = mlir_ir.Operation.create(
        "arith.xori", operands=[operand, one], results=[operand.type]
    )
    return _BoolExpr(op.results[0])


def _internal_frontend_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    return _BoolExpr(_coerce_bool_value(value))


def _internal_frontend_any(iterable: Any) -> Any:
    values = list(iterable)
    if not values:
        return False
    if all(isinstance(value, bool) for value in values):
        return any(values)
    return _internal_frontend_bool_or(*values)


def _internal_frontend_all(iterable: Any) -> Any:
    values = list(iterable)
    if not values:
        return True
    if all(isinstance(value, bool) for value in values):
        return all(values)
    return _internal_frontend_bool_and(*values)


def _internal_frontend_min(*values: Any) -> Any:
    return _internal_frontend_minmax("min", *values)


def _internal_frontend_max(*values: Any) -> Any:
    return _internal_frontend_minmax("max", *values)


def _internal_frontend_minmax(kind: str, *values: Any) -> Any:
    flat_values = _flatten_minmax_values(values)
    if not flat_values:
        raise TlaCoreAPIError(f"{kind}() expected at least one argument")
    if len(flat_values) == 1:
        return flat_values[0]
    if not any(_is_dynamic_index_like(value) for value in flat_values):
        op = builtins.min if kind == "min" else builtins.max
        return op(*flat_values)

    current = flat_values[0]
    for value in flat_values[1:]:
        current = _select_minmax_index(kind, current, value)
    return current


def _flatten_minmax_values(values: tuple[Any, ...]) -> list[Any]:
    if len(values) == 1 and not _is_dynamic_index_like(values[0]):
        try:
            return list(values[0])
        except TypeError:
            return [values[0]]
    flat_values: list[Any] = []
    for value in values:
        if _is_dynamic_index_like(value):
            flat_values.append(value)
            continue
        if isinstance(value, (str, bytes)):
            flat_values.append(value)
            continue
        try:
            flat_values.extend(value)
        except TypeError:
            flat_values.append(value)
    return flat_values


def _is_dynamic_index_like(value: Any) -> bool:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    resolved = _resolve_frontend_bound_value(value)
    return isinstance(resolved, mlir_ir.Value) or isinstance(value, mlir_ir.Value)


def _select_minmax_index(kind: str, left: Any, right: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]
    from mlir.dialects import arith  # type: ignore[import-not-found]

    lhs = _coerce_index_value(left)
    rhs = _coerce_index_value(right)
    predicate = arith.CmpIPredicate.slt if kind == "min" else arith.CmpIPredicate.sgt
    cond = arith.CmpIOp(predicate, lhs, rhs).result
    op = mlir_ir.Operation.create(
        "arith.select", operands=[cond, lhs, rhs], results=[lhs.type]
    )
    return _runtime._IndexExpr(op.results[0])


def _internal_frontend_compare(
    left: Any, comparators: tuple[Any, ...], ops: tuple[str, ...]
) -> Any:
    if len(comparators) != len(ops):
        raise TlaCoreAPIError(
            "Comparison metadata mismatch: "
            f"{len(comparators)} comparator(s), {len(ops)} operator(s)"
        )
    current = left
    results: list[Any] = []
    for op, comparator in zip(ops, comparators, strict=False):
        results.append(_internal_frontend_compare_pair(current, comparator, op))
        current = comparator
    if not results:
        return True
    if all(isinstance(result, bool) for result in results):
        return all(results)
    return _internal_frontend_bool_and(*results)


def _internal_frontend_compare_pair(left: Any, right: Any, op: str) -> Any:
    from mlir.dialects import arith  # type: ignore[import-not-found]

    if op == "==":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.eq, op)
    if op == "!=":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.ne, op)
    if op == "<":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.slt, op)
    if op == "<=":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.sle, op)
    if op == ">":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.sgt, op)
    if op == ">=":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.sge, op)
    if op == "is":
        return left is right
    if op == "is not":
        return left is not right
    if op == "in":
        return left in right
    if op == "not in":
        return left not in right
    raise TlaCoreAPIError(f"Unsupported comparison operator: {op}")


def _compare_index_or_python(left: Any, right: Any, predicate: Any, op: str) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]
    from mlir.dialects import arith  # type: ignore[import-not-found]

    lhs = _resolve_frontend_bound_value(left)
    rhs = _resolve_frontend_bound_value(right)
    if lhs is None:
        lhs = left
    if rhs is None:
        rhs = right
    if isinstance(lhs, mlir_ir.Value) or isinstance(rhs, mlir_ir.Value):
        lhs_index = _coerce_index_value(lhs)
        rhs_index = _coerce_index_value(rhs)
        return _BoolExpr(arith.CmpIOp(predicate, lhs_index, rhs_index).result)
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise TlaCoreAPIError(f"Unsupported comparison operator: {op}")


def _internal_frontend_if(
    condition: Any,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any] | None,
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
) -> Any:
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    carried_names_tuple = tree_utils.normalize_frontend_if_carried_names(
        carried_names, len(carried_values)
    )
    carried_specs = [
        tree_utils.frontend_if_tree_spec(value) for value in carried_values
    ]
    if isinstance(condition, bool):
        selected = then_fn if condition else else_fn
        if selected is None:
            return tree_utils.return_carried_values(carried_values)
        result = selected(*carried_values)
        return tree_utils.normalize_frontend_if_result_with_names(
            result, carried_values, carried_names_tuple, carried_specs
        )

    cond = _coerce_bool_value(condition)
    carried_mlir, carried_pytree_def = _core_api.unpack_to_irvalue(
        carried_values, "if", len(carried_values), carried_names_tuple
    )
    carried_leaf_names = carried_pytree_def[1]
    has_else = else_fn is not None or bool(carried_mlir)

    def create_if_op(ir_values: list[Any]) -> Any:
        return scf.IfOp(cond, [value.type for value in ir_values], hasElse=has_else)

    def then_builder(
        _op: Any,
        _block_args: list[Any],
        ir_values: list[Any],
        pytree_def: tuple[list[Any], list[str]],
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> list[Any]:
        then_args = _core_api.pack_from_irvalue(
            ir_values, pytree_def, mix_iter_args, full_write_args_count
        )
        then_result = then_fn(*then_args)
        return tree_utils.extract_frontend_if_yields(
            then_result,
            carried_values,
            carried_specs,
            carried_names_tuple,
            carried_leaf_names,
            "then",
        )

    region_builders: list[
        Callable[
            [
                Any,
                list[Any],
                list[Any],
                tuple[list[Any], list[str]],
                list[Any] | tuple[Any, ...],
                int,
            ],
            list[Any] | tuple[Any, ...],
        ]
    ] = [then_builder]
    if has_else:

        def else_builder(
            _op: Any,
            _block_args: list[Any],
            ir_values: list[Any],
            pytree_def: tuple[list[Any], list[str]],
            mix_iter_args: list[Any] | tuple[Any, ...],
            full_write_args_count: int,
        ) -> list[Any]:
            if else_fn is None:
                return ir_values
            else_args = _core_api.pack_from_irvalue(
                ir_values, pytree_def, mix_iter_args, full_write_args_count
            )
            else_result = else_fn(*else_args)
            return tree_utils.extract_frontend_if_yields(
                else_result,
                carried_values,
                carried_specs,
                carried_names_tuple,
                carried_leaf_names,
                "else",
            )

        region_builders.append(else_builder)

    return ScfGenerator().scf_execute_dynamic(
        op_type_name="if",
        mix_iter_args=carried_values,
        full_write_args_count=len(carried_values),
        mix_iter_arg_names=carried_names_tuple,
        create_op_func=create_if_op,
        region_builders=region_builders,
    )


def _internal_frontend_if_expr(
    condition: Any, true_fn: Callable[[], Any], false_fn: Callable[[], Any]
) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    if isinstance(condition, bool):
        return true_fn() if condition else false_fn()

    cond = _coerce_bool_value(condition)

    execution_region = scf.ExecuteRegionOp(result=[])
    execution_region.region.blocks.append()
    with mlir_ir.InsertionPoint(execution_region.region.blocks[0]):
        true_probe = true_fn()
        true_mlir, result_pytree_def = _core_api.unpack_to_irvalue(
            [true_probe], "if expression", 1, ["if expression"]
        )
        result_spec = result_pytree_def[0][0]
        true_leaf_names = result_pytree_def[1]
        false_probe = false_fn()
        false_mlir, false_pytree_def = _core_api.unpack_to_irvalue(
            [false_probe], "if expression", 1, ["if expression"]
        )
        false_spec = false_pytree_def[0][0]
        false_leaf_names = false_pytree_def[1]
        _validate_if_expr_branch(
            false_mlir,
            false_spec,
            false_leaf_names,
            result_spec,
            true_mlir,
            true_leaf_names,
            "else",
        )
        result_types = [value.type for value in true_mlir]
    execution_region.operation.erase()

    def create_if_op(_ir_values: list[Any]) -> Any:
        return scf.IfOp(cond, result_types, hasElse=True)

    def then_builder(
        _op: Any,
        _block_args: list[Any],
        _ir_values: list[Any],
        _pytree_def: tuple[list[Any], list[str]],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        _full_write_args_count: int,
    ) -> list[Any]:
        true_result = true_fn()
        true_values, true_pytree_def = _core_api.unpack_to_irvalue(
            [true_result], "if expression", 1, ["if expression"]
        )
        true_spec = true_pytree_def[0][0]
        true_names = true_pytree_def[1]
        _validate_if_expr_branch(
            true_values,
            true_spec,
            true_names,
            result_spec,
            true_mlir,
            true_leaf_names,
            "then",
        )
        return true_values

    def else_builder(
        _op: Any,
        _block_args: list[Any],
        _ir_values: list[Any],
        _pytree_def: tuple[list[Any], list[str]],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        _full_write_args_count: int,
    ) -> list[Any]:
        false_result = false_fn()
        false_values, false_pytree_def = _core_api.unpack_to_irvalue(
            [false_result], "if expression", 1, ["if expression"]
        )
        false_spec = false_pytree_def[0][0]
        false_names = false_pytree_def[1]
        _validate_if_expr_branch(
            false_values,
            false_spec,
            false_names,
            result_spec,
            true_mlir,
            true_leaf_names,
            "else",
        )
        return false_values

    return ScfGenerator().scf_execute_dynamic(
        op_type_name="if",
        mix_iter_args=[true_probe],
        full_write_args_count=1,
        mix_iter_arg_names=["if expression"],
        create_op_func=create_if_op,
        region_builders=[then_builder, else_builder],
        initial_ir_values=[None] * len(result_types),
        initial_pytree_def=result_pytree_def,
        initial_ir_types=result_types,
    )


def _validate_if_expr_branch(
    actual_values: list[Any],
    actual_spec: tree_utils.FrontendIfTreeSpec,
    actual_names: list[str],
    expected_spec: tree_utils.FrontendIfTreeSpec,
    expected_values: list[Any],
    expected_names: list[str],
    branch_name: str,
) -> None:
    if actual_spec != expected_spec:
        raise TlaCoreAPIError(
            f"Conditional expression {branch_name} branch has incompatible structure"
        )
    if len(actual_values) != len(expected_values):
        raise TlaCoreAPIError(
            f"Conditional expression {branch_name} branch returned "
            f"{len(actual_values)} value(s), expected {len(expected_values)}"
        )
    for index, (actual, expected) in enumerate(zip(actual_values, expected_values)):
        if str(actual.type) != str(expected.type):
            leaf_name = actual_names[index] if index < len(actual_names) else None
            if leaf_name is None and index < len(expected_names):
                leaf_name = expected_names[index]
            suffix = f" for {leaf_name!r}" if leaf_name is not None else ""
            raise TlaCoreAPIError(
                f"Conditional expression {branch_name} branch result{suffix} has "
                f"type {actual.type}, expected {expected.type}"
            )
