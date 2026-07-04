"""Minimal AST rewrites used to lower structured frontend control flow."""

from __future__ import annotations

import ast
import builtins
import contextlib
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, Iterator


_INTERNAL_FOR = "__tladsl_internal_for__"
_INTERNAL_REGION = "__tladsl_internal_region__"
_INTERNAL_IF = "__tladsl_internal_if__"
_INTERNAL_IF_EXPR = "__tladsl_internal_if_expr__"
_INTERNAL_BOOL_AND = "__tladsl_internal_bool_and__"
_INTERNAL_BOOL_OR = "__tladsl_internal_bool_or__"
_INTERNAL_BOOL_NOT = "__tladsl_internal_bool_not__"
_INTERNAL_COMPARE = "__tladsl_internal_compare__"
_INTERNAL_ANY = "__tladsl_internal_any__"
_INTERNAL_ALL = "__tladsl_internal_all__"
_INTERNAL_BOOL = "__tladsl_internal_bool__"
_INTERNAL_MIN = "__tladsl_internal_min__"
_INTERNAL_MAX = "__tladsl_internal_max__"
_INTERNAL_CF_SYMBOL_CHECK = "__tladsl_internal_cf_symbol_check__"
_INTERNAL_INDEX_ADD = "__tladsl_internal_index_add__"
_INTERNAL_INDEX_SUB = "__tladsl_internal_index_sub__"
_WHILE_SELECTOR = "while_selector"
_WHILE_EXECUTOR = "while_executor"
_BUILTIN_REDIRECTS = {
    "any": _INTERNAL_ANY,
    "all": _INTERNAL_ALL,
    "bool": _INTERNAL_BOOL,
    "min": _INTERNAL_MIN,
    "max": _INTERNAL_MAX,
}


@dataclass(frozen=True)
class RegionVariableAnalysis:
    active_symbols: set[str]
    active_callables: set[str]
    assigned_by_region: tuple[set[str], ...]
    assigned_names: set[str]
    invoked_names: set[str]
    carried_names: list[str]
    full_write_args_count: int


class ScopeManager:
    """Manage frontend AST variable and callable scopes during preprocessing."""

    def __init__(self) -> None:
        self.scopes: list[set[str]] = []
        self.callables: list[set[str]] = []

    @classmethod
    def create(cls) -> "ScopeManager":
        return cls()

    def add_to_scope(self, name: str) -> None:
        if name == "_":
            return
        if not self.scopes:
            self.scopes.append(set())
        self.scopes[-1].add(name)

    def add_names_to_scope(self, names: set[str] | list[str] | tuple[str, ...]) -> None:
        for name in names:
            self.add_to_scope(name)

    def add_to_callables(self, name: str) -> None:
        if not self.callables or name == "_":
            return
        self.callables[-1].add(name)

    def get_active_symbols(self) -> set[str]:
        active: set[str] = set()
        for scope in self.scopes:
            active.update(scope)
        return active

    def get_active_callables(self) -> set[str]:
        active: set[str] = set()
        for callables in self.callables:
            active.update(callables)
        return active

    @contextlib.contextmanager
    def enter_local_scope(
        self,
        initial_names: set[str] | list[str] | tuple[str, ...] | None = None,
        initial_callables: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> Iterator[None]:
        self.scopes.append(set(initial_names or ()))
        self.callables.append(set(initial_callables or ()))
        try:
            yield
        finally:
            self.callables.pop()
            self.scopes.pop()

    @contextlib.contextmanager
    def enter_control_flow_scope(self) -> Iterator[None]:
        self.scopes.append(set())
        try:
            yield
        finally:
            self.scopes.pop()


class _FrontendControlFlowTransformer(ast.NodeTransformer):
    def __init__(self, global_symbols: dict[str, Any] | None = None) -> None:
        self._counter = 0
        self._range_alias_stack: list[set[str]] = []
        self._scope_manager = ScopeManager.create()
        self._following_loads_stack: list[set[str]] = []
        self._global_symbols = global_symbols or {}
        self._tla_range_names = _tla_function_names_from_globals(
            self._global_symbols, "range"
        )
        self._tla_range_constexpr_names = _tla_function_names_from_globals(
            self._global_symbols, "range_constexpr"
        )
        self._tla_const_expr_names = _tla_const_expr_names_from_globals(
            self._global_symbols
        )
        self._tla_module_aliases = _tla_module_aliases_from_globals(
            self._global_symbols
        )

    def _fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"__tladsl_{prefix}_{self._counter}"

    def _range_aliases(self) -> set[str]:
        if not self._range_alias_stack:
            self._range_alias_stack.append(set())
        return self._range_alias_stack[-1]

    def _local_scope(self) -> set[str]:
        return self._scope_manager.get_active_symbols()

    def _active_callables(self) -> set[str]:
        return self._scope_manager.get_active_callables()

    def _following_loads(self) -> set[str]:
        if not self._following_loads_stack:
            return set()
        return self._following_loads_stack[-1]

    def _is_static_control_flow_test(self, node: ast.AST) -> bool:
        return _is_static_python_if_test(
            node,
            self._tla_const_expr_names,
            self._tla_module_aliases,
            self._local_scope(),
        )

    def analyze_region_variables(
        self,
        *,
        construct_name: str,
        assigned_regions: list[list[ast.stmt]],
        active_call_nodes: list[ast.AST],
    ) -> RegionVariableAnalysis:
        active_symbols = set(self._local_scope())
        active_callables = set(self._active_callables())
        _reject_unsupported_dynamic_active_callable_calls(
            active_call_nodes, active_callables, construct_name
        )
        assigned_by_region = tuple(
            _assigned_names_from_statements(region) for region in assigned_regions
        )
        assigned_names: set[str] = set()
        for assigned in assigned_by_region:
            assigned_names.update(assigned)
        invoked_names = _invoked_active_names_from_statements(
            active_call_nodes, active_symbols
        )
        carried_names = sorted((assigned_names & active_symbols) | invoked_names)
        return RegionVariableAnalysis(
            active_symbols=active_symbols,
            active_callables=active_callables,
            assigned_by_region=assigned_by_region,
            assigned_names=assigned_names,
            invoked_names=invoked_names,
            carried_names=carried_names,
            full_write_args_count=len(carried_names),
        )

    def _visit_statement_list(self, body: list[ast.stmt]) -> list[ast.stmt]:
        rewritten_body: list[ast.stmt] = []
        aliases = self._range_aliases()
        for index, stmt in enumerate(body):
            self._following_loads_stack.append(
                _loaded_names_from_statements(body[index + 1 :])
            )
            try:
                rewritten = self.visit(stmt)
                stmts = rewritten if isinstance(rewritten, list) else [rewritten]
                for out_stmt in stmts:
                    if out_stmt is None:
                        continue
                    rewritten_body.append(out_stmt)
                    _update_range_aliases(
                        out_stmt,
                        aliases,
                        self._tla_range_names,
                        self._tla_module_aliases,
                        self._local_scope(),
                    )
                    self._scope_manager.add_names_to_scope(_assigned_names(out_stmt))
                    for callable_name in _callable_names(out_stmt):
                        self._scope_manager.add_to_callables(callable_name)
            finally:
                self._following_loads_stack.pop()
        return rewritten_body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._range_alias_stack.append(set())
        with self._scope_manager.enter_local_scope(_function_arg_names(node.args)):
            try:
                node.args = self.visit(node.args)
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
                node.decorator_list = [
                    self.visit(decorator) for decorator in node.decorator_list
                ]
                if node.returns is not None:
                    node.returns = self.visit(node.returns)
                return node
            finally:
                self._range_alias_stack.pop()

    def visit_For(self, node: ast.For) -> Any:
        node.iter = self.visit(node.iter)
        if _is_tla_range_constexpr_call(
            node.iter,
            self._tla_range_constexpr_names,
            self._tla_module_aliases,
            self._local_scope(),
        ):
            check_stmt = _cf_symbol_check_stmt(node.iter)
            node.iter = _builtin_range_call_from_range_constexpr(node.iter)
            if isinstance(node.target, ast.Name):
                self._scope_manager.add_to_scope(node.target.id)
            self._range_alias_stack.append(set())
            try:
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
            finally:
                self._range_alias_stack.pop()
            self._range_alias_stack.append(set())
            try:
                node.orelse = self._visit_statement_list(node.orelse)
            finally:
                self._range_alias_stack.pop()
            return [check_stmt, node]

        is_tla_range = _is_tla_range_iter(
            node.iter,
            self._range_aliases(),
            self._tla_range_names,
            self._tla_module_aliases,
            self._local_scope(),
        )
        if is_tla_range:
            if _is_tla_range_call(
                node.iter,
                self._tla_range_names,
                self._tla_module_aliases,
                self._local_scope(),
            ):
                _validate_dynamic_range_call_syntax(node.iter)
            _reject_unsupported_dynamic_for_control_flow(node)
            if node.orelse:
                raise SyntaxError("dynamic Tla for does not support for-else")
        if not is_tla_range:
            if isinstance(node.target, ast.Name):
                self._scope_manager.add_to_scope(node.target.id)
            self._range_alias_stack.append(set())
            try:
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
            finally:
                self._range_alias_stack.pop()
            self._range_alias_stack.append(set())
            try:
                node.orelse = self._visit_statement_list(node.orelse)
            finally:
                self._range_alias_stack.pop()
            return node
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("dynamic Tla for requires a simple local name target")

        negative_step_prelude: list[ast.stmt] = []
        if _is_tla_range_call(
            node.iter,
            self._tla_range_names,
            self._tla_module_aliases,
            self._local_scope(),
        ):
            negative_step_prelude = self._rewrite_negative_step_range(node)

        analysis = self.analyze_region_variables(
            construct_name="for",
            assigned_regions=[node.body],
            active_call_nodes=node.body,
        )
        _reject_unsupported_dynamic_for_new_defs(
            analysis.active_symbols,
            analysis.assigned_names,
            node.target.id,
            self._following_loads(),
        )
        carried_names = analysis.carried_names
        self._scope_manager.add_names_to_scope(carried_names)

        body = self._transform_nested_function_body(
            node.body,
            [*analysis.active_symbols, node.target.id, *carried_names],
            analysis.active_callables,
        )
        range_name = self._fresh("range")
        body_name = self._fresh("loop_body")

        range_assign = ast.Assign(
            targets=[ast.Name(id=range_name, ctx=ast.Store())],
            value=node.iter,
        )
        ast.copy_location(range_assign, node)

        body_fn = ast.FunctionDef(
            name=body_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=arg, annotation=None)
                    for arg in [node.target.id, *carried_names]
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=_append_return_for_carried_names(body, carried_names),
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        ast.copy_location(body_fn, node)

        helper_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_INTERNAL_FOR, ctx=ast.Load()),
                args=[
                    ast.Name(id=range_name, ctx=ast.Load()),
                    ast.Name(id=body_name, ctx=ast.Load()),
                    *[ast.Name(id=name, ctx=ast.Load()) for name in carried_names],
                ],
                keywords=[
                    ast.keyword(
                        arg="carried_names",
                        value=ast.Tuple(
                            elts=[ast.Constant(value=name) for name in carried_names],
                            ctx=ast.Load(),
                        ),
                    )
                ],
            )
        )
        ast.copy_location(helper_call, node)

        if not carried_names:
            helper_stmt: ast.stmt = helper_call
        elif len(carried_names) == 1:
            helper_stmt = ast.Assign(
                targets=[ast.Name(id=carried_names[0], ctx=ast.Store())],
                value=helper_call.value,
            )
        else:
            helper_stmt = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store()) for name in carried_names
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=helper_call.value,
            )
        ast.copy_location(helper_stmt, node)

        result: list[ast.stmt] = []
        if _is_tla_range_call(
            node.iter,
            self._tla_range_names,
            self._tla_module_aliases,
            self._local_scope(),
        ):
            result.append(_cf_symbol_check_stmt(node.iter))
        result.extend([*negative_step_prelude, range_assign, body_fn, helper_stmt])
        return result

    def _rewrite_negative_step_range(self, node: ast.For) -> list[ast.stmt]:
        if not isinstance(node.iter, ast.Call) or not isinstance(node.target, ast.Name):
            return []
        bounds = _extract_range_call_bounds(node.iter)
        if bounds is None or not bounds.has_explicit_step:
            return []

        start_original = self._fresh("start_ori")
        stop_original = self._fresh("stop_ori")
        step_original = self._fresh("step_ori")
        is_negative = self._fresh("is_negative")
        start_name = self._fresh("start")
        stop_name = self._fresh("stop")
        step_name = self._fresh("step")
        offset_name = self._fresh("offset")

        prelude: list[ast.stmt] = [
            _assign_name(start_original, bounds.start, node),
            _assign_name(stop_original, bounds.end, node),
            _assign_name(step_original, bounds.step, node),
            _assign_name(
                is_negative,
                ast.Compare(
                    left=ast.Name(id=step_original, ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0)],
                ),
                node,
            ),
            _assign_name(
                start_name,
                _if_expr(
                    is_negative,
                    ast.Name(id=stop_original, ctx=ast.Load()),
                    ast.Name(id=start_original, ctx=ast.Load()),
                ),
                node,
            ),
            _assign_name(
                stop_name,
                _if_expr(
                    is_negative,
                    ast.Name(id=start_original, ctx=ast.Load()),
                    ast.Name(id=stop_original, ctx=ast.Load()),
                ),
                node,
            ),
            _assign_name(
                step_name,
                _if_expr(
                    is_negative,
                    _index_sub_call(
                        ast.Constant(value=0),
                        ast.Name(id=step_original, ctx=ast.Load()),
                    ),
                    ast.Name(id=step_original, ctx=ast.Load()),
                ),
                node,
            ),
            _assign_name(
                offset_name,
                _if_expr(
                    is_negative,
                    _index_add_call(
                        ast.Name(id=start_name, ctx=ast.Load()),
                        ast.Name(id=stop_name, ctx=ast.Load()),
                    ),
                    ast.Constant(value=0),
                ),
                node,
            ),
        ]

        node.iter.args = [
            ast.Name(id=start_name, ctx=ast.Load()),
            ast.Name(id=stop_name, ctx=ast.Load()),
            ast.Name(id=step_name, ctx=ast.Load()),
        ]
        node.iter.keywords = bounds.loop_keywords

        target_name = node.target.id
        remap = _assign_name(
            target_name,
            _if_expr(
                is_negative,
                _index_sub_call(
                    ast.Name(id=offset_name, ctx=ast.Load()),
                    ast.Name(id=target_name, ctx=ast.Load()),
                ),
                ast.Name(id=target_name, ctx=ast.Load()),
            ),
            node.target,
        )
        node.body.insert(0, remap)

        transformed: list[ast.stmt] = []
        for stmt in prelude:
            visited = self.visit(stmt)
            transformed.extend(visited if isinstance(visited, list) else [visited])
        return transformed

    def visit_If(self, node: ast.If) -> Any:
        if self._is_static_control_flow_test(node.test):
            self.generic_visit(node)
            if isinstance(node.test, ast.Call) and _is_constexpr_cf_test(
                node.test,
                self._tla_const_expr_names,
                self._tla_module_aliases,
                self._local_scope(),
            ):
                return [_cf_symbol_check_stmt(node.test), node]
            return node
        _reject_unsupported_dynamic_if_control_flow(node)
        _reject_unsupported_dynamic_if_assignment_targets(node)
        analysis = self.analyze_region_variables(
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[*node.body, *node.orelse],
        )
        then_assigned = analysis.assigned_by_region[0]
        else_assigned = analysis.assigned_by_region[1]
        _reject_unsupported_dynamic_if_new_defs(
            analysis.active_symbols,
            then_assigned,
            else_assigned,
            self._following_loads(),
        )
        carried_names = analysis.carried_names

        test = self.visit(node.test)
        then_body = self._transform_nested_function_body(
            node.body, carried_names, analysis.active_callables
        )
        else_body = (
            self._transform_nested_function_body(
                node.orelse, carried_names, analysis.active_callables
            )
            if node.orelse
            else [ast.Return(value=_names_list(carried_names))]
        )

        then_name = self._fresh("if_then")
        else_name = self._fresh("if_else") if node.orelse or carried_names else None

        then_fn = self._branch_function(then_name, carried_names, then_body)
        result: list[ast.stmt] = [then_fn]
        if else_name is not None:
            result.append(self._branch_function(else_name, carried_names, else_body))

        helper_call = ast.Call(
            func=ast.Name(id=_INTERNAL_IF, ctx=ast.Load()),
            args=[
                test,
                ast.Name(id=then_name, ctx=ast.Load()),
                (
                    ast.Name(id=else_name, ctx=ast.Load())
                    if else_name is not None
                    else ast.Constant(value=None)
                ),
                *[ast.Name(id=name, ctx=ast.Load()) for name in carried_names],
            ],
            keywords=[
                ast.keyword(
                    arg="carried_names",
                    value=ast.Tuple(
                        elts=[ast.Constant(value=name) for name in carried_names],
                        ctx=ast.Load(),
                    ),
                )
            ],
        )
        ast.copy_location(helper_call, node)
        if not carried_names:
            helper_stmt: ast.stmt = ast.Expr(value=helper_call)
        elif len(carried_names) == 1:
            helper_stmt = ast.Assign(
                targets=[ast.Name(id=carried_names[0], ctx=ast.Store())],
                value=helper_call,
            )
        else:
            helper_stmt = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store()) for name in carried_names
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=helper_call,
            )
        ast.copy_location(helper_stmt, node)
        result.append(helper_stmt)
        return result

    def visit_While(self, node: ast.While) -> Any:
        if self._is_static_control_flow_test(node.test):
            self.generic_visit(node)
            if isinstance(node.test, ast.Call) and _is_constexpr_cf_test(
                node.test,
                self._tla_const_expr_names,
                self._tla_module_aliases,
                self._local_scope(),
            ):
                return [_cf_symbol_check_stmt(node.test), node]
            return node
        _reject_unsupported_dynamic_while_control_flow(node)
        _reject_unsupported_dynamic_if_assignment_targets(
            ast.If(test=node.test, body=node.body, orelse=node.orelse)
        )
        if node.orelse:
            raise SyntaxError("dynamic Tla while does not support while-else")

        analysis = self.analyze_region_variables(
            construct_name="while",
            assigned_regions=[node.body],
            active_call_nodes=[node.test, *node.body],
        )
        _reject_unsupported_dynamic_while_new_defs(
            analysis.active_symbols, analysis.assigned_names, self._following_loads()
        )
        carried_names = analysis.carried_names
        self._scope_manager.add_names_to_scope(carried_names)

        before_body = self._transform_nested_function_body(
            [
                ast.Return(
                    value=ast.List(
                        elts=[self.visit(node.test), _names_list(carried_names)],
                        ctx=ast.Load(),
                    )
                )
            ],
            carried_names,
            analysis.active_callables,
        )
        after_body = self._transform_nested_function_body(
            node.body, carried_names, analysis.active_callables
        )
        if not _ends_with_return(after_body):
            after_body.append(ast.Return(value=_names_list(carried_names)))

        before_name = self._fresh("while_before")
        after_name = self._fresh("while_after")
        region_name = self._fresh("while_region")
        before_fn = self._branch_function(before_name, carried_names, before_body)
        after_fn = self._branch_function(after_name, carried_names, after_body)
        ast.copy_location(before_fn, node)
        ast.copy_location(after_fn, node)

        execute_call = ast.Call(
            func=ast.Name(id=_WHILE_EXECUTOR, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(
                    arg="while_before_block",
                    value=ast.Name(id=before_name, ctx=ast.Load()),
                ),
                ast.keyword(
                    arg="while_after_block",
                    value=ast.Name(id=after_name, ctx=ast.Load()),
                ),
                ast.keyword(
                    arg="write_args",
                    value=ast.List(
                        elts=[
                            ast.Name(id=name, ctx=ast.Load()) for name in carried_names
                        ],
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="full_write_args_count",
                    value=ast.Constant(value=analysis.full_write_args_count),
                ),
                ast.keyword(
                    arg="write_args_names",
                    value=ast.List(
                        elts=[ast.Constant(value=name) for name in carried_names],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )
        ast.copy_location(execute_call, node)

        region_fn = ast.FunctionDef(
            name=region_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name, annotation=None) for name in carried_names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[before_fn, after_fn, ast.Return(value=execute_call)],
            decorator_list=[
                ast.Call(
                    func=ast.Name(id=_WHILE_SELECTOR, ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="write_args",
                            value=ast.List(
                                elts=[
                                    ast.Name(id=name, ctx=ast.Load())
                                    for name in carried_names
                                ],
                                ctx=ast.Load(),
                            ),
                        )
                    ],
                )
            ],
            returns=None,
            type_comment=None,
        )
        ast.copy_location(region_fn, node)

        region_result = ast.Name(id=region_name, ctx=ast.Load())
        ast.copy_location(region_result, node)
        if not carried_names:
            helper_stmt: ast.stmt = ast.Expr(value=region_result)
        elif len(carried_names) == 1:
            helper_stmt = ast.Assign(
                targets=[ast.Name(id=carried_names[0], ctx=ast.Store())],
                value=region_result,
            )
        else:
            helper_stmt = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store()) for name in carried_names
                        ],
                        ctx=ast.Store(),
                    )
                ],
                value=region_result,
            )
        ast.copy_location(helper_stmt, node)
        return [region_fn, helper_stmt]

    def _transform_nested_function_body(
        self,
        body: list[ast.stmt],
        initial_scope: list[str] | None = None,
        initial_callables: set[str] | None = None,
    ) -> list[ast.stmt]:
        self._range_alias_stack.append(set())
        with self._scope_manager.enter_local_scope(initial_scope, initial_callables):
            try:
                transformed = self._visit_statement_list(body)
            finally:
                self._range_alias_stack.pop()
        return transformed or [ast.Pass()]

    def _branch_function(
        self, name: str, args: list[str], body: list[ast.stmt]
    ) -> ast.FunctionDef:
        branch_body = list(body)
        if args and not _ends_with_return(branch_body):
            branch_body.append(ast.Return(value=_names_list(args)))
        fn = ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=arg, annotation=None) for arg in args],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=branch_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        return fn

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        then_fn = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=body,
        )
        else_fn = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=orelse,
        )
        ast.copy_location(then_fn, node.body)
        ast.copy_location(else_fn, node.orelse)
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_IF_EXPR, ctx=ast.Load()),
                args=[test, then_fn, else_fn],
                keywords=[],
            ),
            node,
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        node = self.generic_visit(node)
        if not isinstance(node.op, (ast.And, ast.Or)) or len(node.values) < 2:
            return node
        if isinstance(node.op, ast.And):
            helper = _INTERNAL_BOOL_AND
            short_circuit_value = ast.Constant(value=False)
        else:
            helper = _INTERNAL_BOOL_OR
            short_circuit_value = ast.Constant(value=True)

        lhs = node.values[0]
        for rhs in node.values[1:]:
            lhs = ast.copy_location(
                ast.IfExp(
                    test=_static_bool_equals(lhs, short_circuit_value),
                    body=lhs,
                    orelse=ast.Call(
                        func=ast.Name(id=helper, ctx=ast.Load()),
                        args=[lhs, rhs],
                        keywords=[],
                    ),
                ),
                node,
            )
        return lhs

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Not):
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_BOOL_NOT, ctx=ast.Load()),
                args=[node.operand],
                keywords=[],
            ),
            node,
        )

    def visit_Compare(self, node: ast.Compare) -> Any:
        node = self.generic_visit(node)
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_COMPARE, ctx=ast.Load()),
                args=[
                    node.left,
                    ast.Tuple(elts=node.comparators, ctx=ast.Load()),
                    ast.Tuple(
                        elts=[
                            ast.Constant(value=_compare_op_name(op)) for op in node.ops
                        ],
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            ),
            node,
        )

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)
        original_func_name = node.func.id
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        if original_func_name in {"any", "all", "bool"}:
            if node.keywords or len(node.args) != 1:
                return node
            helper = _BUILTIN_REDIRECTS[original_func_name]
        elif original_func_name in {"min", "max"}:
            if node.keywords:
                return node
            helper = _BUILTIN_REDIRECTS[original_func_name]
        else:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=helper, ctx=ast.Load()),
                args=node.args,
                keywords=[],
            ),
            node,
        )

    def visit_Name(self, node: ast.Name) -> Any:
        if not isinstance(node.ctx, ast.Load):
            return node
        helper = _BUILTIN_REDIRECTS.get(node.id)
        if helper is None:
            return node
        return ast.copy_location(
            ast.IfExp(
                test=ast.Compare(
                    left=ast.Name(id=node.id, ctx=ast.Load()),
                    ops=[ast.Is()],
                    comparators=[
                        ast.Attribute(
                            value=ast.Name(id="__tladsl_builtins__", ctx=ast.Load()),
                            attr=node.id,
                            ctx=ast.Load(),
                        )
                    ],
                ),
                body=ast.Name(id=helper, ctx=ast.Load()),
                orelse=ast.Name(id=node.id, ctx=ast.Load()),
            ),
            node,
        )

    def visit_With(self, node: ast.With) -> Any:
        # Names bound in the enclosing scope(s) before this region. A region body
        # is hoisted into a nested function but is semantically an inline block
        # sharing the enclosing scope, so enclosing names reassigned inside it
        # must be threaded via ``nonlocal`` (see below).
        enclosing_symbols = set(self._local_scope())
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name):
                self._scope_manager.add_to_scope(item.optional_vars.id)
        node.items = [self.visit(item) for item in node.items]
        # A region body (e.g. ``with tla.vec.func():``) is hoisted into its own
        # nested function below, so process it like a function body: a fresh
        # scope with sequential statement-by-statement registration. This lets
        # loop-carried values defined inside the region (seeded before a
        # ``tla.range`` loop and reassigned in it) be detected as carried.
        self._range_alias_stack.append(set())
        try:
            with self._scope_manager.enter_local_scope():
                node.body = self._visit_statement_list(node.body) or [ast.Pass()]
        finally:
            self._range_alias_stack.pop()
        if len(node.items) != 1:
            return node
        region_name = _region_name_from_with_item(node.items[0])
        if region_name is None:
            return node
        region_mode = _region_mode_from_with_item(node.items[0])

        body_name = self._fresh(f"{region_name.replace('.', '_')}_body")
        # The region body shares the enclosing scope semantically. Any enclosing
        # variable reassigned inside the body -- most commonly the carried-value
        # round-trip the if/for lowering emits (``x = internal_if(..., x, ...)``)
        # for a mutex/flag that is method-invoked inside nested control flow --
        # would otherwise be treated as a local of this nested function, so the
        # read on the RHS raises UnboundLocalError. Declare those names nonlocal
        # so they keep referring to the enclosing binding.
        region_body: list[ast.stmt] = node.body or [ast.Pass()]
        nonlocal_names = sorted(
            enclosing_symbols & _assigned_names_from_statements(region_body)
        )
        if nonlocal_names:
            region_body = [ast.Nonlocal(names=nonlocal_names), *region_body]
        body_fn = ast.FunctionDef(
            name=body_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=region_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        ast.copy_location(body_fn, node)

        helper_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_INTERNAL_REGION, ctx=ast.Load()),
                args=[
                    ast.Constant(value=region_name),
                    ast.Name(id=body_name, ctx=ast.Load()),
                ],
                keywords=(
                    []
                    if region_mode is None
                    else [ast.keyword(arg="mode", value=region_mode)]
                ),
            )
        )
        ast.copy_location(helper_call, node)
        return [body_fn, helper_call]


def maybe_transform_for_lowering(
    fn: FunctionType,
    *,
    internal_for: Any,
    internal_region: Any,
    internal_if: Any,
    internal_if_expr: Any,
    internal_bool_and: Any,
    internal_bool_or: Any,
    internal_bool_not: Any,
    internal_compare: Any,
    internal_any: Any,
    internal_all: Any,
    internal_bool: Any,
    internal_min: Any,
    internal_max: Any,
) -> FunctionType:
    """Return a transformed callable when source-driven control-flow lowering is needed."""

    try:
        source = inspect.getsource(fn)
    except (OSError, IOError, TypeError):
        return fn

    if (
        "tla.range" not in source
        and "tla.range" not in source
        and "tla.cube" not in source
        and "tla.cube" not in source
        and "tla.vector" not in source
        and "tla.vector" not in source
        and "tla.vec.func" not in source
        and "range(" not in source
        and "range_constexpr(" not in source
        and " if " not in source
        and "while " not in source
        and "any(" not in source
        and "all(" not in source
        and "bool(" not in source
        and "min(" not in source
        and "max(" not in source
    ):
        return fn

    module_ast = ast.parse(
        textwrap.dedent(source), filename=inspect.getsourcefile(fn) or "<unknown>"
    )
    target = _find_function_def(module_ast, fn.__name__)
    if target is None:
        return fn

    target.decorator_list = []
    exec_globals = dict(fn.__globals__)
    transformed = _FrontendControlFlowTransformer(exec_globals).visit(module_ast)
    ast.fix_missing_locations(transformed)

    exec_globals[_INTERNAL_FOR] = internal_for
    exec_globals[_INTERNAL_REGION] = internal_region
    exec_globals[_INTERNAL_IF] = internal_if
    exec_globals[_INTERNAL_IF_EXPR] = internal_if_expr
    exec_globals[_INTERNAL_BOOL_AND] = internal_bool_and
    exec_globals[_INTERNAL_BOOL_OR] = internal_bool_or
    exec_globals[_INTERNAL_BOOL_NOT] = internal_bool_not
    exec_globals[_INTERNAL_COMPARE] = internal_compare
    exec_globals[_INTERNAL_ANY] = internal_any
    exec_globals[_INTERNAL_ALL] = internal_all
    exec_globals[_INTERNAL_BOOL] = internal_bool
    exec_globals[_INTERNAL_MIN] = internal_min
    exec_globals[_INTERNAL_MAX] = internal_max
    exec_globals[_INTERNAL_CF_SYMBOL_CHECK] = _cf_symbol_check
    exec_globals[_INTERNAL_INDEX_ADD] = _index_add
    exec_globals[_INTERNAL_INDEX_SUB] = _index_sub
    from .ast_helpers import while_executor, while_selector

    exec_globals[_WHILE_EXECUTOR] = while_executor
    exec_globals[_WHILE_SELECTOR] = while_selector
    exec_globals["__tladsl_builtins__"] = builtins
    namespace: dict[str, Any] = {}
    code = compile(
        transformed,
        filename=inspect.getsourcefile(fn) or "<unknown>",
        mode="exec",
    )
    exec(code, exec_globals, namespace)
    rewritten = namespace.get(fn.__name__)
    if not isinstance(rewritten, FunctionType):
        return fn
    rewritten.__defaults__ = fn.__defaults__
    rewritten.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    rewritten.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rewritten.__module__ = fn.__module__
    rewritten.__qualname__ = fn.__qualname__
    return rewritten


def _find_function_def(module_ast: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _tla_function_names_from_globals(
    global_symbols: dict[str, Any], function_name: str
) -> set[str]:
    return {
        name
        for name, value in global_symbols.items()
        if getattr(value, "__module__", None) == "catlass.core_api"
        and getattr(value, "__name__", None) == function_name
    }


def _tla_const_expr_names_from_globals(global_symbols: dict[str, Any]) -> set[str]:
    names = {"const_expr"}
    names.update(
        name
        for name, value in global_symbols.items()
        if getattr(value, "__module__", None) == "catlass.runtime"
        and getattr(value, "__name__", None) == "const_expr"
    )
    return names


def _tla_module_aliases_from_globals(global_symbols: dict[str, Any]) -> set[str]:
    return {
        name
        for name, value in global_symbols.items()
        if isinstance(value, ModuleType) and value.__name__ == "catlass"
    }


def _is_tla_range_call(
    node: ast.AST,
    tla_range_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if (
        isinstance(func, ast.Name)
        and func.id in tla_range_names
        and func.id not in local_names
    ):
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in tla_module_aliases and func.attr == "range"
    return False


def _is_tla_range_constexpr_call(
    node: ast.AST,
    tla_range_constexpr_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if (
        isinstance(func, ast.Name)
        and func.id in tla_range_constexpr_names
        and func.id not in local_names
    ):
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in tla_module_aliases and func.attr == "range_constexpr"
    return False


def _cf_symbol_check_stmt(range_call: ast.Call) -> ast.stmt:
    func = range_call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        symbol: ast.expr = ast.Name(id=func.value.id, ctx=ast.Load())
    elif isinstance(func, ast.Name):
        symbol = ast.Name(id=func.id, ctx=ast.Load())
    else:
        raise SyntaxError("dynamic Tla for requires a Tla range symbol")
    check_stmt = ast.Expr(
        value=ast.Call(
            func=ast.Name(id=_INTERNAL_CF_SYMBOL_CHECK, ctx=ast.Load()),
            args=[symbol],
            keywords=[],
        )
    )
    return ast.copy_location(check_stmt, range_call)


def _builtin_range_call_from_range_constexpr(node: ast.Call) -> ast.Call:
    rewritten = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="__tladsl_builtins__", ctx=ast.Load()),
            attr="range",
            ctx=ast.Load(),
        ),
        args=node.args,
        keywords=node.keywords,
    )
    return ast.copy_location(rewritten, node)


class _RangeCallBounds:
    def __init__(
        self,
        *,
        start: ast.expr,
        end: ast.expr,
        step: ast.expr,
        has_explicit_step: bool,
        loop_keywords: list[ast.keyword],
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.has_explicit_step = has_explicit_step
        self.loop_keywords = loop_keywords


_RANGE_LOOP_KEYWORDS = {
    "unroll",
    "unroll_full",
    "prefetch_stages",
    "pipelining",
}


def _validate_dynamic_range_call_syntax(node: ast.Call) -> None:
    seen: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg is None:
            raise SyntaxError("dynamic Tla range does not support **kwargs")
        if keyword.arg in {"start", "stop", "end", "step"}:
            raise SyntaxError(
                "dynamic Tla range bounds must be positional, matching standard range(start[, stop[, step]]) bounds"
            )
        if keyword.arg not in _RANGE_LOOP_KEYWORDS:
            raise SyntaxError(
                f"dynamic Tla range got unsupported keyword {keyword.arg!r}"
            )
        if keyword.arg in seen:
            raise SyntaxError(
                f"dynamic Tla range got duplicate keyword {keyword.arg!r}"
            )
        seen.add(keyword.arg)
    if "prefetch_stages" in seen and "pipelining" in seen:
        raise SyntaxError(
            "dynamic Tla range cannot specify both 'prefetch_stages' and 'pipelining'"
        )
    if len(node.args) not in {1, 2, 3}:
        raise SyntaxError("dynamic Tla range expects 1, 2, or 3 positional bounds")


def _extract_range_call_bounds(node: ast.Call) -> _RangeCallBounds | None:
    if len(node.args) > 3:
        return None

    start: ast.expr
    end: ast.expr
    step: ast.expr
    has_explicit_step = False
    if len(node.args) == 1:
        start = ast.Constant(value=0)
        end = node.args[0]
        step = ast.Constant(value=1)
    elif len(node.args) == 2:
        start = node.args[0]
        end = node.args[1]
        step = ast.Constant(value=1)
    elif len(node.args) == 3:
        start = node.args[0]
        end = node.args[1]
        step = node.args[2]
        has_explicit_step = True
    else:
        return None

    return _RangeCallBounds(
        start=start,
        end=end,
        step=step,
        has_explicit_step=has_explicit_step,
        loop_keywords=node.keywords,
    )


def _assign_name(name: str, value: ast.expr, location: ast.AST) -> ast.Assign:
    return ast.copy_location(
        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value),
        location,
    )


def _if_expr(test_name: str, body: ast.expr, orelse: ast.expr) -> ast.IfExp:
    return ast.IfExp(
        test=ast.Name(id=test_name, ctx=ast.Load()),
        body=body,
        orelse=orelse,
    )


def _index_add_call(lhs: ast.expr, rhs: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=_INTERNAL_INDEX_ADD, ctx=ast.Load()),
        args=[lhs, rhs],
        keywords=[],
    )


def _index_sub_call(lhs: ast.expr, rhs: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=_INTERNAL_INDEX_SUB, ctx=ast.Load()),
        args=[lhs, rhs],
        keywords=[],
    )


def _index_add(lhs: Any, rhs: Any) -> Any:
    from catlass import runtime as _runtime

    return _runtime._IndexExpr(_runtime._coerce_index_value(lhs)) + rhs


def _index_sub(lhs: Any, rhs: Any) -> Any:
    from catlass import runtime as _runtime

    return _runtime._IndexExpr(_runtime._coerce_index_value(lhs)) - rhs


def _cf_symbol_check(symbol: Any) -> None:
    if isinstance(symbol, ModuleType):
        if symbol.__name__ == "catlass":
            return
        name = symbol.__name__.split(".")[-1]
    else:
        module_name = getattr(symbol, "__module__", None)
        symbol_name = getattr(symbol, "__name__", None)
        if module_name == "catlass.core_api" and symbol_name in {
            "range",
            "range_constexpr",
        }:
            return
        if module_name == "catlass.runtime" and symbol_name == "const_expr":
            return
        name = getattr(symbol, "__name__", type(symbol).__name__)
    raise RuntimeError(f"Incorrect `{name}` is used. Please use the Tla DSL symbol.")


def _is_tla_range_iter(
    node: ast.AST,
    aliases: set[str],
    tla_range_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if _is_tla_range_call(node, tla_range_names, tla_module_aliases, local_names):
        return True
    return isinstance(node, ast.Name) and node.id in aliases


def _update_range_aliases(
    stmt: ast.stmt,
    aliases: set[str],
    tla_range_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> None:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return
    if _is_tla_range_call(stmt.value, tla_range_names, tla_module_aliases, local_names):
        _validate_dynamic_range_call_syntax(stmt.value)
        aliases.add(target.id)
    else:
        aliases.discard(target.id)


def _region_name_from_with_item(item: ast.withitem) -> str | None:
    context_expr = item.context_expr
    if not isinstance(context_expr, ast.Call):
        return None
    func = context_expr.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in {"tla"} and func.attr in {"cube", "vector"}:
            return func.attr
    if _is_tla_vec_func(func):
        return "vec.func"
    return None


def _is_tla_vec_func(func: ast.expr) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "func"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "vec"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "tla"
    )


def _raise_vec_func_error(message: str) -> None:
    from catlass.runtime import TlaCoreAPIError

    raise TlaCoreAPIError(f"tla.vec.func: {message}")


def _region_mode_from_with_item(item: ast.withitem) -> ast.expr | None:
    context_expr = item.context_expr
    if not isinstance(context_expr, ast.Call):
        return None
    if not _is_tla_vec_func(context_expr.func):
        return None
    if context_expr.args:
        _raise_vec_func_error("mode must be passed by keyword")
    mode_expr: ast.expr | None = None
    for keyword in context_expr.keywords:
        if keyword.arg != "mode":
            _raise_vec_func_error(f"unknown keyword argument: {keyword.arg}")
        if mode_expr is not None:
            _raise_vec_func_error("mode was passed multiple times")
        mode_expr = keyword.value
    return mode_expr if mode_expr is not None else ast.Constant(value="simd")


def _is_constexpr_cf_test(
    node: ast.AST,
    tla_const_expr_names: set[str],
    tla_module_aliases: set[str],
    local_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        del tla_module_aliases
        return func.attr == "const_expr"
    if isinstance(func, ast.Name):
        if func.id == "const_expr":
            return True
        return func.id in tla_const_expr_names and func.id not in local_names
    return False


def _is_const_expr_call(node: ast.AST) -> bool:
    return _is_constexpr_cf_test(node, {"const_expr"}, {"tla"}, set())


def _is_static_python_if_test(
    node: ast.AST,
    tla_const_expr_names: set[str] | None = None,
    tla_module_aliases: set[str] | None = None,
    local_names: set[str] | None = None,
) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, bool)
        or _is_constexpr_cf_test(
            node,
            tla_const_expr_names or {"const_expr"},
            tla_module_aliases or {"tla"},
            local_names or set(),
        )
    )


def _compare_op_name(op: ast.cmpop) -> str:
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    if isinstance(op, ast.Is):
        return "is"
    if isinstance(op, ast.IsNot):
        return "is not"
    if isinstance(op, ast.In):
        return "in"
    if isinstance(op, ast.NotIn):
        return "not in"
    raise SyntaxError(f"unsupported comparison operator: {type(op).__name__}")


def _static_bool_equals(value: ast.AST, expected: ast.Constant) -> ast.BoolOp:
    return ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Compare(
                left=ast.Call(
                    func=ast.Name(id="type", ctx=ast.Load()),
                    args=[value],
                    keywords=[],
                ),
                ops=[ast.Eq()],
                comparators=[ast.Name(id="bool", ctx=ast.Load())],
            ),
            ast.Compare(
                left=value,
                ops=[ast.Eq()],
                comparators=[expected],
            ),
        ],
    )


def _function_arg_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs}
    names.update(arg.arg for arg in args.args)
    names.update(arg.arg for arg in args.kwonlyargs)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _assigned_names_from_statements(body: list[ast.stmt]) -> set[str]:
    assigned: set[str] = set()
    for stmt in body:
        assigned.update(_assigned_names(stmt))
    return assigned


def _assigned_names(node: ast.AST) -> set[str]:
    assigned: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, name_node: ast.Name) -> None:
            if isinstance(name_node.ctx, ast.Store):
                assigned.add(name_node.id)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            assigned.add(function_node.name)

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            assigned.add(function_node.name)

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            assigned.add(class_node.name)

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

    Visitor().visit(node)
    return assigned


def _callable_names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}
    return set()


def _loaded_names_from_statements(body: list[ast.stmt]) -> set[str]:
    loaded: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, name_node: ast.Name) -> None:
            if isinstance(name_node.ctx, ast.Load):
                loaded.add(name_node.id)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

    for stmt in body:
        Visitor().visit(stmt)
    return loaded


def _invoked_active_names_from_statements(
    body: list[ast.stmt], active_names: set[str]
) -> set[str]:
    invoked: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            base_name = _call_base_name(call_node.func)
            if base_name in active_names:
                invoked.add(base_name)
            self.generic_visit(call_node)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

    for stmt in body:
        Visitor().visit(stmt)
    return invoked


def _reject_unsupported_dynamic_if_control_flow(node: ast.If) -> None:
    class Visitor(ast.NodeVisitor):
        def visit_Return(self, return_node: ast.Return) -> None:
            del return_node
            self._raise()

        def visit_Break(self, break_node: ast.Break) -> None:
            del break_node
            self._raise()

        def visit_Continue(self, continue_node: ast.Continue) -> None:
            del continue_node
            self._raise()

        def visit_Raise(self, raise_node: ast.Raise) -> None:
            del raise_node
            self._raise()

        def visit_If(self, if_node: ast.If) -> None:
            if _is_static_python_if_test(if_node.test):
                self.generic_visit(if_node)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

        def _raise(self) -> None:
            raise SyntaxError(
                "dynamic Tla if does not support return, break, continue, or raise"
            )

    for stmt in [*node.body, *node.orelse]:
        Visitor().visit(stmt)


def _reject_unsupported_dynamic_for_control_flow(node: ast.For) -> None:
    class Visitor(ast.NodeVisitor):
        def visit_Return(self, return_node: ast.Return) -> None:
            del return_node
            self._raise()

        def visit_Break(self, break_node: ast.Break) -> None:
            del break_node
            self._raise()

        def visit_Continue(self, continue_node: ast.Continue) -> None:
            del continue_node
            self._raise()

        def visit_Raise(self, raise_node: ast.Raise) -> None:
            del raise_node
            self._raise()

        def visit_If(self, if_node: ast.If) -> None:
            if _is_static_python_if_test(if_node.test):
                self.generic_visit(if_node)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

        def _raise(self) -> None:
            raise SyntaxError(
                "dynamic Tla for does not support return, break, continue, or raise"
            )

    for stmt in node.body:
        Visitor().visit(stmt)


def _reject_unsupported_dynamic_while_control_flow(node: ast.While) -> None:
    class Visitor(ast.NodeVisitor):
        def visit_Return(self, return_node: ast.Return) -> None:
            del return_node
            self._raise()

        def visit_Break(self, break_node: ast.Break) -> None:
            del break_node
            self._raise()

        def visit_Continue(self, continue_node: ast.Continue) -> None:
            del continue_node
            self._raise()

        def visit_Raise(self, raise_node: ast.Raise) -> None:
            del raise_node
            self._raise()

        def visit_If(self, if_node: ast.If) -> None:
            if _is_static_python_if_test(if_node.test):
                self.generic_visit(if_node)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

        def _raise(self) -> None:
            raise SyntaxError(
                "dynamic Tla while does not support return, break, continue, or raise"
            )

    for stmt in node.body:
        Visitor().visit(stmt)


def _reject_unsupported_dynamic_for_new_defs(
    active_names: set[str],
    body_assigned: set[str],
    target_name: str,
    following_loads: set[str],
) -> None:
    if target_name in following_loads:
        raise SyntaxError(
            "dynamic Tla for induction variables cannot be used after the loop: "
            f"{target_name}"
        )
    new_defs = (body_assigned - active_names) - {target_name}
    used_after = sorted(new_defs & following_loads)
    if used_after:
        raise SyntaxError(
            "dynamic Tla for values used after the loop must be initialized "
            f"before the loop: {', '.join(used_after)}"
        )


def _reject_unsupported_dynamic_if_assignment_targets(node: ast.If) -> None:
    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, assign_node: ast.Assign) -> None:
            for target in assign_node.targets:
                self._check_target(target)
            self.visit(assign_node.value)

        def visit_AnnAssign(self, assign_node: ast.AnnAssign) -> None:
            self._check_target(assign_node.target)
            if assign_node.value is not None:
                self.visit(assign_node.value)

        def visit_AugAssign(self, assign_node: ast.AugAssign) -> None:
            self._check_target(assign_node.target)
            self.visit(assign_node.value)

        def visit_Delete(self, delete_node: ast.Delete) -> None:
            del delete_node
            self._raise()

        def visit_For(self, for_node: ast.For) -> None:
            self._check_target(for_node.target)
            self.visit(for_node.iter)
            for stmt in [*for_node.body, *for_node.orelse]:
                self.visit(stmt)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

        def _check_target(self, target: ast.AST) -> None:
            if isinstance(target, ast.Name):
                return
            if isinstance(target, (ast.Tuple, ast.List)) and all(
                isinstance(element, ast.Name) for element in target.elts
            ):
                return
            self._raise()

        def _raise(self) -> None:
            raise SyntaxError(
                "dynamic Tla if only supports assignments to local names or "
                "tuples/lists of local names"
            )

    for stmt in [*node.body, *node.orelse]:
        Visitor().visit(stmt)


def _reject_unsupported_dynamic_active_callable_calls(
    body: list[ast.stmt], active_callables: set[str], construct_name: str
) -> None:
    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            func = call_node.func
            if isinstance(func, ast.Name) and func.id in active_callables:
                self._raise_callable(func.id)
            self.generic_visit(call_node)

        def visit_FunctionDef(self, function_node: ast.FunctionDef) -> None:
            del function_node

        def visit_AsyncFunctionDef(self, function_node: ast.AsyncFunctionDef) -> None:
            del function_node

        def visit_ClassDef(self, class_node: ast.ClassDef) -> None:
            del class_node

        def visit_Lambda(self, lambda_node: ast.Lambda) -> None:
            del lambda_node

        def _raise_callable(self, name: str) -> None:
            raise SyntaxError(
                f"dynamic Tla {construct_name} does not support calling active local callable "
                f"{name!r}; inline the branch body or move the call outside the "
                f"dynamic {construct_name}"
            )

    for stmt in body:
        Visitor().visit(stmt)


def _call_base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return _call_base_name(node.value)
    if isinstance(node, ast.Name) and node.id not in {"tla"}:
        return node.id
    return None


def _reject_unsupported_dynamic_if_new_defs(
    active_names: set[str],
    then_assigned: set[str],
    else_assigned: set[str],
    following_loads: set[str],
) -> None:
    newly_assigned = (then_assigned | else_assigned) - active_names
    used_later = newly_assigned & following_loads
    if used_later:
        names = ", ".join(sorted(used_later))
        raise SyntaxError(
            "dynamic Tla if requires variables used after the branch to be "
            f"initialized before the if: {names}"
        )


def _reject_unsupported_dynamic_while_new_defs(
    active_names: set[str],
    body_assigned: set[str],
    following_loads: set[str],
) -> None:
    newly_assigned = body_assigned - active_names
    used_later = newly_assigned & following_loads
    if used_later:
        names = ", ".join(sorted(used_later))
        raise SyntaxError(
            "dynamic Tla while requires variables used after the loop to be "
            f"initialized before the while: {names}"
        )


def _names_list(names: list[str]) -> ast.List:
    return ast.List(
        elts=[ast.Name(id=name, ctx=ast.Load()) for name in names],
        ctx=ast.Load(),
    )


def _ends_with_return(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], ast.Return)


def _append_return_for_carried_names(
    body: list[ast.stmt], carried_names: list[str]
) -> list[ast.stmt]:
    if not carried_names or _ends_with_return(body):
        return body
    return [*body, ast.Return(value=_names_list(carried_names))]
