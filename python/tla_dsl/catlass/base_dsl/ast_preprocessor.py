"""Minimal AST rewrites used to lower structured frontend control flow."""

from __future__ import annotations

import ast
import builtins
import contextlib
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, Callable, Iterator


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
_INTERNAL_ATTACH_SOURCE_INFO = "__tladsl_internal_attach_source_info__"
_INTERNAL_UNKNOWN_EFFECT_CALL = "__tladsl_internal_unknown_effect_call__"
_INTERNAL_LAZY_ATTRIBUTE = "__tladsl_internal_lazy_attribute__"
_INTERNAL_LAZY_SUBSCRIPT = "__tladsl_internal_lazy_subscript__"
_INTERNAL_LAZY_BINOP = "__tladsl_internal_lazy_binop__"
_INTERNAL_LAZY_UNARY = "__tladsl_internal_lazy_unary__"
_SOURCE_INFO_ATTR = "__tladsl_source_info__"
_WHILE_SELECTOR = "while_selector"
_WHILE_EXECUTOR = "while_executor"
_BUILTIN_REDIRECTS = {
    "any": _INTERNAL_ANY,
    "all": _INTERNAL_ALL,
    "bool": _INTERNAL_BOOL,
    "min": _INTERNAL_MIN,
    "max": _INTERNAL_MAX,
}
_TRUSTED_LAZY_CALLABLES: tuple[Callable[..., Any], ...] = ()


def _register_trusted_lazy_callables(
    callables: tuple[Callable[..., Any], ...],
) -> None:
    """Freeze genuine DSL call identities once core_api has initialized."""

    global _TRUSTED_LAZY_CALLABLES
    if _TRUSTED_LAZY_CALLABLES:
        raise RuntimeError("trusted lazy callables are already registered")
    _TRUSTED_LAZY_CALLABLES = callables


@dataclass(frozen=True)
class ControlFlowPlan:
    """Immutable analysis consumed by one runtime control-flow rewrite."""

    construct_name: str
    lineno: int
    col_offset: int
    active_symbols: frozenset[str]
    active_callables: frozenset[str]
    assigned_by_region: tuple[frozenset[str], ...]
    assigned_names: frozenset[str]
    invoked_names: frozenset[str]
    carried_names: tuple[str, ...]
    full_write_args_count: int
    assignment_targets: tuple[str, ...]
    tensor_store_assignments: frozenset[int]
    nested_constructs: tuple[tuple[str, int, int], ...]


class _ControlFlowAnalyzer:
    """Analyze and validate a runtime construct without rewriting its AST."""

    def analyze(
        self,
        *,
        node: ast.If | ast.For | ast.While,
        construct_name: str,
        assigned_regions: list[list[ast.stmt]],
        active_call_nodes: list[ast.AST],
        active_symbols: set[str],
        active_callables: set[str],
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
        is_runtime_for: Callable[[ast.For], bool] | None = None,
        is_static_test: Callable[[ast.AST], bool] | None = None,
    ) -> ControlFlowPlan:
        _reject_unsupported_dynamic_active_callable_calls(
            active_call_nodes, active_callables, construct_name
        )
        policy = _DynamicControlFlowPolicy(
            construct_name,
            filename=filename,
            line_offset=line_offset,
            source_text=source_text,
            is_runtime_for=is_runtime_for,
            is_static_test=is_static_test,
        )
        for statement in (stmt for region in assigned_regions for stmt in region):
            policy.visit(statement)
        assigned_by_region = tuple(
            _assigned_names_from_statements(region) for region in assigned_regions
        )
        assigned_names = set().union(*assigned_by_region)
        invoked_names = _invoked_active_names_from_statements(
            active_call_nodes, active_symbols
        )
        carried_names = tuple(sorted((assigned_names & active_symbols) | invoked_names))
        return ControlFlowPlan(
            construct_name=construct_name,
            lineno=int(getattr(node, "lineno", 0) or 0),
            col_offset=int(getattr(node, "col_offset", 0) or 0),
            active_symbols=frozenset(active_symbols),
            active_callables=frozenset(active_callables),
            assigned_by_region=tuple(frozenset(names) for names in assigned_by_region),
            assigned_names=frozenset(assigned_names),
            invoked_names=frozenset(invoked_names),
            carried_names=carried_names,
            full_write_args_count=len(carried_names),
            assignment_targets=tuple(policy.assignment_targets),
            tensor_store_assignments=frozenset(policy.tensor_store_assignments),
            nested_constructs=tuple(policy.nested_constructs),
        )


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


class _DynamicConditionValidator(ast.NodeVisitor):
    """Reject condition syntax whose Python evaluation cannot be preserved."""

    def __init__(
        self,
        construct_name: str,
        *,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
    ) -> None:
        self.construct_name = construct_name
        self.filename = filename
        self.line_offset = line_offset
        self.source_lines = source_text.splitlines()

    def validate(self, node: ast.AST) -> None:
        self.visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._raise(node, "does not support assignment expressions in its condition")

    def visit_Await(self, node: ast.Await) -> None:
        self._raise(node, "does not support await expressions in its condition")

    def visit_Yield(self, node: ast.Yield) -> None:
        self._raise(node, "does not support yield expressions in its condition")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._raise(node, "does not support yield expressions in its condition")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._raise(node, "does not support lambda expressions in its condition")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._raise(node, "does not support comprehension expressions in its condition")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._raise(node, "does not support comprehension expressions in its condition")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._raise(node, "does not support comprehension expressions in its condition")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._raise(node, "does not support generator expressions in its condition")

    def _raise(self, node: ast.AST, message: str) -> None:
        error = SyntaxError(f"dynamic Tla {self.construct_name} {message}")
        relative_lineno = int(getattr(node, "lineno", 0) or 0)
        error.filename = self.filename
        error.lineno = self.line_offset + relative_lineno
        error.offset = (
            int(getattr(node, "col_offset", 0)) + 1
            if getattr(node, "col_offset", None) is not None
            else None
        )
        if 0 < relative_lineno <= len(self.source_lines):
            error.text = self.source_lines[relative_lineno - 1]
        raise error


class _FrontendControlFlowTransformer(ast.NodeTransformer):
    def __init__(
        self,
        global_symbols: dict[str, Any] | None = None,
        *,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
    ) -> None:
        self._counter = 0
        self._reserved_names: set[str] = set()
        self._range_alias_stack: list[set[str]] = []
        self._scope_manager = ScopeManager.create()
        self._following_loads_stack: list[set[str]] = []
        self._tensor_store_assignments: set[int] = set()
        self._control_flow_analyzer = _ControlFlowAnalyzer()
        self._global_symbols = global_symbols or {}
        self._filename = filename
        self._line_offset = line_offset
        self._source_text = source_text
        self._tensor_store_helper_name: str | None = None
        self._lazy_operand_context: str | None = None
        self._call_shadow_stack: list[set[str]] = []
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

    def visit_Module(self, node: ast.Module) -> Any:
        self._reserved_names.update(_identifier_names(node))
        self._tensor_store_helper_name = self._fresh("tensor_store")
        return self.generic_visit(node)

    @property
    def tensor_store_helper_name(self) -> str:
        if self._tensor_store_helper_name is None:
            raise RuntimeError("frontend module has not been initialized")
        return self._tensor_store_helper_name

    def _fresh(self, prefix: str) -> str:
        while True:
            self._counter += 1
            name = f"__tladsl_{prefix}_{self._counter}"
            if name in self._reserved_names:
                continue
            self._reserved_names.add(name)
            return name

    def _source_info_dict(
        self,
        generated_name: str,
        node: ast.AST,
        *,
        construct: str,
        region: str,
    ) -> ast.Dict:
        source = ast.get_source_segment(self._source_text, node) or ""
        source = next(
            (line.strip() for line in source.splitlines() if line.strip()), ""
        )
        return ast.Dict(
            keys=[
                ast.Constant(value="filename"),
                ast.Constant(value="lineno"),
                ast.Constant(value="col_offset"),
                ast.Constant(value="construct"),
                ast.Constant(value="region"),
                ast.Constant(value="generated_name"),
                ast.Constant(value="source"),
            ],
            values=[
                ast.Constant(value=self._filename),
                ast.Constant(
                    value=self._line_offset + int(getattr(node, "lineno", 0) or 0)
                ),
                ast.Constant(value=int(getattr(node, "col_offset", 0) or 0)),
                ast.Constant(value=construct),
                ast.Constant(value=region),
                ast.Constant(value=generated_name),
                ast.Constant(value=source),
            ],
        )

    def _source_info_stmt(
        self,
        function_name: str,
        node: ast.AST,
        *,
        construct: str,
        region: str,
    ) -> ast.stmt:
        info = self._source_info_dict(
            function_name, node, construct=construct, region=region
        )
        stmt = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=function_name, ctx=ast.Load()),
                    attr=_SOURCE_INFO_ATTR,
                    ctx=ast.Store(),
                )
            ],
            value=info,
        )
        return ast.copy_location(stmt, node)

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

    def analyze_control_flow(
        self,
        *,
        node: ast.If | ast.For | ast.While,
        construct_name: str,
        assigned_regions: list[list[ast.stmt]],
        active_call_nodes: list[ast.AST],
    ) -> ControlFlowPlan:
        plan = self._control_flow_analyzer.analyze(
            node=node,
            construct_name=construct_name,
            assigned_regions=assigned_regions,
            active_call_nodes=active_call_nodes,
            active_symbols=set(self._local_scope()),
            active_callables=set(self._active_callables()),
            filename=self._filename,
            line_offset=self._line_offset,
            source_text=self._source_text,
            is_runtime_for=lambda nested: _is_tla_range_iter(
                nested.iter,
                self._range_aliases(),
                self._tla_range_names,
                self._tla_module_aliases,
                self._local_scope(),
            ),
            is_static_test=self._is_static_control_flow_test,
        )
        self._tensor_store_assignments.update(plan.tensor_store_assignments)
        return plan

    def visit_Assign(self, node: ast.Assign) -> Any:
        if id(node) not in self._tensor_store_assignments:
            return self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
            raise SyntaxError(
                "dynamic Tla control flow requires a tensor store to have one target"
            )
        target = node.targets[0]
        rewritten = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=self.tensor_store_helper_name, ctx=ast.Load()),
                # Python assignment evaluates RHS before the target and index.
                args=[
                    self.visit(node.value),
                    self.visit(target.value),
                    self.visit(target.slice),
                ],
                keywords=[],
            )
        )
        return ast.copy_location(rewritten, node)

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
        local_names = _function_arg_names(node.args) | _assigned_names_from_statements(
            node.body
        )
        self._call_shadow_stack.append(local_names)
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
                self._call_shadow_stack.pop()
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

        analysis = self.analyze_control_flow(
            node=node,
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
        carried_names = list(analysis.carried_names)
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
        result.extend(
            [
                *negative_step_prelude,
                range_assign,
                body_fn,
                self._source_info_stmt(
                    body_name, node, construct="dynamic for", region="body"
                ),
                helper_stmt,
            ]
        )
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
        self._validate_dynamic_condition(node.test, "if")
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
        analysis = self.analyze_control_flow(
            node=node,
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
        carried_names = list(analysis.carried_names)

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
        result: list[ast.stmt] = [
            then_fn,
            self._source_info_stmt(
                then_name,
                node.body[0] if node.body else node,
                construct="dynamic if",
                region="then-region",
            ),
        ]
        if else_name is not None:
            result.append(self._branch_function(else_name, carried_names, else_body))
            result.append(
                self._source_info_stmt(
                    else_name,
                    node.orelse[0] if node.orelse else node,
                    construct="dynamic if",
                    region="else-region",
                )
            )

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
        self._validate_dynamic_condition(node.test, "while")
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
        if node.orelse:
            raise SyntaxError("dynamic Tla while does not support while-else")

        analysis = self.analyze_control_flow(
            node=node,
            construct_name="while",
            assigned_regions=[node.body],
            active_call_nodes=[node.test, *node.body],
        )
        _reject_unsupported_dynamic_while_new_defs(
            analysis.active_symbols, analysis.assigned_names, self._following_loads()
        )
        carried_names = list(analysis.carried_names)
        self._scope_manager.add_names_to_scope(carried_names)

        before_body = self._transform_nested_function_body(
            [
                ast.Return(
                    value=ast.List(
                        elts=[node.test, _names_list(carried_names)],
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

        before_info = self._source_info_stmt(
            before_name, node.test, construct="dynamic while", region="condition-region"
        )
        after_info = self._source_info_stmt(
            after_name,
            node.body[0] if node.body else node,
            construct="dynamic while",
            region="body-region",
        )

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
            body=[
                before_fn,
                before_info,
                after_fn,
                after_info,
                ast.Return(value=execute_call),
            ],
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

    def _validate_dynamic_condition(
        self, condition: ast.AST, construct_name: str
    ) -> None:
        _DynamicConditionValidator(
            construct_name,
            filename=self._filename,
            line_offset=self._line_offset,
            source_text=self._source_text,
        ).validate(condition)

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
                args=[
                    test,
                    ast.Call(
                        func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                        args=[
                            then_fn,
                            self._source_info_dict(
                                "<if expression then>",
                                node.body,
                                construct="conditional expression",
                                region="then-region",
                            ),
                        ],
                        keywords=[],
                    ),
                    ast.Call(
                        func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                        args=[
                            else_fn,
                            self._source_info_dict(
                                "<if expression else>",
                                node.orelse,
                                construct="conditional expression",
                                region="else-region",
                            ),
                        ],
                        keywords=[],
                    ),
                ],
                keywords=[],
            ),
            node,
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, (ast.And, ast.Or)) or len(node.values) < 2:
            node = self.generic_visit(node)
            return node
        lhs = self.visit(node.values[0])
        for original_rhs in node.values[1:]:
            with self._lazy_operand("boolean"):
                rhs = self.visit(original_rhs)
            lhs_name = self._fresh("bool_op_lhs")
            bound_lhs = ast.NamedExpr(
                target=ast.Name(id=lhs_name, ctx=ast.Store()),
                value=lhs,
            )
            ast.copy_location(bound_lhs, lhs)
            bound_name = ast.Name(id=lhs_name, ctx=ast.Load())
            if isinstance(node.op, ast.And):
                true_value, false_value = rhs, bound_name
            else:
                true_value, false_value = bound_name, rhs
            lhs = self._lazy_conditional_expression(
                bound_lhs,
                true_value,
                false_value,
                original_rhs,
                construct="boolean expression",
            )
            if isinstance(original_rhs, ast.Constant) and isinstance(
                original_rhs.value, bool
            ):
                if isinstance(node.op, ast.And) and not original_rhs.value:
                    break
                if isinstance(node.op, ast.Or) and original_rhs.value:
                    break
        return lhs

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if self._lazy_operand_context is not None and not isinstance(node.op, ast.Not):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_INTERNAL_LAZY_UNARY, ctx=ast.Load()),
                    args=[ast.Constant(value=type(node.op).__name__), operand],
                    keywords=[],
                ),
                node,
            )
        node.operand = operand
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
        left = self.visit(node.left)
        comparators = []
        for index, comparator in enumerate(node.comparators):
            if index == 0:
                comparators.append(self.visit(comparator))
            else:
                with self._lazy_operand("chained-comparison"):
                    comparators.append(self.visit(comparator))

        def compare_pair(lhs: ast.expr, rhs: ast.expr, op: ast.cmpop) -> ast.Call:
            return ast.Call(
                func=ast.Name(id=_INTERNAL_COMPARE, ctx=ast.Load()),
                args=[
                    lhs,
                    ast.Tuple(elts=[rhs], ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Constant(value=_compare_op_name(op))], ctx=ast.Load()
                    ),
                ],
                keywords=[],
            )

        def build(index: int, lhs: ast.expr) -> ast.expr:
            rhs = comparators[index]
            if index == len(comparators) - 1:
                return compare_pair(lhs, rhs, node.ops[index])
            rhs_name = self._fresh("compare_rhs")
            bound_rhs = ast.NamedExpr(
                target=ast.Name(id=rhs_name, ctx=ast.Store()), value=rhs
            )
            condition_name = self._fresh("compare_result")
            bound_condition = ast.NamedExpr(
                target=ast.Name(id=condition_name, ctx=ast.Store()),
                value=compare_pair(lhs, bound_rhs, node.ops[index]),
            )
            later = build(index + 1, ast.Name(id=rhs_name, ctx=ast.Load()))
            return self._lazy_conditional_expression(
                bound_condition,
                later,
                ast.Name(id=condition_name, ctx=ast.Load()),
                node.comparators[index + 1],
                construct="chained comparison",
            )

        return ast.copy_location(build(0, left), node)

    def _lazy_conditional_expression(
        self,
        condition: ast.expr,
        true_value: ast.expr,
        false_value: ast.expr,
        source_node: ast.AST,
        *,
        construct: str,
    ) -> ast.Call:
        def region(value: ast.expr, name: str) -> ast.Call:
            function = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    vararg=None,
                    kwarg=None,
                ),
                body=value,
            )
            ast.copy_location(function, source_node)
            return ast.Call(
                func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                args=[
                    function,
                    self._source_info_dict(
                        name, source_node, construct=construct, region=name
                    ),
                ],
                keywords=[],
            )

        call = ast.Call(
            func=ast.Name(id=_INTERNAL_IF_EXPR, ctx=ast.Load()),
            args=[
                condition,
                region(true_value, "lazy-true-region"),
                region(false_value, "lazy-false-region"),
            ],
            keywords=[],
        )
        return ast.copy_location(call, source_node)

    def visit_Call(self, node: ast.Call) -> Any:
        if self._lazy_operand_context is not None and not self._is_dsl_call(node):
            original = self.generic_visit(node)
            thunk = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=original,
            )
            ast.copy_location(thunk, node)
            attached = ast.Call(
                func=ast.Name(id=_INTERNAL_ATTACH_SOURCE_INFO, ctx=ast.Load()),
                args=[
                    thunk,
                    self._source_info_dict(
                        "unknown-effect-call",
                        node,
                        construct="runtime-lazy operand",
                        region=self._lazy_operand_context,
                    ),
                ],
                keywords=[],
            )
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_INTERNAL_UNKNOWN_EFFECT_CALL, ctx=ast.Load()),
                    args=[attached],
                    keywords=[],
                ),
                node,
            )
        if not isinstance(node.func, ast.Name):
            node.args = [self.visit(arg) for arg in node.args]
            node.keywords = [self.visit(keyword) for keyword in node.keywords]
            return node
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

    @contextlib.contextmanager
    def _lazy_operand(self, context: str) -> Iterator[None]:
        previous = self._lazy_operand_context
        self._lazy_operand_context = previous or context
        try:
            yield
        finally:
            self._lazy_operand_context = previous

    def _is_dsl_call(self, node: ast.Call) -> bool:
        # Expanding user-provided iterables or mappings invokes arbitrary Python
        # protocols before the trusted call target itself is entered.
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return False
        target = self._resolve_global_call_target(node.func)
        if target is None:
            return False
        return any(target is trusted for trusted in _trusted_lazy_callables())

    def _resolve_global_call_target(self, node: ast.expr) -> Any | None:
        shadowed_names = {
            name
            for scope in self._call_shadow_stack
            for name in scope
        }
        if isinstance(node, ast.Name):
            if node.id in shadowed_names or node.id in _BUILTIN_REDIRECTS:
                return None
            return self._global_symbols.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        attributes: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            attributes.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        if current.id in shadowed_names:
            return None
        target = self._global_symbols.get(current.id)
        for attribute in reversed(attributes):
            try:
                target = inspect.getattr_static(target, attribute)
            except (AttributeError, TypeError):
                members = inspect.getattr_static(target, "_members", None)
                if object.__getattribute__(members, "__class__") is not dict:
                    return None
                if attribute not in members:
                    return None
                target = members[attribute]
        return target

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        if self._lazy_operand_context is None or not isinstance(node.ctx, ast.Load):
            node.value = value
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_LAZY_ATTRIBUTE, ctx=ast.Load()),
                args=[value, ast.Constant(value=node.attr)],
                keywords=[],
            ),
            node,
        )

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        slice_value = self.visit(node.slice)
        if self._lazy_operand_context is None or not isinstance(node.ctx, ast.Load):
            node.value, node.slice = value, slice_value
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_LAZY_SUBSCRIPT, ctx=ast.Load()),
                args=[value, slice_value],
                keywords=[],
            ),
            node,
        )

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        if self._lazy_operand_context is None:
            node.left, node.right = left, right
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_INTERNAL_LAZY_BINOP, ctx=ast.Load()),
                args=[ast.Constant(value=type(node.op).__name__), left, right],
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
        return [
            body_fn,
            self._source_info_stmt(
                body_name,
                node.body[0] if node.body else node,
                construct=f"tla.{region_name}",
                region="region-body",
            ),
            helper_call,
        ]


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
        source_lines, first_lineno = inspect.getsourcelines(fn)
    except (OSError, IOError, TypeError):
        return fn
    source = "".join(source_lines)
    filename = inspect.getsourcefile(fn) or "<unknown>"
    line_offset = int(first_lineno) - 1

    source = textwrap.dedent(source)
    module_ast = ast.parse(source, filename=filename)
    target = _find_function_def(module_ast, fn.__name__)
    if target is None:
        return fn
    if not _function_needs_frontend_transform(target, fn.__globals__):
        return fn

    target.decorator_list = []
    exec_globals = dict(fn.__globals__)
    transformer = _FrontendControlFlowTransformer(
        exec_globals,
        filename=filename,
        line_offset=line_offset,
        source_text=source,
    )
    transformed = transformer.visit(module_ast)
    ast.fix_missing_locations(transformed)
    if line_offset:
        ast.increment_lineno(transformed, line_offset)

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
    exec_globals[transformer.tensor_store_helper_name] = _tensor_store
    exec_globals[_INTERNAL_ATTACH_SOURCE_INFO] = _attach_source_info
    from ..tla_ast_decorators import (
        _internal_lazy_attribute,
        _internal_lazy_binop,
        _internal_lazy_subscript,
        _internal_lazy_unary,
        _internal_unknown_effect_call,
    )

    exec_globals[_INTERNAL_UNKNOWN_EFFECT_CALL] = _internal_unknown_effect_call
    exec_globals[_INTERNAL_LAZY_ATTRIBUTE] = _internal_lazy_attribute
    exec_globals[_INTERNAL_LAZY_SUBSCRIPT] = _internal_lazy_subscript
    exec_globals[_INTERNAL_LAZY_BINOP] = _internal_lazy_binop
    exec_globals[_INTERNAL_LAZY_UNARY] = _internal_lazy_unary
    from .ast_helpers import while_executor, while_selector

    exec_globals[_WHILE_EXECUTOR] = while_executor
    exec_globals[_WHILE_SELECTOR] = while_selector
    exec_globals["__tladsl_builtins__"] = builtins
    namespace: dict[str, Any] = {}
    code = compile(
        transformed,
        filename=filename,
        mode="exec",
    )
    exec(code, exec_globals, namespace)
    rewritten = namespace.get(fn.__name__)
    if not isinstance(rewritten, FunctionType):
        return fn
    rewritten.__defaults__ = fn.__defaults__
    rewritten.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    rewritten.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rewritten.__tladsl_source_info__ = {
        "filename": filename,
        "generated_name": rewritten.__name__,
        "construct": "kernel body",
        "region": "frontend-execution",
    }
    rewritten.__module__ = fn.__module__
    rewritten.__qualname__ = fn.__qualname__
    return rewritten


def _find_function_def(module_ast: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _trusted_lazy_callables() -> tuple[Callable[..., Any], ...]:
    """Return exact frontend call targets allowed to execute in lazy IR regions."""

    return _TRUSTED_LAZY_CALLABLES


def _function_needs_frontend_transform(
    target: ast.FunctionDef, global_symbols: dict[str, Any]
) -> bool:
    """Return whether *target* contains syntax handled by the frontend rewrite."""

    range_names = _tla_function_names_from_globals(global_symbols, "range")
    range_constexpr_names = _tla_function_names_from_globals(
        global_symbols, "range_constexpr"
    )
    module_aliases = _tla_module_aliases_from_globals(global_symbols) | {"tla"}

    class Discovery(ast.NodeVisitor):
        needed = False

        def visit_If(self, node: ast.If) -> None:
            self.needed = True

        def visit_While(self, node: ast.While) -> None:
            self.needed = True

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self.needed = True

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id in {
                "range",
                "range_constexpr",
                *_BUILTIN_REDIRECTS,
                *range_names,
                *range_constexpr_names,
            }:
                self.needed = True
                return
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in module_aliases
                and func.attr in {"range", "range_constexpr", "cube", "vector"}
            ):
                self.needed = True
                return
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "func"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "vec"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in module_aliases
            ):
                self.needed = True
                return
            self.generic_visit(node)

    discovery = Discovery()
    for statement in target.body:
        discovery.visit(statement)
        if discovery.needed:
            break
    return discovery.needed


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
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.has_explicit_step = has_explicit_step


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
    from catlass.base_dsl.typing import as_numeric

    return as_numeric(lhs) + as_numeric(rhs)


def _index_sub(lhs: Any, rhs: Any) -> Any:
    from catlass.base_dsl.typing import as_numeric

    return as_numeric(lhs) - as_numeric(rhs)


def _tensor_store(value: Any, target: Any, index: Any) -> None:
    """Guard a rewritten subscription assignment before invoking tensor storage."""

    from catlass.core_api import _require_category

    _require_category("tensor_store", "target", target, "tensor", 0)
    target[index] = value


def _contains_slice(node: ast.AST) -> bool:
    return any(isinstance(candidate, ast.Slice) for candidate in ast.walk(node))


def _attach_source_info(fn: Any, info: dict[str, Any]) -> Any:
    setattr(fn, _SOURCE_INFO_ATTR, info)
    return fn


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


def _identifier_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            names.add(item.id)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(item.name)
        elif isinstance(item, ast.alias):
            names.add(item.asname or item.name.split(".", 1)[0])
        elif isinstance(item, ast.ExceptHandler) and item.name is not None:
            names.add(item.name)
        elif isinstance(item, (ast.Global, ast.Nonlocal)):
            names.update(item.names)
        elif isinstance(item, (ast.MatchAs, ast.MatchStar)) and item.name is not None:
            names.add(item.name)
        elif isinstance(item, ast.MatchMapping) and item.rest is not None:
            names.add(item.rest)
    return names


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

        def visit_Import(self, import_node: ast.Import) -> None:
            for alias in import_node.names:
                assigned.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, import_node: ast.ImportFrom) -> None:
            for alias in import_node.names:
                if alias.name != "*":
                    assigned.add(alias.asname or alias.name)

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


class _DynamicControlFlowPolicy(ast.NodeVisitor):
    """Shared statement and assignment policy for runtime control flow."""

    def __init__(
        self,
        construct_name: str,
        *,
        filename: str = "<unknown>",
        line_offset: int = 0,
        source_text: str = "",
        is_runtime_for: Callable[[ast.For], bool] | None = None,
        is_static_test: Callable[[ast.AST], bool] | None = None,
    ) -> None:
        self.construct_name = construct_name
        self.filename = filename
        self.line_offset = line_offset
        self.source_lines = source_text.splitlines()
        self.is_runtime_for = is_runtime_for or _is_syntactic_tla_range_for
        self.is_static_test = is_static_test or _is_static_python_if_test
        self.python_loop_depth = 0
        self.assignment_targets: list[str] = []
        self.tensor_store_assignments: set[int] = set()
        self.nested_constructs: list[tuple[str, int, int]] = []

    def visit_Return(self, node: ast.Return) -> None:
        self._unsupported_exit(node)

    def visit_Break(self, node: ast.Break) -> None:
        if self.python_loop_depth == 0:
            self._unsupported_exit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        if self.python_loop_depth == 0:
            self._unsupported_exit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self._unsupported_exit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._raise(node, "does not support deletion")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            if any(isinstance(target, ast.Subscript) for target in node.targets):
                self._raise(node, "does not support chained tensor stores")
            for target in node.targets:
                self._check_local_target(target)
        else:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                self._check_tensor_store(node, target)
            else:
                self._check_local_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._raise(node, "does not support annotated tensor stores")
        self._check_local_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._raise(node, "does not support augmented tensor stores")
        self._check_local_target(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.nested_constructs.append(self._construct_key("for", node))
        if self.is_runtime_for(node):
            return
        self._check_local_target(node.target)
        self.visit(node.iter)
        self._visit_python_loop(node.body, node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self.nested_constructs.append(self._construct_key("while", node))
        if not self.is_static_test(node.test):
            return
        self.visit(node.test)
        self._visit_python_loop(node.body, node.orelse)

    def visit_If(self, node: ast.If) -> None:
        self.nested_constructs.append(self._construct_key("if", node))
        if self.is_static_test(node.test):
            self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._check_local_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        self.generic_visit(node)
        self._raise(node, "does not support try statements")

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self.generic_visit(node)
        self._raise(node, "does not support try-star statements")

    def visit_Match(self, node: ast.Match) -> None:
        self.generic_visit(node)
        self._raise(node, "does not support match statements")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._raise(node, "does not support assignment expressions")

    def visit_Await(self, node: ast.Await) -> None:
        self._raise(node, "does not support await expressions")

    def visit_Yield(self, node: ast.Yield) -> None:
        self._raise(node, "does not support yield expressions")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._raise(node, "does not support yield expressions")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def _check_tensor_store(
        self, assignment: ast.Assign, target: ast.Subscript
    ) -> None:
        if _contains_slice(target.slice):
            self._raise(target, "does not support tensor slice assignment")
        self.assignment_targets.append("tensor subscript")
        self.tensor_store_assignments.add(id(assignment))
        self.visit(target.value)
        self.visit(target.slice)

    def _check_local_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.assignment_targets.append(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and all(
            isinstance(element, ast.Name) for element in target.elts
        ):
            self.assignment_targets.extend(element.id for element in target.elts)
            return
        self._raise(
            target,
            "only supports assignments to local names, tuples/lists of local "
            "names, or Catlass tensor elements",
        )

    def _unsupported_exit(self, node: ast.AST) -> None:
        self._raise(node, "does not support return, break, continue, or raise")

    def _visit_python_loop(
        self, body: list[ast.stmt], orelse: list[ast.stmt]
    ) -> None:
        self.python_loop_depth += 1
        try:
            for statement in [*body, *orelse]:
                self.visit(statement)
        finally:
            self.python_loop_depth -= 1

    def _raise(self, node: ast.AST, message: str) -> None:
        error = SyntaxError(f"dynamic Tla {self.construct_name} {message}")
        relative_lineno = int(getattr(node, "lineno", 0) or 0)
        error.filename = self.filename
        error.lineno = self.line_offset + relative_lineno
        error.offset = (
            int(getattr(node, "col_offset", 0)) + 1
            if getattr(node, "col_offset", None) is not None
            else None
        )
        if 0 < relative_lineno <= len(self.source_lines):
            error.text = self.source_lines[relative_lineno - 1]
        raise error

    @staticmethod
    def _construct_key(kind: str, node: ast.AST) -> tuple[str, int, int]:
        return (
            kind,
            int(getattr(node, "lineno", 0) or 0),
            int(getattr(node, "col_offset", 0) or 0),
        )


def _is_syntactic_tla_range_for(node: ast.For) -> bool:
    if not isinstance(node.iter, ast.Call):
        return False
    func = node.iter.func
    if isinstance(func, ast.Name):
        return func.id == "tla_range"
    if isinstance(func, ast.Attribute):
        return func.attr == "range" and isinstance(func.value, ast.Name)
    return False


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
