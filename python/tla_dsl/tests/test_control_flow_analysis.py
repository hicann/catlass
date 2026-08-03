from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError

import pytest
import catlass.core_api as core_api
import catlass.tla_ast_decorators as ast_decorators

from catlass.base_dsl.ast_preprocessor import (
    _ControlFlowAnalyzer,
    _DynamicConditionValidator,
    _FrontendControlFlowTransformer,
    maybe_transform_for_lowering,
)
from catlass.tla_ast_decorators import (
    _dynamic_lazy_region,
    _internal_frontend_compare_pair,
    _internal_frontend_if_expr,
    _internal_frontend_if,
    _internal_frontend_bool_not,
    _internal_lazy_attribute,
    _internal_lazy_binop,
    _internal_lazy_subscript,
    _internal_lazy_unary,
    _internal_unknown_effect_call,
)


def _parse_first_statement(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def _validate_condition(source: str, *, construct: str = "if") -> None:
    node = _parse_first_statement(source)
    assert isinstance(node, (ast.If, ast.While))
    _DynamicConditionValidator(
        construct,
        filename="condition_kernel.py",
        line_offset=40,
        source_text=source,
    ).validate(node.test)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "if (bound := predicate):\n    pass\n",
            "does not support assignment expressions in its condition",
        ),
        (
            "if await predicate:\n    pass\n",
            "does not support await expressions in its condition",
        ),
        (
            "if (yield predicate):\n    pass\n",
            "does not support yield expressions in its condition",
        ),
        (
            "if (yield from predicates):\n    pass\n",
            "does not support yield expressions in its condition",
        ),
        (
            "if (lambda: predicate)():\n    pass\n",
            "does not support lambda expressions in its condition",
        ),
        (
            "if [item for item in values]:\n    pass\n",
            "does not support comprehension expressions in its condition",
        ),
        (
            "if {item for item in values}:\n    pass\n",
            "does not support comprehension expressions in its condition",
        ),
        (
            "if {item: item for item in values}:\n    pass\n",
            "does not support comprehension expressions in its condition",
        ),
        (
            "if (item for item in values):\n    pass\n",
            "does not support generator expressions in its condition",
        ),
    ],
    ids=[
        "walrus",
        "await",
        "yield",
        "yield-from",
        "lambda",
        "list-comprehension",
        "set-comprehension",
        "dict-comprehension",
        "generator-expression",
    ],
)
def test_dynamic_condition_rejects_unsupported_syntax(
    source: str, message: str
) -> None:
    with pytest.raises(SyntaxError, match=message):
        _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if False and (lambda: predicate)():\n    pass\n",
        "if True or [item for item in values]:\n    pass\n",
        "if staged_false() and (item for item in values):\n    pass\n",
        "if (1,) is (1,) is (lambda: predicate)():\n    pass\n",
        "while False and (bound := predicate):\n    pass\n",
    ],
)
def test_forbidden_condition_syntax_is_rejected_even_when_unreachable(
    source: str,
) -> None:
    construct = "while" if source.startswith("while") else "if"
    with pytest.raises(SyntaxError, match="dynamic Tla"):
        _validate_condition(source, construct=construct)


def _compact_if_for_transform_discovery(predicate: object) -> None:
    if(predicate):
        pass


def _compact_while_for_transform_discovery(predicate: object) -> None:
    while(predicate):
        predicate = False


def _irrelevant_function_for_transform_discovery(value: object) -> object:
    return value


def _maybe_transform(fn):
    passthrough = lambda *args, **kwargs: None
    return maybe_transform_for_lowering(
        fn,
        internal_for=passthrough,
        internal_region=passthrough,
        internal_if=passthrough,
        internal_if_expr=passthrough,
        internal_bool_and=passthrough,
        internal_bool_or=passthrough,
        internal_bool_not=passthrough,
        internal_compare=passthrough,
        internal_any=passthrough,
        internal_all=passthrough,
        internal_bool=passthrough,
        internal_min=passthrough,
        internal_max=passthrough,
    )


@pytest.mark.parametrize(
    "fn",
    [_compact_if_for_transform_discovery, _compact_while_for_transform_discovery],
)
def test_ast_discovery_transforms_compact_control_flow(fn) -> None:
    assert _maybe_transform(fn) is not fn


def test_ast_discovery_does_not_transform_irrelevant_function() -> None:
    assert (
        _maybe_transform(_irrelevant_function_for_transform_discovery)
        is _irrelevant_function_for_transform_discovery
    )


class _ProtocolSpy:
    calls = 0

    def _called(self):
        type(self).calls += 1
        return True

    __bool__ = _called
    __len__ = _called
    __getattr__ = lambda self, _name: self._called()
    __getitem__ = lambda self, _key: self._called()
    __add__ = lambda self, _other: self._called()
    __neg__ = _called
    __eq__ = lambda self, _other: self._called()
    __contains__ = lambda self, _item: self._called()
    __hash__ = _called
    __iter__ = _called


@pytest.mark.parametrize(
    "operation",
    [
        lambda spy: _internal_lazy_attribute(spy, "field"),
        lambda spy: _internal_lazy_subscript(spy, 0),
        lambda spy: _internal_lazy_binop("Add", spy, 1),
        lambda spy: _internal_lazy_unary("USub", spy),
        lambda spy: _internal_frontend_compare_pair(spy, 1, "=="),
        lambda spy: _internal_frontend_compare_pair(1, spy, "in"),
        lambda spy: _internal_frontend_compare_pair(spy, {1}, "in"),
        lambda spy: _internal_frontend_if_expr(spy, lambda: 1, lambda: 0),
    ],
    ids=[
        "attribute",
        "subscript",
        "binary",
        "unary",
        "compare",
        "membership",
        "hash-membership",
        "truth",
    ],
)
def test_runtime_lazy_protocols_are_rejected_before_user_code(operation) -> None:
    _ProtocolSpy.calls = 0
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        operation(_ProtocolSpy())
    assert _ProtocolSpy.calls == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda spy: _internal_frontend_if(spy, lambda: None, lambda: None),
        lambda spy: _internal_frontend_if_expr(spy, lambda: 1, lambda: 0),
    ],
    ids=["statement-if", "if-expression"],
)
def test_outer_condition_rejects_custom_truthiness_before_user_code(operation) -> None:
    _ProtocolSpy.calls = 0
    with pytest.raises(SyntaxError, match="effect unknown"):
        operation(_ProtocolSpy())
    assert _ProtocolSpy.calls == 0


def test_boolean_not_rejects_custom_truthiness_before_user_code() -> None:
    _ProtocolSpy.calls = 0
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_frontend_bool_not(_ProtocolSpy())
    assert _ProtocolSpy.calls == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (0, True), (1, False), ("", True), ([], True), ([1], False)],
)
def test_boolean_not_stages_exact_builtin_truthiness(value, expected) -> None:
    assert _internal_frontend_bool_not(value) is expected


def test_dynamic_lazy_context_resets_after_unknown_effect_failure() -> None:
    calls: list[str] = []
    with pytest.raises(SyntaxError, match="effect unknown"):
        with _dynamic_lazy_region():
            _internal_unknown_effect_call(lambda: calls.append("dynamic"))
    assert calls == []
    _internal_unknown_effect_call(lambda: calls.append("staged"))
    assert calls == ["staged"]


def test_dynamic_lazy_context_is_reentrant_and_restores_each_scope() -> None:
    calls: list[str] = []
    with _dynamic_lazy_region():
        with pytest.raises(SyntaxError, match="effect unknown"):
            _internal_unknown_effect_call(lambda: calls.append("outer"))
        with _dynamic_lazy_region():
            with pytest.raises(SyntaxError, match="effect unknown"):
                _internal_unknown_effect_call(lambda: calls.append("inner"))
        with pytest.raises(SyntaxError, match="effect unknown"):
            _internal_unknown_effect_call(lambda: calls.append("outer-again"))
    _internal_unknown_effect_call(lambda: calls.append("staged"))
    assert calls == ["staged"]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda: _internal_lazy_subscript([2, 3], 1), 3),
        (lambda: _internal_lazy_subscript({"key": 4}, "key"), 4),
        (lambda: _internal_lazy_binop("Add", 2, 3), 5),
        (lambda: _internal_lazy_unary("USub", 2), -2),
        (lambda: _internal_frontend_compare_pair((1, 2), (1, 3), "<"), True),
        (lambda: _internal_frontend_compare_pair(2, (1, 2), "in"), True),
        (lambda: _internal_frontend_if_expr([], lambda: "yes", lambda: "no"), "no"),
    ],
)
def test_runtime_lazy_exact_builtin_operations_are_allowed(operation, expected) -> None:
    with _dynamic_lazy_region():
        assert operation() == expected


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (None, "false"),
        (0, "false"),
        (1, "true"),
        ("", "false"),
        ("value", "true"),
        ([], "false"),
        ([1], "true"),
        ({}, "false"),
        ({"key": 1}, "true"),
    ],
)
def test_outer_if_expression_stages_exact_builtin_truthiness(
    condition, expected
) -> None:
    calls: list[str] = []

    def selected(value: str):
        return lambda: calls.append(value) or value

    result = _internal_frontend_if_expr(
        condition, selected("true"), selected("false")
    )
    assert result == expected
    assert calls == [expected]


@pytest.mark.parametrize("condition", [None, 0, "", [], {}, set()])
def test_outer_statement_if_stages_false_exact_builtin_truthiness(condition) -> None:
    calls: list[str] = []
    result = _internal_frontend_if(
        condition,
        lambda: calls.append("then"),
        lambda: calls.append("else") or 7,
    )
    assert result is None
    assert calls == ["else"]


@pytest.mark.parametrize(
    ("value", "name", "expected"),
    [
        (1, "real", 1),
        ("abc", "upper", str.upper),
        ([], "append", list.append),
        ({}, "keys", dict.keys),
    ],
)
def test_runtime_lazy_exact_builtin_attribute_reads_are_allowed(
    value, name, expected
) -> None:
    with _dynamic_lazy_region():
        result = _internal_lazy_attribute(value, name)
    if callable(expected):
        assert result.__self__ is value
        assert result.__name__ == expected.__name__
    else:
        assert result == expected


def test_runtime_lazy_exact_builtin_missing_attribute_raises_naturally() -> None:
    with _dynamic_lazy_region(), pytest.raises(AttributeError):
        _internal_lazy_attribute([], "missing")


def test_exact_builtin_attribute_does_not_inspect_custom_elements() -> None:
    _ProtocolSpy.calls = 0
    value = [_ProtocolSpy()]
    with _dynamic_lazy_region():
        result = _internal_lazy_attribute(value, "append")
    assert result.__self__ is value
    assert _ProtocolSpy.calls == 0


def test_runtime_lazy_builtin_subclass_is_rejected_without_protocol_dispatch() -> None:
    class ListSubclass(list):
        def __getitem__(self, key):
            _ProtocolSpy.calls += 1
            return super().__getitem__(key)

    _ProtocolSpy.calls = 0
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_lazy_subscript(ListSubclass([1]), 0)
    assert _ProtocolSpy.calls == 0


def test_runtime_lazy_slice_rejects_custom_bounds_without_index_dispatch() -> None:
    class IndexSpy:
        calls = 0

        def __index__(self):
            type(self).calls += 1
            return 0

    bound = IndexSpy()
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_lazy_subscript([1], slice(bound))
    assert IndexSpy.calls == 0


def test_runtime_lazy_dsl_subscript_rejects_tuple_subclass_without_iteration(
    monkeypatch,
) -> None:
    class TupleSpy(tuple):
        calls = 0

        def __iter__(self):
            type(self).calls += 1
            return super().__iter__()

    class TrustedValue:
        def __getitem__(self, index):
            tuple(index)
            return 1

    value = TrustedValue()
    monkeypatch.setattr(
        ast_decorators,
        "_is_trusted_dsl_value",
        lambda candidate: candidate is value,
    )
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_lazy_subscript(value, TupleSpy((0,)))
    assert TupleSpy.calls == 0


def _transform_source(source: str, globals_: dict[str, object]) -> str:
    tree = ast.parse(source)
    transformed = _FrontendControlFlowTransformer(
        globals_, source_text=source
    ).visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def test_runtime_lazy_call_trust_uses_exact_identity_not_metadata() -> None:
    def forged():
        return True

    forged.__module__ = "catlass.core_api"
    forged.__name__ = "block_idx"
    rendered = _transform_source(
        "def kernel(p):\n    if p and forged():\n        pass\n", {"forged": forged}
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


def test_runtime_lazy_call_trust_ignores_module_namespace_injection(
    monkeypatch,
) -> None:
    def forged():
        return True

    forged.__module__ = "catlass.core_api"
    monkeypatch.setattr(core_api, "forged", forged, raising=False)
    rendered = _transform_source(
        "def kernel(p):\n    if p and core.forged():\n        pass\n",
        {"core": core_api},
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


def test_runtime_lazy_value_trust_ignores_module_namespace_injection(
    monkeypatch,
) -> None:
    class Forged:
        calls = 0

        def __bool__(self):
            type(self).calls += 1
            return True

    Forged.__module__ = "catlass.core_api"
    monkeypatch.setattr(core_api, "Forged", Forged, raising=False)
    with _dynamic_lazy_region(), pytest.raises(SyntaxError, match="effect unknown"):
        _internal_frontend_if_expr(Forged(), lambda: 1, lambda: 0)
    assert Forged.calls == 0


def test_global_call_resolution_does_not_dispatch_custom_member_mapping() -> None:
    class MemberSpy(dict):
        calls = 0

        def __contains__(self, key):
            type(self).calls += 1
            return super().__contains__(key)

        def __getitem__(self, key):
            type(self).calls += 1
            return super().__getitem__(key)

    class Owner:
        _members = MemberSpy(block=core_api.arch.block_idx)

    rendered = _transform_source(
        "def kernel(p):\n    if p and owner.block():\n        pass\n",
        {"owner": Owner()},
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered
    assert MemberSpy.calls == 0


@pytest.mark.parametrize("name", ["bool", "any", "all", "min", "max"])
def test_redirected_builtin_is_rejected_only_in_runtime_lazy_operand(name: str) -> None:
    rendered = _transform_source(
        f"def kernel(p):\n    if p and {name}([p]):\n        pass\n", {}
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


def test_exact_core_callable_global_alias_is_trusted() -> None:
    rendered = _transform_source(
        "def kernel(p):\n    if p and block() >= 0:\n        pass\n",
        {"block": core_api.arch.block_idx},
    )
    assert "__tladsl_internal_unknown_effect_call__" not in rendered


@pytest.mark.parametrize(
    "call",
    ["block(*values)", "block(**values)", "block(value, *more)"],
)
def test_trusted_dsl_call_with_argument_expansion_is_effect_unknown(call: str) -> None:
    rendered = _transform_source(
        f"def kernel(p, values, more):\n    if p and {call}:\n        pass\n",
        {"block": core_api.arch.block_idx},
    )
    assert "__tladsl_internal_unknown_effect_call__" in rendered


@pytest.mark.parametrize(
    "source",
    [
        "def kernel(p, block):\n    if p and block():\n        pass\n",
        "def kernel(p):\n    block = lambda: True\n    if p and block():\n        pass\n",
        "def kernel(p):\n    if p and block():\n        pass\n    block = lambda: True\n",
        "def outer():\n    block = lambda: True\n    def kernel(p):\n        if p and block():\n            pass\n",
        "def kernel(p):\n    from somewhere import block\n    if p and block():\n        pass\n",
    ],
    ids=["parameter", "assignment", "assignment-after", "closure", "local-import"],
)
def test_local_or_free_name_shadows_trusted_global_callable(source: str) -> None:
    rendered = _transform_source(source, {"block": core_api.arch.block_idx})
    assert "__tladsl_internal_unknown_effect_call__" in rendered


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "if predicate and staged_call():\n    pass\n",
            "conditionally skipped boolean operand has trace-time effect unknown",
        ),
        (
            "if lower < value < staged_call():\n    pass\n",
            "conditionally skipped chained-comparison operand has trace-time effect unknown",
        ),
    ],
    ids=["bool-call", "compare-call"],
)
def test_dynamic_condition_defers_unknown_lazy_call_effects(
    source: str, message: str
) -> None:
    del message
    _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if predicate or owner.attribute:\n    pass\n",
        "if predicate and values[index]:\n    pass\n",
        "if lower < value < owner.attribute:\n    pass\n",
        "if lower < value < values[index]:\n    pass\n",
    ],
)
def test_dynamic_condition_allows_lazy_reads(source: str) -> None:
    _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if lower < value and value < upper:\n    pass\n",
        "if not predicate or fallback:\n    pass\n",
        "if staged_call() and predicate:\n    pass\n",
        "if lower < staged_call() < upper:\n    pass\n",
        "while lower < value <= upper:\n    pass\n",
    ],
    ids=[
        "pure-comparisons",
        "pure-booleans",
        "eager-simple-call",
        "eager-first-comparator-call",
        "while-pure-chain",
    ],
)
def test_dynamic_condition_allows_pure_expressions_and_eager_calls(
    source: str,
) -> None:
    construct = "while" if source.startswith("while") else "if"
    _validate_condition(source, construct=construct)


def test_dynamic_condition_defers_source_located_unknown_call_diagnostic() -> None:
    source = "if (\n    predicate\n    and staged_call()\n):\n    pass\n"
    _validate_condition(source)


@pytest.mark.parametrize(
    "source",
    [
        "if (bound := predicate):\n    pass\n",
    ],
    ids=["if-walrus"],
)
def test_frontend_transformer_validates_each_dynamic_condition(source: str) -> None:
    node = _parse_first_statement(source)
    transformer = _FrontendControlFlowTransformer(
        {},
        filename="condition_kernel.py",
        source_text=source,
    )

    with pytest.raises(SyntaxError, match="dynamic Tla (if|while)"):
        transformer.visit(node)


@pytest.mark.parametrize(
    ("source", "kind", "regions"),
    [
        ("if predicate:\n    state = state + 1\n", "if", ("body", "orelse")),
        ("for i in tla.range(limit):\n    state = i\n", "for", ("body",)),
        ("while state < limit:\n    state = state + 1\n", "while", ("body",)),
    ],
)
def test_control_flow_analyzer_builds_immutable_plans(
    source: str, kind: str, regions: tuple[str, ...]
) -> None:
    node = _parse_first_statement(source)
    assert isinstance(node, (ast.If, ast.For, ast.While))
    assigned_regions = [getattr(node, region) for region in regions]

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name=kind,
        assigned_regions=assigned_regions,
        active_call_nodes=[node],
        active_symbols={"predicate", "limit", "state"},
        active_callables=set(),
    )

    assert plan.construct_name == kind
    assert plan.assigned_names == frozenset({"state"})
    assert plan.carried_names == ("state",)
    with pytest.raises(FrozenInstanceError):
        plan.construct_name = "changed"  # type: ignore[misc]


def test_control_flow_analyzer_classifies_tensor_store_as_side_effect() -> None:
    node = _parse_first_statement("if predicate:\n    out[index] = value\n")
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "out", "index", "value"},
        active_callables=set(),
    )

    assert plan.assignment_targets == ("tensor subscript",)
    assert len(plan.tensor_store_assignments) == 1
    assert plan.assigned_names == frozenset()
    assert plan.carried_names == ()


def test_control_flow_analyzer_collects_tensor_store_in_python_loop() -> None:
    node = _parse_first_statement(
        "if predicate:\n"
        "    for _ in (0,):\n"
        "        out[index] = value\n"
    )
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "out", "index", "value"},
        active_callables=set(),
    )

    assert plan.assignment_targets == ("_", "tensor subscript")
    assert len(plan.tensor_store_assignments) == 1


def test_control_flow_analyzer_records_nested_runtime_constructs() -> None:
    node = _parse_first_statement(
        "while state < limit:\n"
        "    if predicate:\n"
        "        state = state + 1\n"
        "    for i in tla.range(limit):\n"
        "        consume(i)\n"
    )
    assert isinstance(node, ast.While)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="while",
        assigned_regions=[node.body],
        active_call_nodes=[node.test, *node.body],
        active_symbols={"state", "limit", "predicate"},
        active_callables=set(),
    )

    assert [kind for kind, _, _ in plan.nested_constructs] == ["if", "for"]


@pytest.mark.parametrize(
    "nested_source",
    [
        "if True:\n    return\n",
        "for _ in range(1):\n    return\n",
        "while True:\n    return\n",
        "with context:\n    return\n",
        "try:\n    return\nexcept Exception:\n    pass\n",
        "match value:\n    case _:\n        return\n",
    ],
    ids=["if", "for", "while", "with", "try", "match"],
)
def test_control_flow_analyzer_recursively_rejects_nested_return(
    nested_source: str,
) -> None:
    indented = "".join(f"    {line}\n" for line in nested_source.splitlines())
    node = _parse_first_statement(f"if predicate:\n{indented}")
    assert isinstance(node, ast.If)

    with pytest.raises(
        SyntaxError, match="dynamic Tla if does not support return"
    ):
        _ControlFlowAnalyzer().analyze(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[node],
            active_symbols={"predicate", "context", "value"},
            active_callables=set(),
        )


def test_control_flow_analyzer_does_not_cross_nested_function_scope() -> None:
    node = _parse_first_statement(
        "if predicate:\n"
        "    def helper():\n"
        "        return\n"
    )
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate"},
        active_callables=set(),
    )

    assert plan.construct_name == "if"


def test_control_flow_analyzer_allows_python_loop_owned_exits() -> None:
    node = _parse_first_statement(
        "if predicate:\n"
        "    for item in values:\n"
        "        continue\n"
        "    for item in values:\n"
        "        break\n"
    )
    assert isinstance(node, ast.If)

    plan = _ControlFlowAnalyzer().analyze(
        node=node,
        construct_name="if",
        assigned_regions=[node.body, node.orelse],
        active_call_nodes=[node],
        active_symbols={"predicate", "values"},
        active_callables=set(),
    )

    assert plan.assignment_targets == ("item", "item")


@pytest.mark.parametrize(
    "source",
    [
        "if predicate:\n    for owner.attribute in values:\n        pass\n",
        "if predicate:\n    with context as owner.attribute:\n        pass\n",
    ],
    ids=["for-target", "with-target"],
)
def test_control_flow_analyzer_rejects_nonlocal_binding_targets(
    source: str,
) -> None:
    node = _parse_first_statement(source)
    assert isinstance(node, ast.If)

    with pytest.raises(SyntaxError, match="only supports assignments to local names"):
        _ControlFlowAnalyzer().analyze(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[node],
            active_symbols={"predicate", "values", "context", "owner"},
            active_callables=set(),
        )


@pytest.mark.parametrize(
    ("nested_source", "message"),
    [
        (
            "try:\n    consume()\nexcept Exception:\n    recover()\n",
            "does not support try statements",
        ),
        (
            "match value:\n    case _:\n        consume()\n",
            "does not support match statements",
        ),
        ("item = (updated := value)\n", "does not support assignment expressions"),
    ],
    ids=["try", "match", "assignment-expression"],
)
def test_control_flow_analyzer_rejects_unsupported_nested_syntax(
    nested_source: str, message: str
) -> None:
    indented = "".join(f"    {line}\n" for line in nested_source.splitlines())
    node = _parse_first_statement(f"if predicate:\n{indented}")
    assert isinstance(node, ast.If)

    with pytest.raises(SyntaxError, match=message):
        _ControlFlowAnalyzer().analyze(
            node=node,
            construct_name="if",
            assigned_regions=[node.body, node.orelse],
            active_call_nodes=[node],
            active_symbols={"predicate", "value"},
            active_callables=set(),
        )
