"""Internal classification and type-resolution contracts for argument trees."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass import core_api
from catlass._mlir import ir as mlir_ir
from catlass.base_dsl.runtime.argument_tree import build_runtime_argument_tree


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_dynamic_gm_predicate_errors_are_not_reclassified_as_scalars(
    monkeypatch, error_type
):
    failure = error_type("broken Dynamic-GM predicate")

    def fail_classification(value):
        raise failure

    monkeypatch.setattr(core_api, "is_dynamic_gm_tensor_arg", fail_classification)
    with pytest.raises(error_type, match="broken Dynamic-GM predicate") as error:
        build_runtime_argument_tree(1)
    assert error.value is failure


@pytest.mark.parametrize("explicit_context", [False, True])
def test_numeric_tree_uses_declared_protocol_with_a_real_mlir_context(
    monkeypatch, explicit_context
):
    calls = []
    get_mlir_types = tla.Int32.__get_mlir_types__

    def record_types(value, context):
        calls.append(context)
        return get_mlir_types(value, context)

    monkeypatch.setattr(tla.Int32, "__get_mlir_types__", record_types)
    with mlir_ir.Context() as context:
        supplied_context = context if explicit_context else None
        tree = build_runtime_argument_tree(tla.Int32(7), supplied_context)
        assert calls == [supplied_context]
        assert tree.runtime_leaf_count == 1
        assert tree.bindings[0].kind == "scalar"
        assert isinstance(tree.bindings[0].mlir_type, mlir_ir.Type)
        assert tree.bindings[0].mlir_type == tla.Int32.mlir_type(context)


@pytest.mark.parametrize("error_type", [TypeError, ValueError, AssertionError])
def test_numeric_type_factory_errors_do_not_fall_back_to_dtype(
    monkeypatch, error_type
):
    failure = error_type("broken Numeric MLIR type factory")

    def fail_type_factory():
        raise failure

    monkeypatch.setattr(tla.Int32, "_mlir_type_factory", staticmethod(fail_type_factory))
    with mlir_ir.Context():
        with pytest.raises(
            error_type, match="broken Numeric MLIR type factory"
        ) as error:
            build_runtime_argument_tree(tla.Int32(7))
    assert error.value is failure
