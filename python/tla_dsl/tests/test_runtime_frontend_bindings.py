from __future__ import annotations

from catlass import runtime


def test_frontend_value_binding_requires_exact_registered_object() -> None:
    registered = object()
    query = object()
    bindings = {id(query): (registered, "bound-value")}

    with runtime._frontend_emission(arg_bindings=bindings):
        assert runtime._resolve_frontend_bound_value(query) is None


def test_frontend_category_binding_requires_exact_registered_object() -> None:
    registered = object()
    query = object()
    bindings = {id(query): (registered, "tensor")}

    with runtime._frontend_emission(category_bindings=bindings):
        assert runtime._resolve_frontend_bound_category(query) is None


def test_frontend_bindings_resolve_exact_registered_object() -> None:
    registered = object()
    bound_value = object()

    with runtime._frontend_emission() as state:
        runtime._bind_frontend_value(registered, bound_value)
        runtime._bind_frontend_category(registered, "tensor")

        assert state.arg_bindings[id(registered)][0] is registered
        assert state.category_bindings[id(registered)][0] is registered
        assert runtime._resolve_frontend_bound_value(registered) is bound_value
        assert runtime._resolve_frontend_bound_category(registered) == "tensor"
