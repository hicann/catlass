"""PR-4 coverage for the Python staging boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import Any

import pytest

import catlass.tla as tla
from catlass import execution_lowering
from catlass.base_dsl.ast_preprocessor import reject_user_class_value


class _CompileTimeBox:
    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def doubled(self) -> int:
        return self.value * 2

    def read(self) -> int:
        return self.value


_BOX = _CompileTimeBox(3)


def _plain_helper(value: int) -> int:
    return value + 1


def _make_closure(offset: int):
    def helper(value: int) -> int:
        return value + offset

    return helper


_CLOSURE = _make_closure(2)


def _emit_coord(value: Any) -> None:
    """Bare staging helper that emits into the caller's active IR region."""

    tla.make_coord(value, 0)


def _dynamic_helper_template(value: Any) -> None:
    tla.make_coord(value, 0)


_DYNAMIC_SOURCE_HELPER = FunctionType(
    _dynamic_helper_template.__code__.replace(co_filename="<dynamic-python-helper>"),
    globals(),
    "dynamic_source_helper",
)


@tla.kernel
def staging_values_kernel(limit: int) -> None:
    def nested(value: int) -> int:
        return value + 1

    value = _plain_helper(_CLOSURE(nested(_BOX.read())))
    tla.make_coord(value + _BOX.doubled, limit)


@tla.kernel
def class_abi_kernel(value: _CompileTimeBox) -> None:
    tla.make_coord(value.read(), 0)


@dataclass
class _KernelArgDataclass:
    value: int


@dataclass
class _NestedKernelArgDataclass:
    value: _KernelArgDataclass


def test_kernel_arg_dataclass_values_are_allowed() -> None:
    reject_user_class_value(
        _NestedKernelArgDataclass(_KernelArgDataclass(1)),
        context="kernel argument 'value'",
    )


def test_kernel_arg_dataclass_rejects_user_object_fields() -> None:
    with pytest.raises(TypeError, match="_CompileTimeBox"):
        reject_user_class_value(
            _KernelArgDataclass(_CompileTimeBox(1)),
            context="kernel argument 'value'",
        )


@tla.kernel
def user_object_runtime_escape_kernel(limit: int) -> None:
    value = _BOX
    if limit > 0:
        value = tla.as_numeric(value)
    tla.make_coord(0, 0)


@tla.kernel
def object_mutation_in_runtime_kernel(limit: int) -> None:
    value = _CompileTimeBox(0)
    if limit > 0:
        value.value = 1


@tla.jit
def helper_requiring_transformation(value: int) -> None:
    if value > 0:
        tla.make_coord(value, 0)


@tla.kernel
def helper_requiring_transformation_kernel(value: int) -> None:
    helper_requiring_transformation(value)


def _make_dynamic_jit_helper(offset: int):
    @tla.jit
    def helper(value: int) -> None:
        if value > 0:
            tla.make_coord(offset, 0)

    return helper


@tla.kernel
def dynamic_jit_helpers_kernel(value: int) -> None:
    _make_dynamic_jit_helper(11)(value)
    _make_dynamic_jit_helper(22)(value)


def plain_helper_requiring_transformation(value: int) -> None:
    for index in tla.range(value):
        tla.make_coord(index, 0)


@tla.kernel
def plain_helper_requiring_transformation_kernel(value: int) -> None:
    plain_helper_requiring_transformation(value)


def plain_helper_with_static_if(value: int) -> int:
    if True:
        return value + 1
    return value


@tla.jit
def jit_helper_with_static_if(value: int) -> int:
    if False:
        return value
    return value + 2


@tla.kernel
def helpers_with_static_if_kernel(value: int) -> None:
    value = plain_helper_with_static_if(value)
    tla.make_coord(jit_helper_with_static_if(value), 0)


@tla.kernel
def staging_local_nested_function_kernel(limit: int) -> None:
    def helper(value: int) -> int:
        return value + 1

    value = helper(2)
    if limit > 0:
        value = tla.as_numeric(value)
    tla.make_coord(value, 0)


@tla.kernel
def bare_helper_in_runtime_if_kernel(limit: int) -> None:
    def emit(value: Any) -> None:
        tla.make_coord(value, 0)

    if limit > 0:
        emit(limit)


@tla.kernel
def bare_helper_in_runtime_for_kernel(limit: int) -> None:
    for index in tla.range(limit):
        _emit_coord(index)


@tla.kernel
def bare_helper_in_runtime_while_kernel(limit: int) -> None:
    index = tla.as_numeric(0)
    while index < limit:
        _emit_coord(index)
        index += 1


@tla.kernel
def bare_helper_in_runtime_with_kernel(limit: int) -> None:
    def emit(value: Any) -> None:
        tla.make_coord(value, 0)

    with tla.vector():
        emit(limit)


@tla.kernel
def aliased_bare_helper_in_runtime_if_kernel(limit: int) -> None:
    def emit(value: Any) -> Any:
        tla.make_coord(value, 0)
        return value

    helper = emit
    if limit > 0:
        tla.make_coord(helper(limit), 0)


@tla.kernel
def aliased_bare_helper_in_runtime_for_kernel(limit: int) -> None:
    def emit(value: Any) -> Any:
        tla.make_coord(value, 0)
        return value

    helper = emit
    for index in tla.range(limit):
        tla.make_coord(helper(index), 0)


@tla.kernel
def aliased_bare_helper_in_runtime_while_kernel(limit: int) -> None:
    def emit(value: Any) -> Any:
        tla.make_coord(value, 0)
        return value

    helper = emit
    index = tla.as_numeric(0)
    while index < limit:
        tla.make_coord(helper(index), 0)
        index += 1


@tla.kernel
def aliased_bare_helper_in_runtime_with_kernel(limit: int) -> None:
    def emit(value: Any) -> Any:
        tla.make_coord(value, 0)
        return value

    helper = emit
    with tla.vector():
        tla.make_coord(helper(limit), 0)


@tla.kernel
def nonlocal_bare_helper_in_runtime_if_kernel(limit: int) -> None:
    value = 0

    def update() -> None:
        nonlocal value
        value = limit

    if limit > 0:
        update()
    tla.make_coord(value, 0)


_STAGING_GLOBAL = 0


@tla.kernel
def global_bare_helper_in_runtime_if_kernel(limit: int) -> None:
    def update() -> None:
        global _STAGING_GLOBAL
        _STAGING_GLOBAL = limit

    if limit > 0:
        update()


@tla.kernel
def jit_calls_bare_helper(value: int) -> None:
    _emit_coord(value)


@tla.kernel
def dynamic_source_helper_kernel(value: int) -> None:
    _DYNAMIC_SOURCE_HELPER(value)


_HOST_EVENTS: list[str] = []


@tla.kernel
def host_launch_kernel() -> None:
    tla.make_coord(0, 0)


@tla.jit
def host_orchestration() -> object:
    _HOST_EVENTS.append("orchestrated")
    return host_launch_kernel


def test_inspectable_python_staging_values_lower() -> None:
    mlir = staging_values_kernel.dump_mlir(type_args=(2,))
    assert "tla.make_coord" in mlir


def test_user_class_kernel_abi_is_rejected() -> None:
    with pytest.raises(Exception, match="unsupported|_CompileTimeBox|ABI"):
        class_abi_kernel.dump_mlir(type_args=(_CompileTimeBox(1),))


def test_user_class_cannot_be_promoted_with_as_numeric() -> None:
    with pytest.raises(Exception, match="as_numeric|_CompileTimeBox|unsupported"):
        user_object_runtime_escape_kernel.dump_mlir(type_args=(1,))


def test_object_mutation_inside_runtime_control_flow_is_rejected() -> None:
    with pytest.raises(SyntaxError, match="only supports assignments"):
        object_mutation_in_runtime_kernel.dump_mlir(type_args=(1,))


def test_decorated_helper_control_flow_is_transformed() -> None:
    mlir = helper_requiring_transformation_kernel.dump_mlir(type_args=(1,))
    assert "tla.func" in mlir
    assert "helper_requiring_transformation_kernel" in mlir
    assert "scf.if" in mlir


def test_dynamic_jit_helpers_do_not_reuse_an_id_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformed_helpers: list[object] = []
    transform = execution_lowering._transform_jit_helper

    def record_transform(helper: object) -> object:
        transformed_helpers.append(helper)
        return transform(helper)

    monkeypatch.setattr(
        execution_lowering, "id", lambda _helper: 0, raising=False
    )
    monkeypatch.setattr(execution_lowering, "_transform_jit_helper", record_transform)

    dynamic_jit_helpers_kernel.dump_mlir(type_args=(1,))

    assert len(transformed_helpers) == 2
    assert transformed_helpers[0] is not transformed_helpers[1]


def test_plain_helper_control_flow_is_not_transformed() -> None:
    with pytest.raises(Exception, match="tla.range is only iterable during lowering"):
        plain_helper_requiring_transformation_kernel.dump_mlir(type_args=(1,))


def test_helpers_with_static_python_if_lower_without_pr5_diagnostic() -> None:
    mlir = helpers_with_static_if_kernel.dump_mlir(type_args=(1,))
    assert "tla.make_coord" in mlir


def test_staging_local_nested_function_survives_root_transformation() -> None:
    mlir = staging_local_nested_function_kernel.dump_mlir(type_args=(1,))
    assert "scf.if" in mlir


def test_host_jit_is_python_orchestration_without_host_mlir() -> None:
    _HOST_EVENTS.clear()
    kernel = host_orchestration()
    assert _HOST_EVENTS == ["orchestrated"]
    assert kernel is host_launch_kernel
    # Phase-1 @tla.jit is a Python wrapper (no Host MLIR attachment).
    assert getattr(host_orchestration, "_tla_jit", False) is True
    assert not hasattr(host_orchestration, "_mlir")


@pytest.mark.parametrize(
    ("kernel", "region"),
    [
        (bare_helper_in_runtime_if_kernel, "scf.if"),
        (bare_helper_in_runtime_for_kernel, "scf.for"),
        (bare_helper_in_runtime_while_kernel, "scf.while"),
        (bare_helper_in_runtime_with_kernel, "tla.vector"),
    ],
)
def test_bare_helper_emits_from_runtime_region(kernel: object, region: str) -> None:
    mlir = kernel.dump_mlir(type_args=(2,))  # type: ignore[attr-defined]
    assert region in mlir
    assert "tla.make_coord" in mlir


@pytest.mark.parametrize(
    ("kernel", "region"),
    [
        (aliased_bare_helper_in_runtime_if_kernel, "scf.if"),
        (aliased_bare_helper_in_runtime_for_kernel, "scf.for"),
        (aliased_bare_helper_in_runtime_while_kernel, "scf.while"),
        (aliased_bare_helper_in_runtime_with_kernel, "tla.vector"),
    ],
)
def test_aliased_bare_helper_emits_from_runtime_region(
    kernel: object, region: str
) -> None:
    mlir = kernel.dump_mlir(type_args=(2,))  # type: ignore[attr-defined]
    assert region in mlir
    assert "tla.make_coord" in mlir


@pytest.mark.parametrize(
    "kernel",
    [
        nonlocal_bare_helper_in_runtime_if_kernel,
        global_bare_helper_in_runtime_if_kernel,
    ],
)
def test_runtime_bare_helper_cannot_rebind_enclosing_scope(kernel: object) -> None:
    with pytest.raises(SyntaxError, match="cannot declare global or nonlocal"):
        kernel.dump_mlir(type_args=(2,))  # type: ignore[attr-defined]


def test_jit_root_executes_bare_helper_during_staging() -> None:
    assert "tla.make_coord" in jit_calls_bare_helper.dump_mlir(type_args=(2,))


def test_bare_helper_without_inspectable_source_emits_during_staging() -> None:
    assert "tla.make_coord" in dynamic_source_helper_kernel.dump_mlir(type_args=(2,))
