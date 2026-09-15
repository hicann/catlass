"""Smoke tests for JitArgAdapterRegistry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pytest

pytest.importorskip("catlass", exc_type=ImportError)

from catlass.base_dsl.jit_executor import ExecutionArgs
from catlass.base_dsl.runtime.jit_arg_adapters import (
    JitArgAdapterRegistry,
    _PointerLaunchArg,
)
import catlass.tla as tla
from catlass import execution
from catlass._mlir import ir as mlir_ir
from catlass.tla.runtime import make_fake_tensor


class _HasDataPtr:
    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


class _Custom:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


def test_duck_typed_data_ptr_is_adapted() -> None:
    adapted = ExecutionArgs().get_rectified_args([_HasDataPtr(0xABC)])
    assert isinstance(adapted[0], _PointerLaunchArg)
    assert adapted[0].__c_pointers__() == [0xABC]


def test_registered_adapter_is_used() -> None:
    JitArgAdapterRegistry.clear()

    @JitArgAdapterRegistry.register_jit_arg_adapter(_Custom)
    def _adapt_custom(obj: _Custom) -> _PointerLaunchArg:
        return _PointerLaunchArg(obj.ptr)

    try:
        adapted = ExecutionArgs().get_rectified_args([_Custom(0x101)])
        assert isinstance(adapted[0], _PointerLaunchArg)
        assert adapted[0].__c_pointers__() == [0x101]
        assert JitArgAdapterRegistry.get_registered_adapter(_Custom(0)) is _adapt_custom
    finally:
        JitArgAdapterRegistry.clear()


class _LaunchValue:
    def __init__(self, value):
        self.value = value


@dataclass
class _Config:
    source: tla.Tensor
    scale: tla.Float32


class _NamedConfig(NamedTuple):
    source: tla.Tensor
    scale: tla.Float32


@tla.kernel
def _flat_launch_kernel(source, scale, output):
    output[0] = source[0] + scale


@tla.kernel
def _structured_launch_kernel(aux, output, access: tla.Constexpr):
    source, scale = access(aux)
    output[0] = source[0] + scale


@pytest.mark.parametrize("container", ["flat", "nested", "dataclass", "namedtuple"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_object_applies_existing_launch_adapters(
    monkeypatch, isolated_compile_cache, container, dynamic
):
    from catlass.base_dsl.runtime.argument_tree import ArgumentTreeError

    monkeypatch.setattr(JitArgAdapterRegistry, "jit_arg_adapter_registry", {})
    monkeypatch.setenv("CATLASS_DSL_CACHE", "1")
    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    source, output = (
        make_fake_tensor(tla.Float32, (8,), (1,), layout_tag=tla.arch.RowMajor)
        for _ in range(2)
    )
    if dynamic:
        source.mark_compact_shape_dynamic(0)
    # Only host payloads are inspected; these addresses are never launched.
    for index, tensor in enumerate((source, output), 1):
        tensor.data_ptr = index * 0x1000
        tensor._external_binding = True

    def values(src, scale):
        if container == "flat":
            return src, scale, output
        if container == "nested":
            return ([src, scale], None), output
        config = _Config if container == "dataclass" else _NamedConfig
        return config(src, scale), output

    def access(aux):
        if container == "nested":
            return aux[0]
        return aux.source, aux.scale

    kernel = _flat_launch_kernel if container == "flat" else _structured_launch_kernel
    samples = values(source, tla.Float32(2))
    compile_args = samples if container == "flat" else (*samples, access)
    compiled = tla.compile(kernel, *compile_args, options="--npu-arch 3510")
    calls = []

    @JitArgAdapterRegistry.register_jit_arg_adapter(_LaunchValue)
    def adapt_value(value):
        calls.append(value)
        return value.value

    cached = tla.compile(kernel, *compile_args, options="--npu-arch 3510")
    assert compiled.cache_key == cached.cache_key
    assert not calls
    for artifact, pointer, scale in (
        (compiled, 0x3000, tla.Float32(3)),
        (cached, 0x4000, tla.Float32(7)),
    ):
        source.data_ptr = pointer
        wrapped = _LaunchValue(source), _LaunchValue(scale)
        calls.clear()
        payload = artifact.execution_args.generate_launch_payload(values(*wrapped))
        assert calls == list(wrapped)
        assert payload == execution._pack_launch_args(
            (source, scale, output), artifact.execution_args.kernel_abi
        )

    with pytest.raises(execution.TlaUnsupportedAbiError, match="f32"):
        compiled.execution_args.generate_launch_payload(
            values(source, _LaunchValue(tla.Int32(7)))
        )
    if container == "nested":
        with pytest.raises(ArgumentTreeError, match="expected None"):
            compiled.execution_args.generate_launch_payload(
                (([source, tla.Float32(3)], 1), output)
            )
    JitArgAdapterRegistry.clear()
    with pytest.raises(execution.TlaUnsupportedAbiError, match="f32"):
        compiled.execution_args.generate_launch_payload(
            values(source, _LaunchValue(tla.Float32(3)))
        )


def test_launch_adapters_preserve_native_provider_precedence(monkeypatch):
    from catlass.base_dsl.runtime.argument_tree import build_runtime_argument_tree

    def unexpected(value):
        pytest.fail("native launch providers must not be overridden by adapters")

    monkeypatch.setattr(
        JitArgAdapterRegistry,
        "jit_arg_adapter_registry",
        {
            tla.Float32: unexpected,
            _PointerLaunchArg: unexpected,
            int: lambda value: tla.Int32(value + 1),
        },
    )
    numeric = tla.Float32(2)
    pointer = _PointerLaunchArg(0x1000)
    tensor = make_fake_tensor(tla.Float32, (8,), (1,))
    with mlir_ir.Context() as context:
        binder = ExecutionArgs(
            argument_trees=tuple(
                build_runtime_argument_tree(arg, context)
                for arg in (numeric, tensor, 1)
            )
        )
    normalized = binder.get_rectified_args((numeric, pointer, 1))
    assert normalized[0] is numeric
    assert normalized[1] is pointer
    assert isinstance(normalized[2], tla.Int32)
    assert normalized[2].value == 2
