"""Smoke tests for JitArgAdapterRegistry."""

from __future__ import annotations

import pytest

pytest.importorskip("catlass", exc_type=ImportError)

from catlass.base_dsl.jit_executor import ExecutionArgs
from catlass.base_dsl.runtime.jit_arg_adapters import (
    JitArgAdapterRegistry,
    _PointerLaunchArg,
)


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
