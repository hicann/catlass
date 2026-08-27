"""Public compile/launch argument-model tests."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("catlass", exc_type=ImportError)

import catlass.tla as tla


@tla.kernel
def _scalar_kernel(value: tla.Int32) -> None:
    del value


@tla.kernel
def _distribution_kernel(
    value: tla.Int32,
    distribution: tla.Constexpr[str],
) -> None:
    del value, distribution


def test_public_compile_uses_positional_sample_arguments(monkeypatch) -> None:
    compiled_function = object()
    recorded: dict[str, Any] = {}

    def fake_compile(self, *, type_args=None, **kwargs):
        recorded["self"] = self
        recorded["type_args"] = type_args
        recorded["kwargs"] = kwargs
        return compiled_function

    monkeypatch.setattr(type(_distribution_kernel), "compile", fake_compile)
    sample = tla.Int32(7)

    compiled = tla.compile(_distribution_kernel, sample, "row_major")

    assert compiled is compiled_function
    assert recorded == {
        "self": _distribution_kernel,
        "type_args": (sample, "row_major"),
        "kwargs": {},
    }


def test_public_compile_rejects_type_args_keyword() -> None:
    with pytest.raises(TypeError, match="sample arguments positionally"):
        tla.compile(_scalar_kernel, type_args=(tla.Int32(7),))


def test_direct_kernel_invocation_is_disabled() -> None:
    sample = tla.Int32(9)
    with pytest.raises(TypeError, match="Direct @tla.kernel invocation is disabled"):
        _scalar_kernel(sample)
