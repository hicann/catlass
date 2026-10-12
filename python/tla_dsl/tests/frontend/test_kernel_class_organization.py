"""Class-organized kernels: compile-time configuration carried by an instance.

These cases are about how ``tla.compile`` picks the entry and where the instance
lands; the kernel bodies are arbitrary and carry no computation of their own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("catlass", exc_type=ImportError)

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.execution import TlaUnsupportedAbiError
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


def _op(value: int) -> None:
    tla.make_shape(value)


class _ForwardingKernel:
    """The issue's shape: a ``@tla.jit __call__`` naming the kernel to compile."""

    def __init__(self, vector_ele: int) -> None:
        self.vector_ele = vector_ele

    @tla.jit
    def __call__(self, lhs, rhs, output) -> None:
        self.kernel(lhs, rhs, output)

    @tla.kernel
    def kernel(self, lhs, rhs, output) -> None:
        _op(self.vector_ele)


class _NoJitCallKernel:
    """A kernel method but no ``@tla.jit __call__``: nothing names the entry."""

    @tla.kernel
    def kernel(self, lhs, rhs) -> None:
        _op(8)


class _UndecoratedCallKernel:
    """A plain ``__call__`` does not name the entry either."""

    def __call__(self, lhs, rhs) -> None:
        pass

    @tla.kernel
    def kernel(self, lhs, rhs) -> None:
        _op(8)


class _RunKernel:
    """The forward, not the method name, picks the kernel to compile."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        self.run(lhs, rhs)

    @tla.kernel
    def run(self, lhs, rhs) -> None:
        _op(8)


class _NoKernelClass:
    """An instance whose class declares no kernel at all."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        pass


class _ReorderedCallKernel:
    """``__call__`` forwards its parameters out of order."""

    @tla.jit
    def __call__(self, lhs, rhs, output) -> None:
        self.kernel(rhs, lhs, output)

    @tla.kernel
    def kernel(self, lhs, rhs, output) -> None:
        _op(8)


class _ShortCallKernel:
    """``__call__`` forwards fewer parameters than the kernel takes."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        self.kernel(lhs, rhs)

    @tla.kernel
    def kernel(self, lhs, rhs, output) -> None:
        _op(8)


class _ForeignCallKernel:
    """``__call__`` forwards to something that is not a kernel."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        self.transform(lhs, rhs)

    @tla.kernel
    def kernel(self, lhs, rhs) -> None:
        _op(8)

    def transform(self, lhs, rhs) -> None:
        _op(8)


class _NoReceiverKernel:
    """A kernel reached as a class attribute: no leading ``self``, no receiver."""

    @tla.kernel
    def kernel(lhs, rhs) -> None:
        _op(8)


class _MultiStatementCallKernel:
    """``__call__`` holds more than the single forwarding statement."""

    @tla.jit
    def __call__(self, lhs, rhs, output) -> None:
        self.count = 1
        self.kernel(lhs, rhs, output)

    @tla.kernel
    def kernel(self, lhs, rhs, output) -> None:
        _op(8)


class _WrappedCallKernel:
    """The single forward wraps for width; it is still one statement."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        self.kernel(
            lhs,
            rhs,
        )

    @tla.kernel
    def kernel(self, lhs, rhs) -> None:
        _op(8)


class _KeywordCallKernel:
    """``__call__`` forwards with a keyword argument."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        self.kernel(lhs, output=rhs)

    @tla.kernel
    def kernel(self, lhs, output) -> None:
        _op(8)


class _ComputedArgumentCallKernel:
    """``__call__`` forwards a computed argument, not its own parameter."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        self.kernel(lhs, rhs[0])

    @tla.kernel
    def kernel(self, lhs, rhs) -> None:
        _op(8)


class _BareNameCallKernel:
    """``__call__`` calls a bare name instead of ``self.<kernel>``."""

    @tla.jit
    def __call__(self, lhs, rhs) -> None:
        kernel(lhs, rhs)

    @tla.kernel
    def kernel(self, lhs, rhs) -> None:
        _op(8)


class _NoReceiverCallKernel:
    """``__call__`` without a leading receiver."""

    @tla.jit
    def __call__(lhs, rhs) -> None:
        pass


@tla.kernel
def _plain_kernel_with_leading_self(self, lhs, output) -> None:
    _op(self.vector_ele)


class _Tiling:
    """A plain configuration object, not a dataclass."""

    def __init__(self, vector_ele: int) -> None:
        self.vector_ele = vector_ele


class _NestedConfigKernel:
    """Configuration reached through a plain (non-dataclass) attribute."""

    def __init__(self, tiling: _Tiling) -> None:
        self.tiling = tiling

    @tla.kernel
    def kernel(self, lhs, rhs, output) -> None:
        _op(self.tiling.vector_ele)


def _fake_tensors(count: int) -> tuple[tla.Tensor, ...]:
    return tuple(
        make_fake_tensor(
            tla.Float16,
            (1, 8),
            (8, 1),
            origin_shape=(1, 8),
            layout_tag=tla.arch.RowMajor,
        )
        for _ in range(count)
    )


def _lower(kernel: Any, type_args: tuple[Any, ...]) -> Any:
    return BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=type_args,
        location=kernel.decorator_location,
    )


def test_instance_compiles_the_kernel_its_jit_call_forwards_to(monkeypatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_compile(self, *, type_args=None, **kwargs):
        recorded["kernel"] = self
        recorded["type_args"] = type_args
        recorded["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(type(_ForwardingKernel.kernel), "compile", fake_compile)
    instance = _ForwardingKernel(vector_ele=7)
    lhs, rhs, output = _fake_tensors(3)

    tla.compile(instance, lhs, rhs, output)

    assert recorded["kernel"] is _ForwardingKernel.kernel
    assert recorded["type_args"] == (instance, lhs, rhs, output)
    assert recorded["kwargs"] == {}


def test_bound_kernel_method_carries_its_instance(monkeypatch) -> None:
    """``instance.kernel`` is bound like a plain method: the instance rides along."""

    recorded: dict[str, Any] = {}

    def fake_compile(self, *, type_args=None, **kwargs):
        recorded["kernel"] = self
        recorded["type_args"] = type_args
        return object()

    monkeypatch.setattr(type(_ForwardingKernel.kernel), "compile", fake_compile)
    instance = _ForwardingKernel(vector_ele=7)
    lhs, rhs, output = _fake_tensors(3)

    tla.compile(instance.kernel, lhs, rhs, output)

    assert recorded["kernel"].fn is _ForwardingKernel.kernel.fn
    assert recorded["type_args"] == (instance, lhs, rhs, output)


def test_kernel_read_off_the_class_takes_no_receiver() -> None:
    assert _ForwardingKernel.kernel.receiver is None
    assert _ForwardingKernel(7).kernel.receiver is not None


def test_bound_method_rejects_a_duplicated_receiver() -> None:
    """A bound method already carries the instance; passing it again is diagnosed."""

    lhs, rhs, output = _fake_tensors(3)
    instance = _ForwardingKernel(7)

    with pytest.raises(
        TlaUnsupportedAbiError, match="already rides along as the receiver"
    ):
        tla.compile(instance.kernel, instance, lhs, rhs, output)


def test_instance_without_jit_call_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(TlaUnsupportedAbiError, match="declares no @tla.jit __call__"):
        tla.compile(_NoJitCallKernel(), lhs, rhs)


def test_plain_call_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(TlaUnsupportedAbiError, match="declares no @tla.jit __call__"):
        tla.compile(_UndecoratedCallKernel(), lhs, rhs)


def test_jit_call_names_the_kernel_to_compile(monkeypatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_compile(self, *, type_args=None, **kwargs):
        recorded["kernel"] = self
        return object()

    monkeypatch.setattr(type(_RunKernel.run), "compile", fake_compile)
    lhs, rhs = _fake_tensors(2)

    tla.compile(_RunKernel(), lhs, rhs)

    assert recorded["kernel"] is _RunKernel.run


def test_class_without_a_kernel_method_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(
        TlaUnsupportedAbiError, match="has a single statement, but it is not a call"
    ):
        tla.compile(_NoKernelClass(), lhs, rhs)


def test_undecorated_function_keeps_the_entry_point_hint() -> None:
    def not_a_kernel(lhs, rhs) -> None:
        pass

    with pytest.raises(TlaUnsupportedAbiError, match="expects a @tla.kernel function"):
        tla.compile(not_a_kernel, *_fake_tensors(2))


def test_jit_call_that_reorders_parameters_is_rejected() -> None:
    lhs, rhs, output = _fake_tensors(3)

    with pytest.raises(
        TlaUnsupportedAbiError, match="instead of its own parameters"
    ):
        tla.compile(_ReorderedCallKernel(), lhs, rhs, output)


def test_jit_call_with_several_statements_is_rejected() -> None:
    lhs, rhs, output = _fake_tensors(3)

    with pytest.raises(
        TlaUnsupportedAbiError, match="has 2 statements instead of one"
    ):
        tla.compile(_MultiStatementCallKernel(), lhs, rhs, output)


def test_jit_call_forward_wrapped_for_width_is_one_statement(monkeypatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_compile(self, *, type_args=None, **kwargs):
        recorded["kernel"] = self
        return object()

    monkeypatch.setattr(type(_WrappedCallKernel.kernel), "compile", fake_compile)
    lhs, rhs = _fake_tensors(2)

    tla.compile(_WrappedCallKernel(), lhs, rhs)

    assert recorded["kernel"] is _WrappedCallKernel.kernel


def test_jit_call_with_keyword_arguments_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(TlaUnsupportedAbiError, match="passes keyword arguments"):
        tla.compile(_KeywordCallKernel(), lhs, rhs)


def test_jit_call_that_passes_a_computed_argument_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(
        TlaUnsupportedAbiError, match=r"passes rhs\[0\] instead of its own parameter"
    ):
        tla.compile(_ComputedArgumentCallKernel(), lhs, rhs)


def test_jit_call_that_does_not_call_self_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(
        TlaUnsupportedAbiError, match=r"calls kernel, not self\.<kernel>"
    ):
        tla.compile(_BareNameCallKernel(), lhs, rhs)


def test_jit_call_without_a_receiver_is_rejected() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(TlaUnsupportedAbiError, match="does not take a receiver first"):
        tla.compile(_NoReceiverCallKernel(), lhs, rhs)


def test_jit_call_must_forward_every_kernel_parameter() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(TlaUnsupportedAbiError, match="forwards 2 parameters to 'kernel'"):
        tla.compile(_ShortCallKernel(), lhs, rhs)


def test_jit_call_must_forward_to_a_kernel_method() -> None:
    lhs, rhs = _fake_tensors(2)

    with pytest.raises(
        TlaUnsupportedAbiError, match="not one of that class's @tla.kernel"
    ):
        tla.compile(_ForeignCallKernel(), lhs, rhs)


def test_receiver_keeps_no_runtime_abi_slot() -> None:
    instance = _ForwardingKernel(vector_ele=7)
    lhs, rhs, output = _fake_tensors(3)

    lowered = _lower(_ForwardingKernel.kernel, (instance, lhs, rhs, output))

    assert len(lowered.argument_trees) == 3


def test_self_attributes_fold_into_the_kernel_as_constants() -> None:
    lhs, rhs, output = _fake_tensors(3)

    seven = _lower(
        _ForwardingKernel.kernel, (_ForwardingKernel(7), lhs, rhs, output)
    ).asm(generic=True)
    nine = _lower(
        _ForwardingKernel.kernel, (_ForwardingKernel(9), lhs, rhs, output)
    ).asm(generic=True)

    assert "!tla.shape<7>" in seven
    assert "!tla.shape<9>" in nine


def test_configuration_through_a_plain_attribute_folds_to_a_constant() -> None:
    lhs, rhs, output = _fake_tensors(3)

    lowered = _lower(
        _NestedConfigKernel.kernel,
        (_NestedConfigKernel(_Tiling(7)), lhs, rhs, output),
    )

    assert len(lowered.argument_trees) == 3
    assert "!tla.shape<7>" in lowered.asm(generic=True)


def test_leading_self_is_a_receiver_outside_a_class_too() -> None:
    """The receiver rule keys off position and name, not on the owning class.

    ``is_arg_annotation_constexpr`` reserves a leading ``self`` / ``cls`` without
    asking whether the function belongs to a class, so a plain kernel written that
    way carries its first argument as compile-time configuration as well.
    """

    lhs, output = _fake_tensors(2)

    lowered = _lower(_plain_kernel_with_leading_self, (_Tiling(7), lhs, output))

    assert len(lowered.argument_trees) == 2
    assert "!tla.shape<7>" in lowered.asm(generic=True)


def test_kernel_without_a_leading_self_has_no_receiver() -> None:
    lhs, rhs = _fake_tensors(2)

    lowered = _lower(_NoReceiverKernel.kernel, (lhs, rhs))

    assert len(lowered.argument_trees) == 2


@pytest.mark.parametrize("receiver", ["tensor", "numeric"])
def test_runtime_value_passed_as_the_receiver_is_rejected(receiver: str) -> None:
    """A receiver is configuration: runtime data there would leave the ABI silently."""

    lhs, rhs, output = _fake_tensors(3)
    value = lhs if receiver == "tensor" else tla.Int32(5)

    with pytest.raises(
        TlaLoweringError, match="is the compile-time receiver, but a runtime value"
    ):
        _ = _lower(_plain_kernel_with_leading_self, (value, rhs, output))


def test_bound_kernel_method_compiles_the_same_binary(
    isolated_compile_cache,
) -> None:
    """One configuration, one binary: ``instance.kernel`` matches ``instance``."""

    lhs, rhs, output = _fake_tensors(3)

    from_instance = tla.compile(
        _ForwardingKernel(7), lhs, rhs, output, options="--npu-arch 3510"
    )
    from_bound = tla.compile(
        _ForwardingKernel(7).kernel, lhs, rhs, output, options="--npu-arch 3510"
    )

    assert from_bound.cache_key == from_instance.cache_key
    assert list(from_bound.execution_args.signature.parameters) == [
        "lhs",
        "rhs",
        "output",
    ]


def test_issue_shape_compiles_one_binary_per_configuration(
    isolated_compile_cache,
) -> None:
    lhs, rhs, output = _fake_tensors(3)

    seven = tla.compile(
        _ForwardingKernel(7), lhs, rhs, output, options="--npu-arch 3510"
    )
    nine = tla.compile(
        _ForwardingKernel(9), lhs, rhs, output, options="--npu-arch 3510"
    )

    assert Path(seven.kernel_binary_path).is_file()
    assert list(seven.execution_args.original_signature.parameters) == [
        "self",
        "lhs",
        "rhs",
        "output",
    ]
    assert list(seven.execution_args.signature.parameters) == [
        "lhs",
        "rhs",
        "output",
    ]
    assert seven.cache_key != nine.cache_key
