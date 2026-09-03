"""Phase-1: ``tla.Constexpr`` Callable kernel arguments (staging inline)."""

from __future__ import annotations

from typing import Callable

import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.base_dsl.jit_executor import ExecutionArgs
from catlass.base_dsl.runtime.jit_arg_adapters import is_arg_annotation_constexpr
from catlass.dsl import _get_typed_call_args, _strip_constexpr_launch_args
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


def _ub_tensor(dtype: type[tla.Numeric] = tla.Float32, size: int = 64) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (size,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(size,),
        layout_tag=tla.arch.RowMajor,
    )


def _abs_epilogue(value):
    return tla.abs(value)


def _neg_epilogue(value):
    return tla.neg(value)


@tla.kernel
def _constexpr_callable_kernel(src: tla.Tensor, epilogue: tla.Constexpr) -> None:
    tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = epilogue(tile.load())


@tla.kernel
def _constexpr_callable_in_range_kernel(
    src: tla.Tensor, epilogue: tla.Constexpr
) -> None:
    n_ele = src.origin_shape[0]
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + 63) // 64):
                tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i))
                _ = epilogue(tile.load())


@tla.kernel
def _bad_runtime_callable_kernel(src: tla.Tensor, epilogue) -> None:
    tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = epilogue(tile.load())


def test_is_arg_annotation_constexpr_matches_tla_marker() -> None:
    assert is_arg_annotation_constexpr(tla.Constexpr, "epilogue", 1, None)
    assert is_arg_annotation_constexpr(tla.Constexpr[int], "limit", 0, None)
    assert is_arg_annotation_constexpr("tla.Constexpr", "epilogue", 1, None)
    assert is_arg_annotation_constexpr("tla.Constexpr[int]", "limit", 0, None)
    assert not is_arg_annotation_constexpr(tla.Tensor, "src", 0, None)
    assert not is_arg_annotation_constexpr("TensorAlias", "src", 0, None)


def test_get_typed_call_args_preserves_constexpr_callable() -> None:
    src = _ub_tensor()
    typed = _get_typed_call_args((src, _abs_epilogue), _constexpr_callable_kernel.fn)
    assert typed is not None
    assert typed[0] is src
    assert typed[1] is _abs_epilogue


def test_get_rectified_args_from_original_args_drops_callable() -> None:
    src = _ub_tensor()
    stripped = ExecutionArgs(
        original_signature=BaseDSL()._get_signature(_constexpr_callable_kernel.fn)
    ).get_rectified_args_from_original_args((src, _abs_epilogue))
    assert stripped == (src,)


def test_execution_args_from_callable_strips_constexpr() -> None:
    src = _ub_tensor()
    stripped = ExecutionArgs.from_callable(
        _constexpr_callable_kernel.fn
    ).get_rectified_args_from_original_args((src, _abs_epilogue))
    assert stripped == (src,)


def test_strip_constexpr_launch_args_keeps_runtime_arity() -> None:
    src = _ub_tensor()
    stripped = _strip_constexpr_launch_args((src,), _constexpr_callable_kernel.fn)
    assert stripped == (src,)


def test_constexpr_callable_inlines_into_device_ir() -> None:
    src = _ub_tensor()
    mlir = _constexpr_callable_kernel.dump_mlir(type_args=(src, _abs_epilogue))
    assert "_constexpr_callable_kernel" in mlir
    assert "%arg1" not in mlir
    assert "tla.func @_constexpr_callable_kernel(%arg0:" in mlir
    assert "abs" in mlir.lower()


def test_constexpr_callable_inside_tla_range_inlines() -> None:
    # e2e kernels call Constexpr callables inside tla.range; that capture must
    # not be rejected as a host-object method side effect.
    src = _ub_tensor()
    mlir = _constexpr_callable_in_range_kernel.dump_mlir(
        type_args=(src, _abs_epilogue)
    )
    assert "tla.func @_constexpr_callable_in_range_kernel(%arg0:" in mlir
    assert "abs" in mlir.lower()
    assert "scf.for" in mlir


def test_different_constexpr_callables_specialize_ir() -> None:
    src = _ub_tensor()
    mlir_abs = _constexpr_callable_kernel.dump_mlir(type_args=(src, _abs_epilogue))
    mlir_neg = _constexpr_callable_kernel.dump_mlir(type_args=(src, _neg_epilogue))
    assert mlir_abs != mlir_neg


def test_lambda_constexpr_callable_inlines() -> None:
    src = _ub_tensor()
    mlir = _constexpr_callable_kernel.dump_mlir(
        type_args=(src, lambda value: tla.exp(value))
    )
    assert "%arg1" not in mlir
    assert "exp" in mlir.lower()


def test_runtime_callable_argument_is_rejected() -> None:
    src = _ub_tensor()
    with pytest.raises(TlaLoweringError, match="has no runtime type"):
        _bad_runtime_callable_kernel.dump_mlir(type_args=(src, _abs_epilogue))


def test_filter_runtime_signature_drops_top_level_constexpr() -> None:
    @tla.kernel
    def kernel(limit: tla.Constexpr[int], value: int) -> None:
        _ = tla.make_coord(limit, value)

    original = BaseDSL()._get_signature(kernel.fn)
    binder = ExecutionArgs(original_signature=original)
    assert binder.original_signature is original
    assert list(binder.original_signature.parameters) == ["limit", "value"]
    assert list(binder.signature.parameters) == ["value"]
    assert list(binder.filter_runtime_signature(original).parameters) == ["value"]
    assert binder.get_rectified_args_from_original_args((4, 1)) == (1,)


@tla.kernel
def _ordinary_constexpr_callable(
    mask_mod: tla.Constexpr[Callable[[int], bool]],
) -> None:
    del mask_mod


def test_ordinary_callable_constexpr_has_no_abi_slot() -> None:
    def ordinary_mask(value: int) -> bool:
        return value >= 0

    mlir = _ordinary_constexpr_callable.dump_mlir(type_args=(ordinary_mask,))
    assert "tla.func" in mlir
    assert "mask_mod" not in mlir


def test_compile_preserves_callable_constexpr_when_an_annotation_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Local type aliases are absent from kernel.fn.__globals__, so
    # BaseDSL._get_signature(eval_str=True) fails and falls back to string
    # annotations. Constexpr detection must still recognize "tla.Constexpr".
    TensorAlias = tla.Tensor

    @tla.kernel
    def kernel(
        src: TensorAlias,
        epilogue: tla.Constexpr,
    ) -> None:
        del src, epilogue

    src = _ub_tensor()
    captured: dict[str, object] = {}
    compiled_sentinel = object()

    def fake_compile(*, type_args=None, **kwargs):
        captured["type_args"] = type_args
        return compiled_sentinel

    # Only exercise public tla.compile argument handling; skip native compile.
    monkeypatch.setattr(kernel, "compile", fake_compile)

    compiled = tla.compile(kernel, src, _abs_epilogue)

    assert compiled is compiled_sentinel
    assert captured["type_args"] == (src, _abs_epilogue)


class _UserCallableEpilogue:
    def __call__(self, value):
        return tla.neg(value)


def test_user_defined_callable_class_is_rejected() -> None:
    src = _ub_tensor()
    epilogue = _UserCallableEpilogue()

    with pytest.raises(
        TypeError,
        match=r"user-defined class '_UserCallableEpilogue' is not supported",
    ):
        _constexpr_callable_kernel.dump_mlir(
            type_args=(src, epilogue),
        )
