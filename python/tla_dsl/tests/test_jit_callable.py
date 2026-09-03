"""``@tla.jit`` helper identity (wraps / ``__wrapped__``) and Constexpr staging."""

from __future__ import annotations

import inspect
from typing import Callable

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


@tla.jit
def _causal_mask(q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx


@tla.kernel
def _jit_helper_kernel(
    mask_mod: tla.Constexpr[Callable[[int, int], bool]],
    value: int,
) -> None:
    mask_mod(value, 0)


def test_jit_callable_uses_wrapped_identity() -> None:
    assert inspect.isfunction(_causal_mask)
    assert not isinstance(_causal_mask, tla.TlaJitFunction)
    assert hasattr(_causal_mask, "__wrapped__")
    assert inspect.unwrap(_causal_mask) is _causal_mask.__wrapped__
    assert tla.is_jit_callable(_causal_mask)
    assert tla.unwrap_jit_callable(_causal_mask) is _causal_mask.__wrapped__


def test_jit_helper_staged_without_abi_slot() -> None:
    mlir = _jit_helper_kernel.dump_mlir(type_args=(_causal_mask, 7))
    assert "tla.func" in mlir
    assert "mask_mod" not in mlir


@tla.kernel
def _unannotated_jit_helper_kernel(src: tla.Tensor, mask_mod) -> None:
    del src, mask_mod


def test_unannotated_jit_helper_rejected_via_generic_path() -> None:
    src = make_fake_tensor(
        tla.Float32,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(TlaLoweringError, match="has no runtime type"):
        _unannotated_jit_helper_kernel.dump_mlir(type_args=(src, _causal_mask))
