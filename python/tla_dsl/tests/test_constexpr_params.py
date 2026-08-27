"""``tla.Constexpr[...]`` kernel parameters must deliver their host value.

Regression guard for a family of bugs where a parameter stopped being treated
as a constexpr and its value was silently dropped: the kernel body saw ``None``,
traced the wrong branch, and — because the compile cache key hashes the emitted
MLIR — every variant collapsed onto one cached kernel.

The later sections work outward from the binding helpers to the public
``tla.compile(...)`` entry point, ending with the launch payload it hands to the
device. No test here needs an NPU: the payload is packed with the real ABI
packer against fake operands.
"""

from __future__ import annotations

import dataclasses

import pytest

import catlass.tla as tla
from catlass import execution
from catlass.base_dsl.typing import is_constexpr_annotation
from catlass.dsl import (
    _bind_kernel_call_args,
    _get_typed_call_args,
)
from catlass.tla.runtime import make_fake_tensor


def _alloc(n: int) -> None:
    tla.allocate(n, tla.Float32, tla.AddressSpace.ub, 256)


# --------------------------------------------------------------------------
# The value reaches the kernel body
# --------------------------------------------------------------------------


@tla.kernel
def _str_selector(sel: tla.Constexpr[str]) -> None:
    _alloc(64 if sel == "wide" else 16)


@tla.kernel
def _int_selector(sel: tla.Constexpr[int]) -> None:
    _alloc(64 if sel == 1 else 16)


@tla.kernel
def _none_probe(sel: tla.Constexpr[str]) -> None:
    _alloc(64 if sel is None else 16)


def test_str_constexpr_selects_distinct_branches() -> None:
    wide = _str_selector.dump_mlir(type_args=("wide",))
    narrow = _str_selector.dump_mlir(type_args=("narrow",))
    assert wide != narrow, "Constexpr[str] did not reach the kernel body"


def test_int_constexpr_control() -> None:
    assert _int_selector.dump_mlir(type_args=(1,)) != _int_selector.dump_mlir(
        type_args=(0,)
    )


@pytest.mark.parametrize("value", ["wide", "narrow"])
def test_str_constexpr_is_distinguishable_from_none(value: str) -> None:
    assert _none_probe.dump_mlir(type_args=(value,)) != _none_probe.dump_mlir(
        type_args=(None,)
    )


def test_str_constexpr_survives_typed_call_args() -> None:
    def fn(out, sel: tla.Constexpr[str]) -> None: ...

    assert _get_typed_call_args((None, "wide"), fn) == (None, "wide")
    assert _get_typed_call_args((None, "narrow"), fn) == (None, "narrow")


def test_non_constexpr_args_are_still_erased() -> None:
    def fn(out, sel) -> None: ...

    assert _get_typed_call_args((None, object()), fn) is None


@tla.kernel
def _tuple_constexpr(shape: tla.Constexpr[tuple]) -> None:
    _alloc(shape[0] * shape[1])


def test_non_scalar_constexpr_payload() -> None:
    assert _tuple_constexpr.dump_mlir(type_args=((4, 8),)) != _tuple_constexpr.dump_mlir(
        type_args=((4, 16),)
    )


@tla.kernel
def _dtype_constexpr(dt: tla.Constexpr[type]) -> None:
    tla.allocate(64, dt, tla.AddressSpace.ub, 256)


def test_dtype_as_constexpr() -> None:
    # A dtype is a class whose unbound ``__get_mlir_types__`` is callable, so
    # type-resolving it as if it were a runtime argument crashes inside MLIR.
    assert _dtype_constexpr.dump_mlir(
        type_args=(tla.Float32,)
    ) != _dtype_constexpr.dump_mlir(type_args=(tla.Float16,))


@tla.kernel
def _unused_constexpr(sel: tla.Constexpr[str]) -> None:
    _alloc(32)


def test_constexpr_that_changes_nothing_shares_one_kernel() -> None:
    # The flip side of the bug: identical MLIR *should* reuse one kernel.
    assert _unused_constexpr.dump_mlir(
        type_args=("wide",)
    ) == _unused_constexpr.dump_mlir(type_args=("narrow",))


@dataclasses.dataclass
class _StrParams:
    mode: tla.Constexpr[str] = "wide"


@tla.kernel
def _dataclass_str_field(p: _StrParams) -> None:
    _alloc(64 if p.mode == "wide" else 16)


def test_str_constexpr_dataclass_field() -> None:
    assert _dataclass_str_field.dump_mlir(
        type_args=(_StrParams("wide"),)
    ) != _dataclass_str_field.dump_mlir(type_args=(_StrParams("narrow"),))


# --------------------------------------------------------------------------
# Annotation recognition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "form", ["Constexpr", "Constexpr[str]", "tla.Constexpr[str]", "x.y.Constexpr[int]"]
)
def test_qualified_constexpr_annotations_are_recognized(form: str) -> None:
    assert is_constexpr_annotation(form)


@pytest.mark.parametrize("form", ["MyConstexpr[str]", "int", "tla.Tensor"])
def test_unrelated_annotations_are_not_constexpr(form: str) -> None:
    assert not is_constexpr_annotation(form)


# --------------------------------------------------------------------------
# Call-site handling
# --------------------------------------------------------------------------


def test_omitted_constexpr_param_uses_its_default() -> None:
    def fn(a, mode: tla.Constexpr[str] = "wide") -> None: ...

    assert _get_typed_call_args((None,), fn) == (None, "wide")


def test_constexpr_passed_by_keyword_is_bound_positionally() -> None:
    def fn(a, mode: tla.Constexpr[str]) -> None: ...

    args, kwargs = _bind_kernel_call_args(fn, (None,), {"mode": "wide"})
    assert args == (None, "wide")
    assert kwargs == {}


def test_runtime_options_are_not_bound_as_kernel_args() -> None:
    def fn(a, mode: tla.Constexpr[str]) -> None: ...

    args, kwargs = _bind_kernel_call_args(
        fn, (None,), {"mode": "wide", "block_num": 4, "options": "--npu-arch 3510"}
    )
    assert args == (None, "wide")
    assert kwargs == {"block_num": 4, "options": "--npu-arch 3510"}


def test_keyword_constexpr_behind_a_defaulted_param() -> None:
    # The defaulted `b` must not end the scan: `sel` would then stay in the
    # runtime options and be silently replaced by its own default.
    def fn(a, b=1, sel: tla.Constexpr[str] = "narrow") -> None: ...

    args, kwargs = _bind_kernel_call_args(fn, (None,), {"sel": "wide"})
    assert args == (None, 1, "wide")
    assert kwargs == {}


def test_sparse_keyword_constexprs() -> None:
    def fn(a, x: tla.Constexpr[int] = 1, y: tla.Constexpr[str] = "narrow") -> None: ...

    args, kwargs = _bind_kernel_call_args(fn, (None,), {"y": "wide"})
    assert args == (None, 1, "wide")
    assert kwargs == {}


def test_unfillable_gap_stops_binding_rather_than_shifting() -> None:
    def fn(a, b, sel: tla.Constexpr[str]) -> None: ...

    args, kwargs = _bind_kernel_call_args(fn, (None,), {"sel": "wide"})
    assert args == (None,), "a gap with no default must not pull `sel` into `b`"
    assert kwargs == {"sel": "wide"}


def test_variadic_signature_is_left_untouched() -> None:
    # No fixed parameter list to bind positional host values against.
    def fn(a, *rest, sel: tla.Constexpr[str] = "narrow") -> None: ...

    args, kwargs = _bind_kernel_call_args(fn, (None,), {"sel": "wide"})
    assert args == (None,)
    assert kwargs == {"sel": "wide"}


def test_param_colliding_with_a_runtime_option_is_reported() -> None:
    def fn(options: tla.Constexpr[str]) -> None: ...

    with pytest.raises(TypeError, match="collides with the runtime option"):
        _bind_kernel_call_args(fn, (), {"options": "wide"})


@tla.kernel
def _keyword_only_constexpr(*, sel: tla.Constexpr[str]) -> None:
    _alloc(64 if sel == "wide" else 16)


def test_keyword_only_constexpr_param() -> None:
    # Keyword-only params are ordinary entries in `arg_names`; the kernel body
    # must not be called with them positionally.
    assert _keyword_only_constexpr.dump_mlir(
        type_args=("wide",)
    ) != _keyword_only_constexpr.dump_mlir(type_args=("narrow",))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


@tla.kernel
def _unmarked_str_param(sel: str) -> None:
    _alloc(64 if sel == "wide" else 16)


def test_unmarked_compile_time_param_names_itself() -> None:
    with pytest.raises(Exception) as excinfo:
        _unmarked_str_param.dump_mlir(type_args=("wide",))
    message = str(excinfo.value)
    assert "'sel'" in message
    assert "Constexpr" in message


# --------------------------------------------------------------------------
# Public entry points
#
# ``tla.compile(...)`` takes kernel arguments and runtime options in one
# ``**kwargs`` and must keep constexpr values out of the launch ABI. The tests
# above pin the binding helpers directly; these exercise the public API.
# --------------------------------------------------------------------------

VECTOR_ELE = 64
VL_ELE = 64
_OPTIONS = "--npu-arch 3510"


@tla.kernel
def _selectable_binary(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    mode: tla.Constexpr[str],
) -> None:
    """``gm_c = gm_a + gm_b`` or ``gm_a - gm_b``, chosen at compile time."""
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ub_ptr_a = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)

    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ub_ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a)
        tla.copy(ub_b, gm_b)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                shape = tla.make_shape(VL_ELE)
                coord = tla.make_coord(i)
                reg_a = tla.tile_view(ub_a, shape, coord).load()
                reg_b = tla.tile_view(ub_b, shape, coord).load()
                if tla.const_expr(mode == "add"):
                    reg_c = tla.add(reg_a, reg_b)
                else:
                    reg_c = tla.sub(reg_a, reg_b)
                tla.tile_view(ub_c, shape, coord).store(reg_c)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


def _fake_operands():
    return tuple(
        make_fake_tensor(
            tla.Float32,
            (VECTOR_ELE,),
            (1,),
            origin_shape=(VECTOR_ELE,),
            coord=(0,),
            layout_tag=tla.arch.RowMajor,
        )
        for _ in range(3)
    )


def _bound_operands():
    """Fake operands stamped with a data pointer, so the ABI packer can run."""
    tensors = _fake_operands()
    for index, tensor in enumerate(tensors, start=1):
        tensor.data_ptr = index * 0x1000
        tensor._external_binding = True
    return tensors


def test_compile_separates_named_constexpr_from_runtime_options() -> None:
    a, b, c = _fake_operands()
    add = tla.compile(_selectable_binary, a, b, c, mode="add", options=_OPTIONS)
    sub = tla.compile(_selectable_binary, a, b, c, mode="sub", options=_OPTIONS)

    # A distinct kernel per mode proves `mode=` was bound as the kernel argument
    # and not swallowed as an unknown runtime option, while `options=` still
    # reached the runtime (a bogus option string would have raised).
    assert add.cache_key != sub.cache_key


def test_compiled_artifact_abi_excludes_the_constexpr() -> None:
    a, b, c = _fake_operands()
    compiled = tla.compile(
        _selectable_binary, a, b, c, mode="add", options=_OPTIONS
    )

    # Three tensors in, one compile-time constant that must own no ABI slot.
    assert execution._logical_launch_arg_count(
        compiled.execution_args.kernel_abi
    ) == 3


def _pack_payload(compiled, launch_args):
    """Pack ``launch_args`` through the compiled function's ABI binder."""
    return compiled.execution_args.generate_launch_payload(launch_args)


def test_compiled_artifact_payload_matches_the_tensors_alone() -> None:
    a, b, c = _bound_operands()
    compiled = tla.compile(
        _selectable_binary, a, b, c, mode="add", options=_OPTIONS
    )
    kernel_abi = compiled.execution_args.kernel_abi
    assert kernel_abi is not None

    # tla.compile takes the constexpr; the launch that follows must not.
    assert len(_pack_payload(compiled, (a, b, c))) == kernel_abi.total_size
