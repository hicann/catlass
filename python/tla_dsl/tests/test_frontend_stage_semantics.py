"""Regression coverage for frontend validation and stage tracking."""

import pytest

import catlass.tla as tla


CALLS: list[str] = []


def _plain_helper(value: int) -> int:
    CALLS.append("plain_helper")
    return value


class _UserValue:
    def read(self) -> int:
        CALLS.append("user_method")
        return 1


_USER_VALUES = (_UserValue(),)
_USER_MAP = {"item": _UserValue()}


@tla.kernel
def _alias_branch_bypass(value: int) -> None:
    helper = _plain_helper
    if False:
        helper = tla.make_coord
    tla.make_coord(helper(value), 0)


@tla.kernel
def _container_method_bypass(value: int) -> None:
    del value
    tla.make_coord(_USER_VALUES[0].read(), 0)


@tla.kernel
def _builtin_eval_bypass(value: int) -> None:
    tla.make_coord(eval("_plain_helper(value)"), 0)


@tla.kernel
def _sibling_nested_markers(limit: int) -> None:
    if limit > 0:
        local = 0
        if limit > 1:
            local = limit
        tla.make_coord(local, 0)
    else:
        local = 0
        if limit < -1:
            local = limit
        tla.make_coord(local, 0)


@tla.kernel
def _python_loop_nested_local(limit: int) -> None:
    if limit > 0:
        for local in (0,):
            if limit > 1:
                local = limit
            tla.make_coord(local, 0)


@tla.kernel
def _stale_origin_marker(limit: int) -> None:
    value = limit
    if limit > 0:
        value = limit + 1
    value = 3
    if limit > 1:
        value = 4
    tla.make_coord(value, 0)


@tla.kernel
def _dictionary_method_bypass(value: int) -> None:
    del value
    tla.make_coord(_USER_MAP["item"].read(), 0)


@pytest.mark.parametrize(
    ("kernel", "call"),
    [
        (_alias_branch_bypass, "plain_helper"),
        (_container_method_bypass, "user_method"),
    ],
)
def test_inspectable_staging_calls_execute_during_lowering(kernel, call: str) -> None:
    CALLS.clear()
    assert "tla.make_coord" in kernel.dump_mlir(type_args=(2,))
    assert CALLS == [call]


def test_eval_remains_rejected_without_execution() -> None:
    CALLS.clear()
    with pytest.raises(Exception):
        _builtin_eval_bypass.dump_mlir(type_args=(2,))
    assert CALLS == []


@pytest.mark.parametrize(
    "kernel",
    [_sibling_nested_markers, _python_loop_nested_local],
)
def test_nested_regions_preserve_valid_binding_state(kernel) -> None:
    assert "scf.if" in kernel.dump_mlir(type_args=(2,))


def test_normal_rebinding_refreshes_compiletime_origin() -> None:
    with pytest.raises(Exception, match="use tla.as_numeric"):
        _stale_origin_marker.dump_mlir(type_args=(2,))


def test_dictionary_held_user_object_executes_during_staging() -> None:
    CALLS.clear()
    assert "tla.make_coord" in _dictionary_method_bypass.dump_mlir(type_args=(2,))
    assert CALLS == ["user_method"]
