import pytest

import catlass.tla as tla
import catlass.types as tla_types


_CALLS: list[str] = []
_STATIC_DIVIDEND = 130


def _plain_helper(value):
    _CALLS.append("plain")
    return value


def _impure_compiletime_helper(value: int) -> int:
    _CALLS.append("impure")
    return value


@tla.jit
def _jit_helper(value):
    tla.make_coord(value, 0)


@tla.jit
def _jit_calls_plain_helper(value):
    return _plain_helper(value)


class _UserValue:
    def __init__(self, value=1):
        _CALLS.append("init")
        self._value = value

    @property
    def value(self):
        _CALLS.append("property")
        return self._value

    def read(self):
        _CALLS.append("method")
        return self._value


_CAPTURED_USER_VALUE = object.__new__(_UserValue)
_CAPTURED_USER_VALUE._value = 1


@tla.kernel
def _bad_direct_helper_kernel(value: int) -> None:
    tla.make_coord(_plain_helper(value), 0)


@tla.kernel
def _bad_aliased_helper_kernel(value: int) -> None:
    helper = _plain_helper
    tla.make_coord(helper(value), 0)


@tla.kernel
def _bad_impure_compiletime_helper_kernel() -> None:
    tla.make_coord(_impure_compiletime_helper(_STATIC_DIVIDEND), 0)


@tla.kernel
def _bad_runtime_if_helper_kernel(value: int) -> None:
    if tla.arch.block_idx() == 0:
        _ = _plain_helper(value)


@tla.kernel
def _bad_runtime_for_helper_kernel(value: int) -> None:
    for _ in tla.range(0, value, 1):
        _ = _plain_helper(value)


@tla.kernel
def _bad_runtime_while_helper_kernel(value: int) -> None:
    i = 0
    while i < value:
        _ = _plain_helper(value)
        i += 1


@tla.kernel
def _bad_scope_helper_kernel(value: int) -> None:
    with tla.vector():
        _ = _plain_helper(value)


@tla.kernel
def _jit_helper_kernel(value: int) -> None:
    _jit_helper(value)


@tla.kernel
def _bad_transitive_helper_kernel(value: int) -> None:
    _ = _jit_calls_plain_helper(value)


@tla.kernel
def _callee_kernel(value: int) -> None:
    tla.make_coord(value, 0)


@tla.kernel
def _bad_kernel_call_kernel(value: int) -> None:
    _callee_kernel(value)


@tla.kernel
def _bad_user_constructor_kernel(value: int) -> None:
    item = _UserValue(value)
    tla.make_coord(item.value, 0)


@tla.kernel
def _bad_user_method_kernel(value: int) -> None:
    del value
    tla.make_coord(_CAPTURED_USER_VALUE.read(), 0)


@tla.kernel
def _bad_user_property_kernel(value: int) -> None:
    del value
    tla.make_coord(_CAPTURED_USER_VALUE.value, 0)


@tla.kernel
def _bad_user_argument_kernel(item) -> None:
    del item


@tla.kernel
def _isinstance_kernel(value: int) -> None:
    assert isinstance(value, tla_types.Int32)


@tla.kernel
def _bad_user_isinstance_kernel(value: int) -> None:
    assert not isinstance(value, _UserValue)


def test_allows_direct_inspectable_helper_during_staging() -> None:
    _CALLS.clear()
    assert "tla.make_coord" in _bad_direct_helper_kernel.dump_mlir(type_args=(1,))
    assert _CALLS == ["plain"]


def test_allows_aliased_inspectable_helper_during_staging() -> None:
    _CALLS.clear()
    assert "tla.make_coord" in _bad_aliased_helper_kernel.dump_mlir(type_args=(1,))
    assert _CALLS == ["plain"]


def test_allows_inspectable_helper_side_effects_during_staging() -> None:
    _CALLS.clear()
    assert "tla.make_coord" in _bad_impure_compiletime_helper_kernel.dump_mlir()
    assert _CALLS == ["impure"]


def test_allows_genuine_jit_helper() -> None:
    mlir = _jit_helper_kernel.dump_mlir(type_args=(1,))
    assert "tla.make_coord" in mlir


def test_allows_straight_line_jit_helper_with_plain_staging_call() -> None:
    _CALLS.clear()
    _bad_transitive_helper_kernel.dump_mlir(type_args=(1,))
    assert _CALLS == ["plain"]


def test_rejects_calling_a_kernel_as_a_helper() -> None:
    with pytest.raises(Exception, match="calling @tla.kernel.*use @tla.jit"):
        _bad_kernel_call_kernel.dump_mlir(type_args=(1,))


def test_allows_isinstance_with_builtin_type() -> None:
    _isinstance_kernel.dump_mlir(type_args=(1,))


def test_allows_isinstance_with_user_class_during_staging() -> None:
    _bad_user_isinstance_kernel.dump_mlir(type_args=(1,))


def test_allows_user_class_constructor_during_staging() -> None:
    _CALLS.clear()
    assert "tla.make_coord" in _bad_user_constructor_kernel.dump_mlir(type_args=(1,))
    assert _CALLS == ["init", "property"]


def test_allows_user_method_during_staging() -> None:
    _CALLS.clear()
    assert "tla.make_coord" in _bad_user_method_kernel.dump_mlir(type_args=(1,))
    assert _CALLS == ["method"]


def test_allows_user_property_during_staging() -> None:
    _CALLS.clear()
    assert "tla.make_coord" in _bad_user_property_kernel.dump_mlir(type_args=(1,))
    assert _CALLS == ["property"]


def test_rejects_user_class_kernel_argument() -> None:
    item = object.__new__(_UserValue)
    item._value = 1
    with pytest.raises(Exception, match="user-defined class.*_UserValue"):
        _bad_user_argument_kernel.dump_mlir(type_args=(item,))
