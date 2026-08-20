import pytest

import catlass.tla as tla
from catlass.tla import as_numeric as direct_as_numeric


@tla.kernel
def _plain_runtime_if_write(limit: int) -> None:
    scale = 2
    if limit > 0:
        scale = 3
        tla.make_coord(scale, 0)


@tla.kernel
def _promoted_runtime_if_write(limit: int) -> None:
    scale = 2
    if limit > 0:
        scale = tla.as_numeric(3)
        tla.make_coord(scale, 0)
    tla.make_coord(scale, 0)


@tla.kernel
def _direct_import_runtime_if_write(limit: int) -> None:
    value = 2
    if limit > 0:
        value = direct_as_numeric(3)
    tla.make_coord(value, 0)


def _untrusted_as_numeric(value: int) -> int:
    return value


@tla.kernel
def _globally_shadowable_direct_import_runtime_if_write(limit: int) -> None:
    value = 2
    if limit > 0:
        value = direct_as_numeric(3)
    tla.make_coord(value, 0)


@tla.kernel
def _promoted_runtime_if_else_carry(limit: int) -> None:
    scale = 2
    if limit > 0:
        scale = tla.as_numeric(3)
    else:
        scale = tla.as_numeric(4)
    tla.make_coord(scale, 0)


@tla.kernel
def _as_numeric_value_in_range_constexpr() -> None:
    limit = tla.as_numeric(2)
    for _ in tla.range_constexpr(limit):
        pass


@tla.kernel
def _plain_runtime_for_write(limit: int) -> None:
    scale = 2
    for index in tla.range(0, limit, 1):
        scale = index
        tla.make_coord(scale, 0)


@tla.kernel
def _promoted_runtime_for_write(limit: int) -> None:
    scale = 2
    for index in tla.range(0, limit, 1):
        scale = tla.as_numeric(index)
    tla.make_coord(scale, 0)


@tla.kernel
def _plain_runtime_while_write(limit: int) -> None:
    scale = 2
    index = tla.as_numeric(0)
    while index < limit:
        scale = 3
        tla.make_coord(scale, 0)
        index += 1


@tla.kernel
def _promoted_runtime_while_write(limit: int) -> None:
    scale = 2
    index = tla.as_numeric(0)
    while index < limit:
        scale = tla.as_numeric(3)
        index += 1
    tla.make_coord(scale, 0)


@tla.kernel
def _promoted_literal_runtime_while_write(limit: int) -> None:
    index = tla.as_numeric(0)
    while index < limit:
        index += 1
    tla.make_coord(index, 0)


@tla.kernel
def _plain_runtime_with_write() -> None:
    scale = 2
    with tla.vector():
        scale = 3
        tla.make_coord(scale, 0)


@tla.kernel
def _promoted_runtime_with_write() -> None:
    scale = 2
    with tla.vector():
        scale = tla.as_numeric(3)
        tla.make_coord(scale, 0)


@tla.kernel
def _promoted_runtime_list_comprehension_write(limit: int) -> None:
    state = [1, 2]
    if limit > 0:
        state = [tla.as_numeric(value) for value in (1, 2)]
    tla.make_coord(state[0], 0)


@tla.kernel
def _tuple_runtime_write(limit: int) -> None:
    scale, offset = 2, 0
    if limit > 0:
        scale, offset = 3, 1
    tla.make_coord(scale, offset)


@tla.kernel
def _promoted_tuple_runtime_write(limit: int) -> None:
    scale, offset = 2, 0
    if limit > 0:
        scale, offset = tuple(tla.as_numeric(value) for value in (3, 1))
    tla.make_coord(scale, offset)


@tla.kernel
def _promoted_tuple_if_else_carry(limit: int) -> None:
    x, y = 0, 1
    if limit > 0:
        x, y = tuple(tla.as_numeric(value) for value in (limit, limit + 1))
    else:
        x, y = tuple(tla.as_numeric(value) for value in (limit + 2, limit + 3))
    tla.make_coord(x, y)


@tla.kernel
def _locally_shadowed_tuple_promotion(limit: int) -> None:
    state = (1, 2)
    tuple = _shadowed_tuple
    if limit > 0:
        state = tuple(tla.as_numeric(value) for value in (1, 2))
    tla.make_coord(state[0], 0)


@tla.kernel
def _globally_shadowable_tuple_promotion(limit: int) -> None:
    state = (1, 2)
    if limit > 0:
        state = tuple(tla.as_numeric(value) for value in (1, 2))
    tla.make_coord(state[0], 0)


def _shadowed_tuple(values: object) -> tuple[int, int]:
    del values
    return (1, 2)


@tla.kernel
def _tuple_rebinding_refreshes_origin(limit: int) -> None:
    scale, offset = tuple(tla.as_numeric(value) for value in (2, 0))
    scale, offset = 2, 0
    if limit > 0:
        scale, offset = 3, 1
    tla.make_coord(scale, offset)


@tla.kernel
def _plain_runtime_python_for_target_write(limit: int) -> None:
    scale = 2
    if limit > 0:
        for scale in (limit,):
            tla.make_coord(scale, 0)
    tla.make_coord(scale, 0)


@pytest.mark.parametrize(
    "kernel, type_args",
    [
        (_plain_runtime_if_write, (2,)),
        (_plain_runtime_for_write, (2,)),
        (_plain_runtime_while_write, (2,)),
        (_plain_runtime_with_write, ()),
        (_tuple_runtime_write, (2,)),
        (_plain_runtime_python_for_target_write, (2,)),
    ],
    ids=["if", "for", "while", "with", "tuple-if", "python-for-target"],
)
def test_runtime_control_flow_rejects_plain_compile_time_binding_writes(
    kernel, type_args
) -> None:
    with pytest.raises(Exception, match="use tla.as_numeric"):
        kernel.dump_mlir(type_args=type_args)


@pytest.mark.parametrize(
    "kernel, operation, type_args",
    [
        (_promoted_runtime_if_write, "scf.if", (2,)),
        (_promoted_runtime_for_write, "scf.for", (2,)),
        (_promoted_runtime_while_write, "scf.while", (2,)),
        (_promoted_literal_runtime_while_write, "scf.while", (2,)),
        (_promoted_runtime_with_write, "tla.vector", ()),
        (_promoted_tuple_runtime_write, "scf.if", (2,)),
    ],
    ids=["if", "for", "while", "literal-while", "with", "tuple-if"],
)
def test_runtime_control_flow_accepts_explicit_promotion(
    kernel, operation, type_args
) -> None:
    assert operation in kernel.dump_mlir(type_args=type_args)


def test_runtime_if_else_carries_explicitly_promoted_value_after_branch() -> None:
    assert "scf.if" in _promoted_runtime_if_else_carry.dump_mlir(type_args=(2,))


def test_runtime_if_else_carries_explicitly_promoted_tuple_after_branch() -> None:
    assert "scf.if" in _promoted_tuple_if_else_carry.dump_mlir(type_args=(2,))


def test_tuple_rebinding_to_compile_time_values_requires_promotion() -> None:
    with pytest.raises(
        Exception, match="compile-time binding '(scale|offset)'.*use tla.as_numeric"
    ):
        _tuple_rebinding_refreshes_origin.dump_mlir(type_args=(2,))


def test_as_numeric_value_is_usable_in_range_constexpr() -> None:
    assert "tla.func" in _as_numeric_value_in_range_constexpr.dump_mlir()


def test_runtime_control_flow_accepts_direct_as_numeric_import() -> None:
    assert "scf.if" in _direct_import_runtime_if_write.dump_mlir(type_args=(2,))


def test_runtime_control_flow_rejects_shadowed_direct_as_numeric_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        _globally_shadowable_direct_import_runtime_if_write.fn.__globals__,
        "direct_as_numeric",
        _untrusted_as_numeric,
    )
    with pytest.raises(Exception, match="use tla.as_numeric"):
        _globally_shadowable_direct_import_runtime_if_write.dump_mlir(type_args=(2,))


def test_runtime_control_flow_accepts_list_comprehension_composition() -> None:
    assert "scf.if" in _promoted_runtime_list_comprehension_write.dump_mlir(
        type_args=(2,)
    )


class _UserClass:
    pass


def test_as_numeric_rejects_user_class() -> None:
    with pytest.raises(ValueError, match="unable to convert"):
        tla.as_numeric(_UserClass())


@tla.kernel
def _promoted_runtime_dict_comprehension_write(limit: int) -> None:
    state = {"scale": 3}
    if limit > 0:
        state = {key: tla.as_numeric(value) for key, value in (("scale", 3),)}
    tla.make_coord(state["scale"], 0)


@tla.kernel
def _plain_runtime_container_literal_write(limit: int) -> None:
    state = [1, 2]
    if limit > 0:
        state = [tla.as_numeric(1), tla.as_numeric(2)]
    tla.make_coord(state[0], 0)


@tla.kernel
def _partially_promoted_runtime_collection_write(limit: int) -> None:
    state = [1, 2]
    if limit > 0:
        state = [tla.as_numeric(1), 2]
    tla.make_coord(state[0], 0)


@tla.kernel
def _starred_iterable_runtime_collection_write(limit: int) -> None:
    state = [1, 2]
    if limit > 0:
        state = [tla.as_numeric(value) for value in (1, *(2,))]
    tla.make_coord(state[0], 0)


@tla.kernel
def _runtime_keyed_dict_comprehension_write(limit: int) -> None:
    state = {"scale": 3}
    if limit > 0:
        state = {key: tla.as_numeric(value) for key, value in ((limit, 3),)}
    tla.make_coord(state["scale"], 0)


def test_runtime_control_flow_accepts_tuple_and_dict_comprehension_composition() -> (
    None
):
    assert "scf.if" in _promoted_tuple_if_else_carry.dump_mlir(type_args=(2,))
    assert "scf.if" in _promoted_runtime_dict_comprehension_write.dump_mlir(
        type_args=(2,)
    )


def test_runtime_tuple_promotion_rejects_shadowed_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(Exception):
        _locally_shadowed_tuple_promotion.dump_mlir(type_args=(2,))

    monkeypatch.setitem(
        _globally_shadowable_tuple_promotion.fn.__globals__, "tuple", _shadowed_tuple
    )
    with pytest.raises(
        Exception, match="as_numeric|comprehension|compile-time binding"
    ):
        _globally_shadowable_tuple_promotion.dump_mlir(type_args=(2,))


@pytest.mark.parametrize(
    "kernel",
    [
        _plain_runtime_container_literal_write,
        _partially_promoted_runtime_collection_write,
        _starred_iterable_runtime_collection_write,
        _runtime_keyed_dict_comprehension_write,
    ],
    ids=["literal", "partial", "starred-iterable", "runtime-dict-key"],
)
def test_runtime_control_flow_rejects_non_comprehension_collection_composition(
    kernel,
) -> None:
    with pytest.raises(
        Exception, match="as_numeric|comprehension|compile-time binding"
    ):
        kernel.dump_mlir(type_args=(2,))


def test_to_runtime_is_not_a_public_api() -> None:
    assert not hasattr(tla, "to_runtime")
