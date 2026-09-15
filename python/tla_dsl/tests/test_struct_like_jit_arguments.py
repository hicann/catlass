"""Structure, metadata and replay tests for recursive kernel argument trees."""

from __future__ import annotations

import weakref
from dataclasses import FrozenInstanceError, dataclass
from typing import NamedTuple

import pytest

import catlass.tla as tla
from catlass._mlir import ir as mlir_ir
from catlass.tla import Constexpr as Static
from catlass.base_dsl import BaseDSL
from catlass.base_dsl.runtime.argument_tree import (
    ArgumentTreeError,
    build_runtime_argument_tree,
    flatten_runtime_leaves,
    tree_unflatten,
)
from catlass.tla.runtime import make_fake_tensor


def _tensor(shape=(8,), stride=(1,), dtype=tla.Float32, *, layout_tag=tla.arch.RowMajor):
    return make_fake_tensor(dtype, shape, stride, layout_tag=layout_tag)


class _NamedAux(NamedTuple):
    bias: tla.Tensor
    limit: float


class _OtherNamedAux(NamedTuple):
    bias: tla.Tensor
    limit: float


@pytest.mark.parametrize("base", [tuple, list])
def test_container_subclass_preserves_type(base):
    class Values(base):
        def first(self):
            return self[0]

    tree = build_runtime_argument_tree(Values([1, 2]))
    rebuilt = tree_unflatten([3, 4], tree.treedef)
    assert type(rebuilt) is Values
    assert rebuilt.first() == 3
    assert flatten_runtime_leaves(Values([5, 6]), tree) == (5, 6)
    with pytest.raises(ArgumentTreeError, match="exact type"):
        flatten_runtime_leaves(base([5, 6]), tree)


def test_dataclass_undeclared_state_rejected_on_build_and_replay():
    @dataclass
    class Config:
        value: int

    config = Config(1)
    tree = build_runtime_argument_tree(config)
    config.extra = 7
    with pytest.raises(ArgumentTreeError, match="undeclared instance fields: extra"):
        build_runtime_argument_tree(config)
    with pytest.raises(ArgumentTreeError, match="undeclared instance fields: extra"):
        flatten_runtime_leaves(config, tree)


def test_dataclass_missing_field_reports_field_name():
    @dataclass
    class Config:
        value: int

    config = Config(1)
    del config.value
    with pytest.raises(ArgumentTreeError, match="missing field 'value'") as error:
        build_runtime_argument_tree(config)
    assert isinstance(error.value.__cause__, AttributeError)


def test_dataclass_fields_are_read_once_when_building_tree():
    reads = []

    @dataclass
    class Config:
        value: int

        def __getattribute__(self, name):
            if name == "value":
                reads.append(name)
            return object.__getattribute__(self, name)

    tree = build_runtime_argument_tree(Config(3))
    assert tree.runtime_leaf_count == 1
    assert reads == ["value"]


@dataclass
class _AliasedStaticConfig:
    tile: Static[int]
    value: int


class _AliasedStaticTuple(NamedTuple):
    tile: Static[int]
    value: int


@pytest.mark.parametrize("config_type", [_AliasedStaticConfig, _AliasedStaticTuple])
def test_deferred_constexpr_alias_preserves_static_field(config_type):
    value = config_type(128, 4)
    tree = build_runtime_argument_tree(value)
    assert tree.runtime_leaf_count == 1
    assert flatten_runtime_leaves(config_type(128, 9), tree) == (9,)
    rebuilt = tree_unflatten([7], tree.treedef)
    assert (rebuilt.tile, rebuilt.value) == (128, 7)

    def kernel(aux):
        _ = aux.value + aux.tile

    lowered = BaseDSL()._lower(kernel, kind="kernel", options={}, type_args=(value,))
    assert lowered.argument_trees[0].runtime_leaf_count == 1
    assert "arith.constant" in lowered.asm(generic=True)


def test_dataclass_tree_does_not_retain_sample_instance():
    @dataclass
    class Pair:
        left: int
        right: float

    sample = Pair(2, 3.0)
    sample_ref = weakref.ref(sample)
    tree = build_runtime_argument_tree(sample)
    del sample
    assert sample_ref() is None
    rebuilt = tree_unflatten([4, 5.0], tree.treedef)
    assert type(rebuilt) is Pair
    assert (rebuilt.left, rebuilt.right) == (4, 5.0)


@dataclass
class _Aux:
    tile: tla.Constexpr[int]
    nested: _NamedAux
    enabled: bool


def test_structural_leaves_index_the_single_type_binding_table():
    tensor = _tensor()
    gm = _tensor().mark_compact_shape_dynamic(0)
    value = (tensor, [gm, 3, None], _AliasedStaticConfig(128, 4))
    tree = build_runtime_argument_tree(value)
    first, nested, config = tree.treedef.child_treedefs
    dynamic_leaves = [first, *nested.child_treedefs[:2], config.child_treedefs[1]]
    assert [leaf.binding_index for leaf in dynamic_leaves] == [0, 1, 2, 3]
    assert all(not hasattr(leaf, "mlir_type") for leaf in dynamic_leaves)
    assert [binding.path for binding in tree.bindings] == [
        (0,),
        (1, 0),
        (1, 1),
        (2, "value"),
    ]
    assert tree.bindings[0].kind == "pointer"
    assert tree.bindings[1].kind == "dynamic_gm"
    assert config.child_treedefs[0].const_value == 128
    assert nested.child_treedefs[2].is_none
    assert flatten_runtime_leaves(value, tree) == (tensor, gm, 3, 4)


def test_dynamic_gm_markers_follow_top_level_and_nested_physical_slots() -> None:
    def kernel(prefix, top, aux, suffix) -> None:
        pass

    top = _tensor().mark_compact_shape_dynamic(0)
    nested = _tensor().mark_compact_shape_dynamic(0)
    named = _tensor().mark_compact_shape_dynamic(0)
    aux = (_tensor(), [nested, _NamedAux(named, 0.5)])
    lowered = BaseDSL()._lower(
        kernel,
        kind="kernel",
        options={},
        type_args=(7, top, aux, 1.0),
    )

    func = lowered.module.body.operations[0]
    entry = func.regions[0].blocks[0]
    arg_attrs = mlir_ir.ArrayAttr(func.attributes["arg_attrs"])
    assert len(arg_attrs) == len(entry.arguments) == 13
    # Each Dynamic-GM leaf owns a memref followed by two untagged origin indices.
    # Prefix/suffix scalars, the static tensor at slot 4, and named.limit stay bare.
    assert [
        index
        for index, attrs in enumerate(arg_attrs)
        if "tla.dynamic_gm" in mlir_ir.DictAttr(attrs)
    ] == [1, 5, 8]
    assert all(
        mlir_ir.UnitAttr.isinstance(mlir_ir.DictAttr(arg_attrs[index])["tla.dynamic_gm"])
        for index in (1, 5, 8)
    )
    assert all(
        str(entry.arguments[index].type) == "index"
        for index in (2, 3, 6, 7, 9, 10)
    )


def test_recursive_tree_preserves_order_and_rebuilds_structure() -> None:
    bias0 = _tensor()
    bias1 = _tensor()
    value = (bias0, _Aux(128, _NamedAux(bias1, 0.5), True))

    tree = build_runtime_argument_tree(value)
    assert tree.runtime_leaf_count == 4
    leaves = flatten_runtime_leaves(value, tree)
    assert leaves == (bias0, bias1, 0.5, True)

    rebuilt = tree_unflatten(("b0", "b1", "limit", "enabled"), tree.treedef)
    assert rebuilt[0] == "b0"
    assert isinstance(rebuilt[1], _Aux)
    assert isinstance(rebuilt[1].nested, _NamedAux)
    assert rebuilt[1].tile == 128
    assert rebuilt[1].nested == _NamedAux("b1", "limit")
    assert rebuilt[1].enabled == "enabled"


def test_bool_is_scalar_i1_and_none_is_unit() -> None:
    tree = build_runtime_argument_tree((False, None, 3, 0.25))
    assert tree.runtime_leaf_count == 3
    assert [binding.kind for binding in tree.bindings] == [
        "scalar",
        "scalar",
        "scalar",
    ]
    assert [str(binding.mlir_type) for binding in tree.bindings] == [
        "i1",
        "i32",
        "f32",
    ]
    assert flatten_runtime_leaves((False, None, 3, 0.25), tree) == (
        False,
        3,
        0.25,
    )
    assert tree_unflatten((False, 3, 0.25), tree.treedef) == (
        False,
        None,
        3,
        0.25,
    )


def test_int_conversion_alone_does_not_make_a_runtime_scalar():
    class Value:
        def __int__(self):
            return 3

    with pytest.raises(ArgumentTreeError):
        build_runtime_argument_tree(Value())


@pytest.mark.parametrize("static", [float("nan"), 0.0, -0.0, 1, True, None])
def test_static_fields_are_compile_time_inputs_only(static):
    @dataclass
    class Config:
        constant: tla.Constexpr[object]
        value: int

    value = Config(static, 3)
    tree = build_runtime_argument_tree(value)
    assert flatten_runtime_leaves(value, tree) == (3,)
    assert flatten_runtime_leaves(Config(object(), 4), tree) == (4,)
    rebuilt = tree_unflatten([5], tree.treedef)
    assert rebuilt.constant is static
    assert rebuilt.value == 5
    unit_tree = build_runtime_argument_tree((None,))
    with pytest.raises(ArgumentTreeError, match="expected None"):
        flatten_runtime_leaves((3,), unit_tree)


@pytest.mark.parametrize("kind", ["tuple", "list", "namedtuple", "dataclass"])
def test_container_replay_reuses_metadata_and_reads_current_values(monkeypatch, kind):
    from catlass.base_dsl.runtime import argument_tree, jit_arg_adapters

    class Pair(NamedTuple):
        left: int
        right: int

    @dataclass
    class Config:
        left: int
        right: int

    factory = {
        "tuple": tuple,
        "list": list,
        "namedtuple": lambda v: Pair(*v),
        "dataclass": lambda v: Config(*v),
    }[kind]
    tree = build_runtime_argument_tree(factory((1, 2)))

    def unexpected(*args, **kwargs):
        pytest.fail("launch must not rebuild container metadata or resolve annotations")

    monkeypatch.setattr(argument_tree, "_node_metadata", unexpected)
    monkeypatch.setattr(argument_tree, "_namedtuple_constexpr_fields", unexpected)
    monkeypatch.setattr(jit_arg_adapters, "_dataclass_members", unexpected)
    if kind == "dataclass":
        monkeypatch.setattr(jit_arg_adapters, "fields", unexpected)
    assert flatten_runtime_leaves(factory((3, 4)), tree) == (3, 4)
    if kind in ("tuple", "list"):
        with pytest.raises(ArgumentTreeError, match="children"):
            flatten_runtime_leaves(factory((3,)), tree)
    if kind == "dataclass":
        value = factory((3, 4))
        value.extra = 5
        with pytest.raises(ArgumentTreeError, match="undeclared"):
            flatten_runtime_leaves(value, tree)
        del value.extra
        del value.left
        with pytest.raises(ArgumentTreeError, match="missing"):
            flatten_runtime_leaves(value, tree)


def test_container_replay_releases_runtime_values_without_cyclic_gc():
    import gc

    tree = build_runtime_argument_tree((_tensor(),))
    enabled = gc.isenabled()
    gc.disable()
    try:
        value = _tensor()
        reference = weakref.ref(value)
        leaves = flatten_runtime_leaves((value,), tree)
        del leaves, value
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def test_namedtuple_type_is_part_of_the_replayed_schema() -> None:
    value = _NamedAux(_tensor(), 0.5)
    tree = build_runtime_argument_tree(value)
    with pytest.raises(ArgumentTreeError, match="runtime argument structure changed"):
        flatten_runtime_leaves(_OtherNamedAux(_tensor(), 0.5), tree)


def test_cycles_and_unordered_containers_fail_closed() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ArgumentTreeError, match="cyclic runtime argument"):
        build_runtime_argument_tree(cyclic)
    for value in ({1, 2}, {"bias": _tensor()}):
        with pytest.raises(ArgumentTreeError, match="unsupported runtime argument"):
            build_runtime_argument_tree(value)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_tensor_replay_reads_current_value_without_metadata_checks(
    monkeypatch, dynamic, nested
) -> None:
    def wrap(value):
        return ([value, None],) if nested else value

    value = _tensor()
    if dynamic:
        value.mark_compact_shape_dynamic(0)
    tree = build_runtime_argument_tree(wrap(value))
    replacement = _tensor((16,), (1,), tla.Float16)
    if dynamic:
        replacement.mark_compact_shape_dynamic(0)
    replacement.data_ptr = 0x2000

    def unexpected(*args):
        pytest.fail("replay must not reconstruct a type descriptor")

    monkeypatch.setattr(type(value), "tla_tensor_type_descriptor", unexpected)
    # Replay only: the replacement deliberately violates the compute contract
    # and must not be launched against the compiled specialization.
    (leaf,) = flatten_runtime_leaves(wrap(replacement), tree)
    assert leaf is replacement
    assert leaf.data_ptr == 0x2000


def test_tree_metadata_is_immutable() -> None:
    tree = build_runtime_argument_tree((_tensor(), 3))
    metadata = tree.treedef.node_metadata
    with pytest.raises(FrozenInstanceError):
        metadata.child_keys = (0,)  # type: ignore[misc]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_custom_protocol_does_not_enable_kernel_argument_admission(nested, registered):
    from catlass.base_dsl import BaseDSL
    from catlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry

    class Custom:
        def __get_mlir_types__(self, context):
            return ["i32", "i32"]

        def __c_pointers__(self):
            return [1, 2]

        def __new_from_mlir_values__(self, values):
            return values

    value = (Custom(),) if nested else Custom()

    def kernel(aux):
        pass

    if registered:
        JitArgAdapterRegistry.register_jit_arg_adapter(Custom)(lambda value: (1, 2))
    try:
        with pytest.raises(ArgumentTreeError, match="unsupported runtime argument"):
            build_runtime_argument_tree(value)
        with pytest.raises(TypeError, match="user-defined class 'Custom'"):
            BaseDSL()._lower(kernel, kind="kernel", options={}, type_args=(value,))
    finally:
        JitArgAdapterRegistry.jit_arg_adapter_registry.pop(Custom, None)
