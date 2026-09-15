"""TLA runtime support for recursive struct-like kernel arguments.

The structural vocabulary lives in :mod:`base_dsl.utils.tree_utils`.  This
module adds the TLA leaf classification and the host/device bridge used by
execution lowering.  A Dynamic-GM tensor is one logical tree leaf; its three
physical descriptor arguments are handled by the TLA binding layer.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Iterable, Sequence

from ..typing import Numeric
from ..utils.tree_utils import (
    ArgumentTreeError,
    Leaf,
    LeafKind,
    NodeType,
    PyTreeDef,
    TreeDef,
    TreeMetadata,
    _walk_runtime_tree,
    is_namedtuple_instance,
    tree_unflatten,
)
from ...tla.typing import Tensor
from .argument_binding import (
    RuntimeArgumentTree,
    RuntimeLeafBinding,
)
from .jit_arg_adapters import (
    _value_mlir_types,
    _flatten_dataclass,
    _unflatten_dataclass,
    _is_constexpr_annotation,
    _resolved_field_annotations,
)


def _is_dynamic_gm_host(value: Any) -> bool:
    """Return whether a host value belongs to the separate Dynamic-GM ABI."""

    from ...core_api import is_dynamic_gm_tensor_arg

    return is_dynamic_gm_tensor_arg(value)


def _scalar_mlir_type(value: Any) -> str | None:
    # bool is an int subclass and must be handled before int.
    if isinstance(value, bool):
        return "i1"
    if isinstance(value, int):
        return "i32"
    if isinstance(value, float):
        return "f32"
    return None


def _node_metadata(
    *,
    node_type: type[Any],
    fields_: tuple[str, ...] = (),
    constexpr_indices: tuple[int, ...] = (),
    child_keys: tuple[Any, ...] = (),
) -> TreeMetadata:
    return TreeMetadata(
        python_type=node_type,
        fields=fields_,
        constexpr_indices=constexpr_indices,
        child_keys=child_keys,
    )


def _flatten_tuple(
    value: Any, expected_metadata: TreeMetadata | None
) -> tuple[TreeMetadata, Sequence[Any]]:
    if expected_metadata is not None:
        return expected_metadata, value
    return (
        _node_metadata(
            node_type=type(value),
            child_keys=tuple(range(len(value))),
        ),
        list(value),
    )


def _unflatten_tuple(
    metadata: TreeMetadata, children: Iterable[Any]
) -> tuple[Any, ...]:
    return metadata.python_type(tuple(children))


def _flatten_list(
    value: Any, expected_metadata: TreeMetadata | None
) -> tuple[TreeMetadata, Sequence[Any]]:
    if expected_metadata is not None:
        return expected_metadata, value
    return (
        _node_metadata(
            node_type=type(value),
            child_keys=tuple(range(len(value))),
        ),
        list(value),
    )


def _unflatten_list(metadata: TreeMetadata, children: Iterable[Any]) -> list[Any]:
    return metadata.python_type(list(children))


def _namedtuple_constexpr_fields(value: Any) -> tuple[str, ...]:
    annotations = _resolved_field_annotations(type(value))
    return tuple(
        name
        for name in type(value)._fields
        if _is_constexpr_annotation(annotations.get(name))
    )


def _flatten_namedtuple(
    value: Any, expected_metadata: TreeMetadata | None
) -> tuple[TreeMetadata, Sequence[Any]]:
    names = tuple(type(value)._fields)
    if expected_metadata is not None:
        metadata = expected_metadata
        if names != metadata.fields:
            raise ArgumentTreeError("NamedTuple fields changed after compilation")
        return metadata, value
    constexpr_fields = _namedtuple_constexpr_fields(value)
    return (
        _node_metadata(
            node_type=type(value),
            fields_=names,
            constexpr_indices=tuple(
                i for i, name in enumerate(names) if name in constexpr_fields
            ),
            child_keys=tuple(range(len(value))),
        ),
        list(value),
    )


def _unflatten_namedtuple(metadata: TreeMetadata, children: Iterable[Any]) -> Any:
    return metadata.python_type(*children)


_NODE_TYPES = {
    "tuple": NodeType("tuple", _flatten_tuple, _unflatten_tuple),
    "list": NodeType("list", _flatten_list, _unflatten_list),
    "namedtuple": NodeType("namedtuple", _flatten_namedtuple, _unflatten_namedtuple),
    "dataclass": NodeType("dataclass", _flatten_dataclass, _unflatten_dataclass),
}


def _aggregate_node_type(value: Any) -> NodeType | None:
    # NamedTuple is a tuple subclass and must be selected first.
    if is_namedtuple_instance(value):
        return _NODE_TYPES["namedtuple"]
    if isinstance(value, tuple):
        return _NODE_TYPES["tuple"]
    if isinstance(value, list):
        return _NODE_TYPES["list"]
    if is_dataclass(value) and not isinstance(value, type):
        return _NODE_TYPES["dataclass"]
    return None


def is_runtime_argument_tree_candidate(value: Any) -> bool:
    """Return whether *value* should enter the recursive tree builder.

    Unknown values (notably an unannotated callable or string) must remain on
    the ordinary type-resolution path so the existing diagnostic can explain
    that they are not runtime kernel arguments.  A value is a tree candidate
    only when it is a supported leaf or an ordered aggregate.
    """

    # ``None`` is the zero-payload Unit leaf.  It still belongs to the
    # logical kernel signature and must be represented by a tree so launch
    # validation can distinguish it from an omitted argument.
    if value is None:
        return True
    if isinstance(value, (Numeric, Tensor, int, float)):
        return True
    if _aggregate_node_type(value) is not None:
        return True
    return False


def build_runtime_argument_trees(
    arg_names: Sequence[str],
    call_args: Sequence[Any],
    constexpr_names: set[str],
    context: Any,
) -> dict[str, RuntimeArgumentTree]:
    """Build one tree per supported runtime argument, including Dynamic-GM."""

    trees: dict[str, RuntimeArgumentTree] = {}
    for name, value in zip(arg_names, call_args, strict=False):
        if name in constexpr_names or not is_runtime_argument_tree_candidate(value):
            continue
        try:
            trees[name] = build_runtime_argument_tree(value, context)
        except ArgumentTreeError as exc:
            raise ArgumentTreeError(
                f"kernel argument {name!r} has an unsupported runtime "
                f"parameter tree: {exc}"
            ) from exc
    return trees


def _build_tree(
    value: Any,
    context: Any,
    *,
    path: tuple[Any, ...],
    active: set[int],
    bindings: list[RuntimeLeafBinding],
    constexpr: bool = False,
) -> TreeDef:
    def bind(kind: str, mlir_type: Any) -> Leaf:
        index = len(bindings)
        bindings.append(RuntimeLeafBinding(path, mlir_type, kind))
        return Leaf(kind="runtime", binding_index=index)

    if constexpr:
        return Leaf(kind="constexpr", const_value=value)
    if value is None:
        return Leaf(kind="unit")
    if _is_dynamic_gm_host(value):
        types = _value_mlir_types(value, context)
        if len(types) != 1:
            raise ArgumentTreeError(
                f"Dynamic-GM tensor at {path!r} must expose one logical tensor value"
            )
        return bind("dynamic_gm", types[0])
    if isinstance(value, Numeric):
        types = _value_mlir_types(value, context)
        if len(types) != 1:
            raise ArgumentTreeError(
                f"Numeric at {path!r} must expose one runtime value"
            )
        return bind("scalar", types[0])
    if isinstance(value, Tensor):
        types = _value_mlir_types(value, context)
        if len(types) != 1:
            raise ArgumentTreeError(f"Tensor at {path!r} must expose one runtime value")
        return bind("pointer", types[0])

    node_type = _aggregate_node_type(value)
    if node_type is not None:
        identity = id(value)
        if identity in active:
            raise ArgumentTreeError(f"cyclic runtime argument at {path!r}")
        active.add(identity)
        try:
            metadata, children_values = node_type.to_iterable(value, None)
            keys = metadata.child_keys
            constexpr_keys = {keys[index] for index in metadata.constexpr_indices}
            children: list[TreeDef] = []
            for key, child in zip(keys, children_values, strict=True):
                children.append(
                    _build_tree(
                        child,
                        context,
                        path=path + (key,),
                        active=active,
                        bindings=bindings,
                        constexpr=key in constexpr_keys,
                    )
                )
        finally:
            active.remove(identity)
        return PyTreeDef(node_type, metadata, tuple(children))

    scalar = _scalar_mlir_type(value)
    if scalar is not None:
        return bind("scalar", scalar)
    raise ArgumentTreeError(
        f"unsupported runtime argument at {path or ('argument',)!r}: "
        f"{type(value).__name__}; expected Tensor/Numeric, scalar, tuple/list, "
        "NamedTuple, or dataclass"
    )


def build_runtime_argument_tree(value: Any, context: Any = None) -> RuntimeArgumentTree:
    """Build a structural tree and physical leaf bindings for one argument."""

    bindings: list[RuntimeLeafBinding] = []
    treedef = _build_tree(value, context, path=(), active=set(), bindings=bindings)
    return RuntimeArgumentTree(treedef=treedef, bindings=tuple(bindings))


def flatten_runtime_leaves(value: Any, tree: RuntimeArgumentTree) -> tuple[Any, ...]:
    """Replay the argument structure and extract logical launch leaves."""

    # A plain runtime argument has no container structure to replay.
    if isinstance(tree.treedef, Leaf) and tree.treedef.is_runtime:
        return (value,)

    def handle_leaf(item: Any, node: Leaf, path: tuple[Any, ...]) -> Iterable[Any]:
        if node.is_runtime:
            return (item,)
        if node.is_none:
            if item is not None:
                raise ArgumentTreeError(f"expected None at {path!r}")
        return ()

    return _walk_runtime_tree(value, tree.treedef, leaf_handler=handle_leaf)


def runtime_leaf_templates(value: Any, treedef: TreeDef) -> tuple[Any, ...]:
    """Return host owners in runtime binding order."""

    def handle_leaf(item: Any, node: Leaf, path: tuple[Any, ...]) -> Iterable[Any]:
        del path
        return (item,) if node.is_runtime else ()

    return _walk_runtime_tree(value, treedef, leaf_handler=handle_leaf)


__all__ = [
    "ArgumentTreeError",
    "Leaf",
    "LeafKind",
    "NodeType",
    "PyTreeDef",
    "RuntimeArgumentTree",
    "RuntimeLeafBinding",
    "TreeDef",
    "TreeMetadata",
    "build_runtime_argument_tree",
    "build_runtime_argument_trees",
    "flatten_runtime_leaves",
    "is_namedtuple_instance",
    "is_runtime_argument_tree_candidate",
    "runtime_leaf_templates",
    "tree_unflatten",
]
