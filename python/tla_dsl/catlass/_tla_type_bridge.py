"""Native bridge for Tla TypeDef construction/access from structured data."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from mlir import ir as mlir_ir  # type: ignore[assignment]


class TlaTypeBridgeUnavailableError(RuntimeError):
    """Raised when native Tla type construction is unavailable."""


_EXTENSION: ModuleType | None | bool = None
_TREE_OPEN = -2
_TREE_CLOSE = -3
_DYNAMIC = -(2**63)


def shape_type_get(context: mlir_ir.Context, tree: Any) -> mlir_ir.Type:
    return _load_bridge_extension().shape_type_get(
        context, list(_encode_index_tree(tree))
    )


def coord_type_get(context: mlir_ir.Context, tree: Any) -> mlir_ir.Type:
    return _load_bridge_extension().coord_type_get(
        context, list(_encode_index_tree(tree))
    )


def stride_type_get(context: mlir_ir.Context, tree: Any) -> mlir_ir.Type:
    return _load_bridge_extension().stride_type_get(
        context, list(_encode_index_tree(tree))
    )


def layout_type_get(
    context: mlir_ir.Context,
    shape: Any,
    stride: Any,
    origin_shape: Any | None = None,
    layout: str = "row_major",
) -> mlir_ir.Type:
    origin = None if origin_shape is None else list(_encode_index_tree(origin_shape))
    return _load_bridge_extension().layout_type_get(
        context,
        list(_encode_index_tree(shape)),
        list(_encode_index_tree(stride)),
        origin,
        str(layout),
    )


def tensor_type_get(
    context: mlir_ir.Context,
    *,
    shape: Any,
    stride: Any,
    coord: Any,
    origin_shape: Any,
    element_type: mlir_ir.Type,
    addrspace: str,
    layout: str,
    ptr_alignment: int,
) -> mlir_ir.Type:
    if not isinstance(element_type, mlir_ir.Type):
        raise TypeError(
            "Tla tensor bridge expects element_type as mlir.ir.Type, "
            f"got {type(element_type).__name__}"
        )
    return _load_bridge_extension().tensor_type_get(
        context,
        list(_encode_index_tree(shape)),
        list(_encode_index_tree(stride)),
        list(_encode_index_tree(coord)),
        list(_encode_index_tree(origin_shape)),
        element_type,
        str(addrspace),
        str(layout),
        int(ptr_alignment),
    )


def ptr_type_get(
    context: mlir_ir.Context,
    pointee: mlir_ir.Type,
    addrspace: str,
    alignment: int,
) -> mlir_ir.Type:
    return _load_bridge_extension().ptr_type_get(
        context, pointee, str(addrspace), int(alignment)
    )


def ptr_pointee_type_get(
    context: mlir_ir.Context, ptr_type: mlir_ir.Type
) -> mlir_ir.Type:
    del context
    return _load_bridge_extension().ptr_pointee_type_get(ptr_type)


def ptr_addrspace(ptr_type: mlir_ir.Type) -> str:
    return str(_load_bridge_extension().ptr_addrspace(ptr_type))


def ptr_alignment(ptr_type: mlir_ir.Type) -> int:
    return int(_load_bridge_extension().ptr_alignment(ptr_type))


def memref_type_get(
    context: mlir_ir.Context,
    shape: Any,
    element_type: mlir_ir.Type,
    addrspace: str,
) -> mlir_ir.Type:
    if not isinstance(element_type, mlir_ir.Type):
        raise TypeError(
            "Tla memref bridge expects element_type as mlir.ir.Type, "
            f"got {type(element_type).__name__}"
        )
    return _load_bridge_extension().memref_type_get(
        context, list(_encode_memref_shape(shape)), element_type, str(addrspace)
    )


def memref_element_type_get(
    context: mlir_ir.Context, memref_type: mlir_ir.Type
) -> mlir_ir.Type:
    del context
    return _load_bridge_extension().memref_element_type_get(memref_type)


def memref_addrspace(memref_type: mlir_ir.Type) -> str:
    return str(_load_bridge_extension().memref_addrspace(memref_type))


def memref_shape(memref_type: mlir_ir.Type) -> tuple[int | None, ...]:
    return tuple(_load_bridge_extension().memref_shape(memref_type))


def layout_type_from_components_get(
    context: mlir_ir.Context,
    shape_type: mlir_ir.Type,
    stride_type: mlir_ir.Type,
    origin_shape_type: mlir_ir.Type | None = None,
    layout: str = "row_major",
) -> mlir_ir.Type:
    return _load_bridge_extension().layout_type_from_components_get(
        context, shape_type, stride_type, origin_shape_type, str(layout)
    )


def value_element_type_get(
    context: mlir_ir.Context, value_type: mlir_ir.Type
) -> mlir_ir.Type | None:
    del context
    try:
        return _load_bridge_extension().value_element_type_get(value_type)
    except ValueError:
        return None


def value_type_get(
    context: mlir_ir.Context, element_type: mlir_ir.Type | None = None
) -> mlir_ir.Type:
    if element_type is not None and not isinstance(element_type, mlir_ir.Type):
        raise TypeError(
            "Tla value bridge expects element_type as mlir.ir.Type, "
            f"got {type(element_type).__name__}"
        )
    return _load_bridge_extension().value_type_get(context, element_type)


def flag_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().flag_type_get(context)


def cross_flag_type_get(context: mlir_ir.Context) -> mlir_ir.Type:
    return _load_bridge_extension().cross_flag_type_get(context)


def load_tla_dialect(context: mlir_ir.Context) -> None:
    _load_bridge_extension().load_tla_dialect(context)


def type_is_ptr(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_ptr(type_like))


def type_is_tensor(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_tensor(type_like))


def type_is_memref(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_memref(type_like))


def type_is_shape(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_shape(type_like))


def type_is_coord(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_coord(type_like))


def type_is_stride(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_stride(type_like))


def type_is_layout(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_layout(type_like))


def type_is_value(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_value(type_like))


def type_is_flag(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_flag(type_like))


def type_is_cross_flag(type_like: mlir_ir.Type) -> bool:
    return bool(_load_bridge_extension().type_is_cross_flag(type_like))


def tla_type_category(type_like: mlir_ir.Type) -> str | None:
    category = _load_bridge_extension().tla_type_category(type_like)
    return None if category is None else str(category)


def _encode_memref_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, int) or shape is None:
        dims = (shape,)
    elif isinstance(shape, tuple):
        dims = shape
    else:
        raise TypeError(
            "Tla memref shape expects an int/None dimension or a tuple of dimensions; "
            f"got {type(shape).__name__}"
        )
    encoded: list[int] = []
    for dim in dims:
        if isinstance(dim, bool):
            raise TypeError("boolean not allowed in Tla memref shape")
        if dim is None:
            encoded.append(_DYNAMIC)
        elif isinstance(dim, int):
            if dim < 0:
                raise ValueError("Tla memref dimensions must be non-negative or None")
            encoded.append(int(dim))
        else:
            raise TypeError(
                "Tla memref shape expects static int dimensions or None; "
                f"got {type(dim).__name__}"
            )
    return tuple(encoded)


def _encode_index_tree(tree: Any) -> tuple[int, ...]:
    encoded: list[int] = []

    def append(node: Any, *, tuple_depth: int) -> None:
        if isinstance(node, bool):
            raise TypeError("boolean not allowed in Tla index tree")
        if node is None:
            encoded.append(_DYNAMIC)
            return
        if isinstance(node, int):
            encoded.append(int(node))
            return
        if isinstance(node, tuple):
            if len(node) == 0:
                raise ValueError("empty tuple in index tree")
            if tuple_depth >= 2:
                raise ValueError(
                    "Tla index tree supports only top-level leaves or one-level leaf groups"
                )
            encoded.append(_TREE_OPEN)
            for child in node:
                append(child, tuple_depth=tuple_depth + 1)
            encoded.append(_TREE_CLOSE)
            return
        if isinstance(node, list):
            raise ValueError(
                "expected nested tuple tree for index components (use parentheses, not brackets)"
            )
        raise TypeError(
            "Tla index type metadata expects static int leaves or None for dynamic leaves; "
            f"got {type(node).__name__}"
        )

    if isinstance(tree, tuple):
        if len(tree) == 0:
            raise ValueError("empty tuple in index tree")
        for child in tree:
            append(child, tuple_depth=1)
    else:
        append(tree, tuple_depth=0)
    return tuple(encoded)


def _load_bridge_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is False:
        raise TlaTypeBridgeUnavailableError("Tla type bridge extension is unavailable")
    if isinstance(_EXTENSION, ModuleType):
        return _EXTENSION
    path = _resolve_bridge_extension_path()
    if path is None:
        _EXTENSION = False
        raise TlaTypeBridgeUnavailableError(
            "Tla type bridge extension not found. Build the native type bridge "
            "module or set TLA_DSL_TYPE_BRIDGE_EXTENSION."
        )
    spec = importlib.util.spec_from_file_location(
        "catlass._tla_type_bridge_native", path
    )
    if spec is None or spec.loader is None:
        _EXTENSION = False
        raise TlaTypeBridgeUnavailableError(f"Failed to load Tla type bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTENSION = module
    return module


def _resolve_bridge_extension_path() -> Path | None:
    explicit = os.getenv("TLA_DSL_TYPE_BRIDGE_EXTENSION")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate
    packaged_root = Path(__file__).resolve().parent
    packaged = sorted(packaged_root.glob("_tla_type_bridge_native*.so"))
    if packaged:
        return packaged[0]
    # Nested in Catlass: .../python/tla_dsl/catlass/this.py -> .../python/tla_dsl/csrc/mlir/build/...
    dsl_root = Path(__file__).resolve().parents[1]
    nested = sorted(
        (dsl_root / "csrc" / "mlir" / "build" / "python" / "catlass").glob(
            "_tla_type_bridge_native*.so"
        )
    )
    if nested:
        return nested[0]
    # Legacy standalone ascend-catlass-DSL repo: .../python/<pkg>/this.py -> repo/mlir/build/...
    repo_root = Path(__file__).resolve().parents[2]
    candidates = sorted(
        (repo_root / "mlir" / "build" / "python" / "catlass").glob(
            "_tla_type_bridge_native*.so"
        )
    )
    if candidates:
        return candidates[0]
    return None


__all__ = [
    "TlaTypeBridgeUnavailableError",
    "tla_type_category",
    "coord_type_get",
    "cross_flag_type_get",
    "flag_type_get",
    "layout_type_from_components_get",
    "layout_type_get",
    "load_tla_dialect",
    "memref_addrspace",
    "memref_element_type_get",
    "memref_shape",
    "memref_type_get",
    "ptr_addrspace",
    "ptr_alignment",
    "ptr_pointee_type_get",
    "ptr_type_get",
    "shape_type_get",
    "stride_type_get",
    "tensor_type_get",
    "type_is_coord",
    "type_is_cross_flag",
    "type_is_flag",
    "type_is_layout",
    "type_is_memref",
    "type_is_ptr",
    "type_is_shape",
    "type_is_stride",
    "type_is_tensor",
    "type_is_value",
    "value_element_type_get",
    "value_type_get",
]
