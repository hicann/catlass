"""Lightweight Tla DSL type helpers, MLIR ``!tla.ptr`` (:class:`PtrType`), and addrspace helpers."""

from __future__ import annotations

import ctypes
import importlib
import struct
import weakref
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Iterator, Literal, TypeAlias

from .base_dsl.typing import (
    Numeric,
    Pointer as PointerABC,
    _elem_token_to_mlir_type,
)

from mlir import ir as mlir_ir  # type: ignore[assignment]

from ._mlir_bindings import tla_ops_gen as _tla_ops_gen  # noqa: F401 — register ``tla`` dialect
from . import _tla_type_bridge
from .address_space import AddressSpace


def _coerce_host_tensor_dtype(value: Any) -> str:
    if isinstance(value, type) and issubclass(value, Numeric):
        token = getattr(value, "dtype", "") or ""
        if not token:
            raise TypeError(
                "Tensor dtype=... expected a concrete Tla element type (e.g. tla.Float16)"
            )
        return str(token).strip().lower()
    raise TypeError(
        "Tensor dtype=... expected a Tla element type (e.g. tla.Float16), not str; "
        f"got {type(value).__name__}"
    )


def _coerce_host_tensor_addrspace(value: Any) -> str:
    if isinstance(value, AddressSpace):
        return value.name
    raise TypeError(
        "Tensor addrspace=... expected tla.AddressSpace, not str; "
        f"got {type(value).__name__}"
    )


_cached_ptr_typeid: mlir_ir.TypeID | None = None
_cached_memref_typeid: mlir_ir.TypeID | None = None


class PtrType(mlir_ir.Type):
    """``!tla.ptr<...>`` as a proper Python subclass of :class:`mlir.ir.Type` (Tla-style)."""

    def __init__(self, cast_from_type: mlir_ir.Type) -> None:
        super().__init__(cast_from_type)

    def __repr__(self) -> str:
        return f"PtrType({str(self)})"

    @classmethod
    def get_static_typeid(cls) -> mlir_ir.TypeID:
        global _cached_ptr_typeid
        if _cached_ptr_typeid is None:
            ctx = mlir_ir.Context()
            with ctx:
                probe = cls.get(mlir_ir.IndexType.get(), "gm", 1, context=ctx)
            _cached_ptr_typeid = probe.typeid
        return _cached_ptr_typeid

    @classmethod
    def isinstance(cls, ty: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_ptr(ty)

    @classmethod
    def try_cast(cls, ty: mlir_ir.Type) -> PtrType | None:
        if not cls.isinstance(ty):
            return None
        return ty if isinstance(ty, cls) else cls(ty)

    @staticmethod
    def get(
        pointee: mlir_ir.Type,
        addrspace: str,
        alignment: int,
        *,
        context: mlir_ir.Context | None = None,
    ) -> PtrType:
        """Build ``!tla.ptr<pointee, tok, align>`` (frontend uses arch tokens, e.g. ``l0c``)."""
        ctx = context if context is not None else pointee.context
        return PtrType(
            _tla_type_bridge.ptr_type_get(ctx, pointee, str(addrspace), int(alignment))
        )

    @property
    def pointee(self) -> mlir_ir.Type:
        return _tla_type_bridge.ptr_pointee_type_get(self.context, self)

    @property
    def addrspace(self) -> str:
        return _tla_type_bridge.ptr_addrspace(self)

    @property
    def alignment(self) -> int:
        return _tla_type_bridge.ptr_alignment(self)

    @property
    def mlir_type(self) -> mlir_ir.Type:
        """Former façade field: the underlying MLIR type is ``self``."""
        return self


class MemrefType(mlir_ir.Type):
    """``!tla.memref<...>`` as a Python subclass of :class:`mlir.ir.Type`."""

    def __init__(self, cast_from_type: mlir_ir.Type) -> None:
        super().__init__(cast_from_type)

    def __repr__(self) -> str:
        return f"MemrefType({str(self)})"

    @classmethod
    def get_static_typeid(cls) -> mlir_ir.TypeID:
        global _cached_memref_typeid
        if _cached_memref_typeid is None:
            ctx = mlir_ir.Context()
            with ctx:
                probe = cls.get((1,), mlir_ir.IndexType.get(), "gm", context=ctx)
            _cached_memref_typeid = probe.typeid
        return _cached_memref_typeid

    @classmethod
    def isinstance(cls, ty: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_memref(ty)

    @classmethod
    def try_cast(cls, ty: mlir_ir.Type) -> "MemrefType | None":
        if not cls.isinstance(ty):
            return None
        return ty if isinstance(ty, cls) else cls(ty)

    @staticmethod
    def get(
        shape: tuple[int | None, ...],
        element_type: mlir_ir.Type,
        addrspace: str,
        *,
        context: mlir_ir.Context | None = None,
    ) -> "MemrefType":
        ctx = context if context is not None else element_type.context
        return MemrefType(
            _tla_type_bridge.memref_type_get(
                ctx, tuple(shape), element_type, str(addrspace)
            )
        )

    @property
    def shape(self) -> tuple[int | None, ...]:
        return _tla_type_bridge.memref_shape(self)

    @property
    def element_type(self) -> mlir_ir.Type:
        return _tla_type_bridge.memref_element_type_get(self.context, self)

    @property
    def addrspace(self) -> str:
        return _tla_type_bridge.memref_addrspace(self)

    @property
    def mlir_type(self) -> mlir_ir.Type:
        return self


class LayoutType(mlir_ir.Type):
    """``!tla.layout<!tla.shape<...>, !tla.stride<...>[, !tla.shape<...>]>`` type wrapper."""

    def __init__(self, cast_from_type: mlir_ir.Type) -> None:
        super().__init__(cast_from_type)

    def __repr__(self) -> str:
        return f"LayoutType({str(self)})"

    @classmethod
    def isinstance(cls, ty: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_layout(ty)

    @classmethod
    def try_cast(cls, ty: mlir_ir.Type) -> "LayoutType | None":
        if not cls.isinstance(ty):
            return None
        return ty if isinstance(ty, cls) else cls(ty)

    @staticmethod
    def get(
        shape_val: mlir_ir.Value,
        stride_val: mlir_ir.Value,
        origin_shape_val: mlir_ir.Value | None = None,
        *,
        layout_tag: str = "row_major",
        context: mlir_ir.Context | None = None,
    ) -> "LayoutType":
        """Construct a structured ``!tla.layout`` from shape/stride SSA values."""
        if not isinstance(shape_val, mlir_ir.Value):
            raise TypeError(
                f"LayoutType.get shape_val expects mlir.ir.Value, got {type(shape_val).__name__}"
            )
        if not isinstance(stride_val, mlir_ir.Value):
            raise TypeError(
                f"LayoutType.get stride_val expects mlir.ir.Value, got {type(stride_val).__name__}"
            )
        if origin_shape_val is not None and not isinstance(
            origin_shape_val, mlir_ir.Value
        ):
            raise TypeError(
                "LayoutType.get origin_shape_val expects mlir.ir.Value or None, "
                f"got {type(origin_shape_val).__name__}"
            )
        ctx = context if context is not None else shape_val.type.context
        tag = str(layout_tag).strip()
        if not tag:
            raise TypeError("LayoutType.get layout_tag must be non-empty")
        if not _tla_type_bridge.type_is_shape(shape_val.type):
            raise TypeError(
                f"LayoutType.get expected !tla.shape operand, got {shape_val.type}"
            )
        if not _tla_type_bridge.type_is_stride(stride_val.type):
            raise TypeError(
                f"LayoutType.get expected !tla.stride operand, got {stride_val.type}"
            )
        if origin_shape_val is not None and not _tla_type_bridge.type_is_shape(
            origin_shape_val.type
        ):
            raise TypeError(
                f"LayoutType.get expected !tla.shape origin operand, got {origin_shape_val.type}"
            )
        return LayoutType(
            _tla_type_bridge.layout_type_from_components_get(
                ctx,
                shape_val.type,
                stride_val.type,
                None if origin_shape_val is None else origin_shape_val.type,
                tag,
            )
        )

    @property
    def mlir_type(self) -> mlir_ir.Type:
        return self


TlaIndexTreeKind = Literal["shape", "coord", "stride"]
TlaIndexTreeLeaf: TypeAlias = int | None
TlaIndexTree: TypeAlias = TlaIndexTreeLeaf | tuple["TlaIndexTree", ...]


def _tla_type_context(context: mlir_ir.Context | None = None) -> mlir_ir.Context:
    if context is not None:
        return context
    try:
        current = mlir_ir.Context.current
    except ValueError:
        return mlir_ir.Context()
    return current if current is not None else mlir_ir.Context()


def _normalize_index_tree_leaf(value: Any) -> int | None:
    if isinstance(value, bool):
        raise TypeError("boolean not allowed in Tla index tree")
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError(
                "Tla index tree leaves must be non-negative integers or None"
            )
        return int(value)
    raise TypeError(
        "Tla index type metadata expects static int leaves or None for dynamic leaves; "
        f"got {type(value).__name__}"
    )


def _normalize_index_tree(tree: Any, *, _tuple_depth: int = 0) -> Any:
    if isinstance(tree, list):
        raise ValueError(
            "expected nested tuple tree for index components (use parentheses, not brackets)"
        )
    if isinstance(tree, tuple):
        if len(tree) == 0:
            raise ValueError("empty tuple in index tree")
        if _tuple_depth >= 2:
            raise ValueError(
                "Tla index tree supports only top-level leaves or one-level leaf groups"
            )
        return tuple(
            _normalize_index_tree(x, _tuple_depth=_tuple_depth + 1) for x in tree
        )
    return _normalize_index_tree_leaf(tree)


def _index_tree_to_asm(tree: Any) -> str:
    if isinstance(tree, tuple):
        return "(" + ",".join(_index_tree_to_asm(x) for x in tree) + ")"
    leaf = _normalize_index_tree_leaf(tree)
    return "?" if leaf is None else str(leaf)


def _index_tree_top_level_to_asm(tree: Any) -> str:
    if isinstance(tree, tuple):
        return ",".join(_index_tree_to_asm(x) for x in tree)
    return _index_tree_to_asm(tree)


def _index_tree_to_metadata(tree: Any, dynamic_values: Iterator[Any]) -> Any:
    if isinstance(tree, tuple):
        return tuple(_index_tree_to_metadata(x, dynamic_values) for x in tree)
    leaf = _normalize_index_tree_leaf(tree)
    if leaf is not None:
        return leaf
    return next(dynamic_values)


@dataclass(frozen=True)
class TlaIndexTreeType:
    """Structured Python descriptor for ``!tla.shape`` / ``!tla.coord`` / ``!tla.stride``."""

    kind: TlaIndexTreeKind
    tree: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree", _normalize_index_tree(self.tree))

    @classmethod
    def from_components(
        cls, kind: TlaIndexTreeKind, components: tuple[Any, ...]
    ) -> "TlaIndexTreeType":
        return cls(kind, tuple(_normalize_index_tree(c) for c in components))

    def body_asm(self) -> str:
        return _index_tree_top_level_to_asm(self.tree)

    def to_asm(self) -> str:
        return str(self.to_mlir_type())

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        if self.kind == "shape":
            return _tla_type_bridge.shape_type_get(ctx, self.tree)
        if self.kind == "coord":
            return _tla_type_bridge.coord_type_get(ctx, self.tree)
        if self.kind == "stride":
            return _tla_type_bridge.stride_type_get(ctx, self.tree)
        raise ValueError(f"unsupported Tla index tree type kind: {self.kind!r}")

    def metadata(self, dynamic_values: Iterable[Any] = ()) -> Any:
        return _index_tree_to_metadata(self.tree, iter(dynamic_values))


@dataclass(frozen=True)
class TlaLayoutDescriptor:
    """Structured Python descriptor for ``!tla.layout``."""

    shape: TlaIndexTreeType
    stride: TlaIndexTreeType
    origin_shape: TlaIndexTreeType | None = None
    layout_tag: str = "row_major"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_tag", str(self.layout_tag).strip())
        if not self.layout_tag:
            raise ValueError("TlaLayoutDescriptor layout_tag must be non-empty")

    def to_asm(self) -> str:
        return str(self.to_mlir_type())

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        origin_tree = self.origin_shape.tree if self.origin_shape is not None else None
        return _tla_type_bridge.layout_type_get(
            ctx, self.shape.tree, self.stride.tree, origin_tree, self.layout_tag
        )


@dataclass(frozen=True)
class TlaTensorTypeDescriptor:
    """Structured Python descriptor for ``!tla.tensor`` metadata.

    The Python frontend uses this as its data model. Text is produced only at the MLIR API
    boundary because this repo currently exposes Tla op bindings, not Tla TypeDef builders.
    """

    layout: TlaLayoutDescriptor
    coord: TlaIndexTree
    element_type: str
    addrspace: str
    ptr_alignment: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "coord", _normalize_index_tree(self.coord))
        object.__setattr__(self, "element_type", str(self.element_type).strip())
        object.__setattr__(self, "addrspace", str(self.addrspace).strip())
        object.__setattr__(self, "ptr_alignment", int(self.ptr_alignment))
        if not self.element_type:
            raise ValueError("TlaTensorTypeDescriptor element_type must be non-empty")
        if not self.addrspace:
            raise ValueError("TlaTensorTypeDescriptor addrspace must be non-empty")
        if self.ptr_alignment < 1:
            raise ValueError("TlaTensorTypeDescriptor ptr_alignment must be positive")

    @property
    def shape(self) -> TlaIndexTree:
        return self.layout.shape.tree

    @property
    def stride(self) -> TlaIndexTree:
        return self.layout.stride.tree

    @property
    def origin_shape(self) -> TlaIndexTree:
        if self.layout.origin_shape is None:
            raise ValueError("Tla tensor layout must carry an origin shape")
        return self.layout.origin_shape.tree

    @property
    def layout_tag(self) -> str:
        return self.layout.layout_tag

    @property
    def ptr_asm(self) -> str:
        ctx = _tla_type_context()
        return str(
            PtrType.get(self.element_mlir_type(ctx), self.addrspace, self.ptr_alignment)
        )

    def element_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        with ctx:
            return _elem_token_to_mlir_type(self.element_type)

    def to_asm(self) -> str:
        return str(self.to_mlir_type())

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        return _tla_type_bridge.tensor_type_get(
            ctx,
            shape=self.shape,
            stride=self.stride,
            coord=self.coord,
            origin_shape=self.origin_shape,
            element_type=self.element_mlir_type(ctx),
            addrspace=self.addrspace,
            layout=self.layout_tag,
            ptr_alignment=self.ptr_alignment,
        )

    def with_updates(self, **updates: Any) -> "TlaTensorTypeDescriptor":
        layout_updates: dict[str, Any] = {}
        for source_key, target_key, kind in (
            ("shape", "shape", "shape"),
            ("stride", "stride", "stride"),
            ("origin_shape", "origin_shape", "shape"),
        ):
            if source_key in updates:
                layout_updates[target_key] = TlaIndexTreeType(
                    kind, updates.pop(source_key)
                )
        if "layout_tag" in updates:
            layout_updates["layout_tag"] = updates.pop("layout_tag")
        layout = self.layout
        if layout_updates:
            layout_values = {
                "shape": layout.shape,
                "stride": layout.stride,
                "origin_shape": layout.origin_shape,
                "layout_tag": layout.layout_tag,
            }
            layout_values.update(layout_updates)
            layout = TlaLayoutDescriptor(**layout_values)
        values = {
            "layout": layout,
            "coord": self.coord,
            "element_type": self.element_type,
            "addrspace": self.addrspace,
            "ptr_alignment": self.ptr_alignment,
        }
        values.update(updates)
        return TlaTensorTypeDescriptor(**values)

    def metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "stride": self.stride,
            "coord": self.coord,
            "origin_shape": self.origin_shape,
            "dtype": self.element_type,
            "addrspace": self.addrspace,
            "layout_tag": self.layout_tag,
        }


class TlaValue:
    """Marker annotation for Tla register values."""


class TlaTensor:
    """Marker annotation for Tla tensor/view values."""


TlaTile = TlaTensor


class TlaAllocPtr:
    """Marker annotation for raw allocator-backed byte buffers."""


class TlaFlag:
    """Marker annotation for Tla flag values."""


class TlaCrossFlag:
    """Marker annotation for Tla cross-flag values."""


class TlaIndex:
    """Marker annotation for index values."""


class TlaShape:
    """Marker for ``make_shape`` / ``!tla.shape<...>``; ``tla.tile_view`` / ``broadcast`` take this value."""


class TlaCoord:
    """Marker for ``make_coord`` / ``!tla.coord<...>``; ``tla.tile_view`` takes this value."""


class TlaStride:
    """Marker annotation for ``make_stride`` / ``!tla.stride<...>``."""


class TlaLayout:
    """Marker for ``make_layout(shape, stride, *, origin_shape=...)`` / ``!tla.layout<!tla.shape<…>, !tla.stride<…>, !tla.shape<…>, row_major>``."""


class TlaRegion:
    """Marker annotation for Tla region stubs."""


_ANNOTATION_CATEGORY = {
    TlaValue: "value",
    TlaTensor: "tensor",
    TlaAllocPtr: "alloc_ptr",
    TlaFlag: "flag",
    TlaCrossFlag: "cross_flag",
    TlaIndex: "index",
    TlaShape: "shape",
    TlaCoord: "coord",
    TlaStride: "stride",
    TlaLayout: "layout",
    TlaRegion: "region",
    PointerABC: "pointer",
}


def annotation_to_category(annotation: Any) -> str | None:
    return _ANNOTATION_CATEGORY.get(annotation)


def _slice_indices(index: slice, length: int) -> list[int]:
    return list(range(*index.indices(length)))


def _coerce_nested_tensor_value(value: Any) -> Any:
    # Keep numpy arrays as-is to avoid expensive tolist() conversion
    np = _require_numpy()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, np.ndarray)):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value
    return value


def _enumerate_leaf_paths_with_prefix(
    shape_comp: Any, prefix: list[int]
) -> list[list[int]]:
    dims = _leaf_dims(shape_comp)
    if len(prefix) > len(dims):
        return []
    tail_dims = dims[len(prefix) :]
    if not tail_dims:
        return [list(prefix)]
    tail_lists = [list(t) for t in product(*[range(d) for d in tail_dims])]
    return [prefix + tl for tl in tail_lists]


def _path_to_flat_index(path: list[int], dims: tuple[int, ...]) -> int:
    flat_idx = 0
    multiplier = 1
    for idx, dim in zip(reversed(path), reversed(dims)):
        flat_idx += idx * multiplier
        multiplier *= dim
    return flat_idx


def _assign_tensor_data_view(
    view: list[Any] | _TensorDataList,
    value: Any,
    *,
    layout_tag: str | None = None,
) -> None:
    del layout_tag
    coerced = _coerce_nested_tensor_value(value)
    if isinstance(view, _TensorDataList):
        np = _require_numpy()
        arr = np.asarray(coerced)
        dims_tail = _leaf_dims(view._shape_comp)[len(view._path_leaves) :]
        expected = int(np.prod(dims_tail, dtype=np.int64)) if dims_tail else 1
        if arr.size != expected:
            raise ValueError(
                f"tensor data assignment expected {expected} value(s), got {arr.size}"
            )
        expected_shape = tuple(dims_tail)
        if expected_shape and tuple(arr.shape) != expected_shape:
            raise ValueError(
                "tensor data assignment expected shape "
                f"{expected_shape}, got {tuple(arr.shape)}"
            )
        if not view._path_leaves:
            view._storage.clear()
        flat = arr.ravel(order="C")
        # Use flat index directly for much faster bulk storage
        total_dims = view._dims
        prefix_len = len(view._path_leaves)
        if prefix_len == 0:
            # Fast path: root level, direct sequential storage
            for i, v in enumerate(flat):
                view._storage[i] = v.item() if hasattr(v, "item") else v
        else:
            # Nested path: compute base index and use multipliers
            base_idx = _path_to_flat_index(view._path_leaves, total_dims)
            # Pre-compute multipliers for remaining dimensions
            remaining_dims = total_dims[prefix_len:]
            multipliers = [1]
            for d in reversed(remaining_dims[1:]):
                multipliers.insert(0, multipliers[0] * d)
            # Assign values with computed flat indices
            for i, v in enumerate(flat):
                # Calculate offset within the remaining dimensions
                offset = 0
                temp_i = i
                for dim, mult in zip(remaining_dims, multipliers):
                    offset += (temp_i % dim) * mult
                    temp_i //= dim
                flat_idx = base_idx + offset
                view._storage[flat_idx] = v.item() if hasattr(v, "item") else v
        if view._on_mutate is not None:
            view._on_mutate()
        return
    if not isinstance(coerced, (list, tuple)):
        raise TypeError(
            "tensor data assignment requires a list, tuple, or numpy-like value"
        )
    if len(view) != len(coerced):
        raise ValueError(
            f"tensor data assignment expected {len(view)} values, got {len(coerced)}"
        )
    view[:] = list(coerced)


def _set_tensor_storage_value(storage: dict[Any, Any], key: Any, value: Any) -> None:
    if value is None:
        storage.pop(key, None)
    else:
        storage[key] = value


def _pack_nested_key_from_leaf_indices(shape_comp: Any, leaf_idx: list[int]) -> Any:
    """Build a storage key isomorphic to ``shape_comp`` (nested tuple tree of indices)."""
    it = iter(leaf_idx)

    def walk(node: Any) -> Any:
        if isinstance(node, int):
            return next(it)
        if not isinstance(node, tuple) or not node:
            raise TypeError("invalid shape component tree")
        if all(isinstance(x, int) for x in node):
            return tuple(next(it) for _ in node)
        return tuple(walk(x) for x in node)

    return walk(shape_comp)


def _leaf_dims(shape_comp: Any) -> tuple[int, ...]:
    return tuple(_flatten_int_leaves_tree(shape_comp))


def _tensor_data_view(
    storage: dict[Any, Any],
    shape_comp: Any | None,
    *,
    on_mutate: Any | None = None,
    path_leaves: list[int] | None = None,
) -> list[Any] | _TensorDataList | None:
    if shape_comp is None:
        return None
    if shape_comp == ():
        return []
    return _TensorDataList(
        storage, shape_comp, path_leaves=list(path_leaves or []), on_mutate=on_mutate
    )


class _TensorDataList:
    """Host-side view over ``dict`` storage with keys matching the shape component tree."""

    def __init__(
        self,
        storage: dict[Any, Any],
        shape_comp: Any,
        *,
        path_leaves: list[int],
        on_mutate: Any | None = None,
    ) -> None:
        self._storage = storage
        self._shape_comp = shape_comp
        self._path_leaves = path_leaves
        self._on_mutate = on_mutate
        self._dims = _leaf_dims(shape_comp)

    def __len__(self) -> int:
        if len(self._path_leaves) >= len(self._dims):
            return 0
        return self._dims[len(self._path_leaves)]

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in _slice_indices(index, len(self))]
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("tensor data index out of range")
        new_path = self._path_leaves + [normalized]
        if len(new_path) == len(self._dims):
            key = _path_to_flat_index(new_path, self._dims)
            return self._storage.get(key)
        return _TensorDataList(
            self._storage,
            self._shape_comp,
            path_leaves=new_path,
            on_mutate=self._on_mutate,
        )

    def __repr__(self) -> str:
        total = 1
        for dim in self._dims:
            total *= dim
        if total <= 64:
            return repr(list(self))
        return f"TensorData(shape_comp={self._shape_comp!r})"

    def __setitem__(self, index: int | slice, value: Any) -> None:
        if isinstance(index, slice):
            indices = _slice_indices(index, len(self))
            coerced = _coerce_nested_tensor_value(value)
            values = list(coerced)
            if len(indices) != len(values):
                raise ValueError(
                    f"attempt to assign sequence of size {len(values)} "
                    f"to extended slice of size {len(indices)}"
                )
            for offset, item in zip(indices, values):
                self[offset] = item
            return
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("tensor data index out of range")
        new_path = self._path_leaves + [normalized]
        if len(new_path) == len(self._dims):
            key = _path_to_flat_index(new_path, self._dims)
            _set_tensor_storage_value(self._storage, key, value)
            if self._on_mutate is not None:
                self._on_mutate()
            return
        coerced = _coerce_nested_tensor_value(value)
        if not isinstance(coerced, (list, tuple)):
            raise TypeError(
                "nested tensor data assignment requires a list or tuple value"
            )
        child = _TensorDataList(
            self._storage,
            self._shape_comp,
            path_leaves=new_path,
            on_mutate=self._on_mutate,
        )
        child[:] = coerced

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("tensor data has fixed dimensions")

    def append(self, value: Any) -> None:
        raise TypeError("tensor data has fixed dimensions")

    def extend(self, values: Iterable[Any]) -> None:
        raise TypeError("tensor data has fixed dimensions")

    def insert(self, index: int, value: Any) -> None:
        raise TypeError("tensor data has fixed dimensions")

    def pop(self, index: int = -1) -> None:
        raise TypeError("tensor data has fixed dimensions")

    def clear(self) -> None:
        self[:] = [None] * len(self)


_TENSOR_DTYPE_TO_NUMPY: dict[str, str] = {
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
}

_TENSOR_DTYPE_SIZES: dict[str, int] = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}


def dtype_size_bytes(dtype: str) -> int:
    """Storage size in bytes for a Tla element type token (e.g. ``f16``, ``i32``).

    Spelling is normalized with :func:`str.strip` / :func:`str.lower`. Unknown names
    return ``0`` (same table as :class:`Tensor` host storage sizing).
    """
    return int(_TENSOR_DTYPE_SIZES.get(dtype.strip().lower(), 0))


def _tensor_storage_to_nested_list(storage: dict[Any, Any], shape_comp: Any) -> Any:
    """Nested Python lists aligned with leaf axes; structure follows ``shape_comp`` grouping."""
    dims = _leaf_dims(shape_comp)
    if not dims:
        return storage.get(_path_to_flat_index([], dims))

    def build(path: list[int]) -> Any:
        if len(path) == len(dims):
            return storage.get(_path_to_flat_index(path, dims))
        return [build(path + [i]) for i in range(dims[len(path)])]

    return build([])


def _load_acl() -> Any:
    try:
        return importlib.import_module("acl")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import `acl`. Ensure the Ascend runtime is installed."
        ) from exc


def _require_acl_success(ret: Any, op_name: str) -> None:
    if int(ret) != 0:
        raise RuntimeError(f"{op_name} failed with ret={int(ret)}")


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for Tensor host/device transfers."
        ) from exc
    return np


def _register_runtime_pointer(kind: str, ptr: int) -> None:
    if ptr == 0:
        return
    try:
        from . import runtime as runtime_mod
    except Exception:
        return
    try:
        if kind == "device":
            runtime_mod.register_device_ptr(int(ptr))
        elif kind == "host":
            runtime_mod.register_host_ptr(int(ptr))
    except Exception:
        # Tensor allocation can still be used even if the global runtime registry
        # is unavailable or uninitialized.
        return


def _has_tensor_host_data(storage: dict[Any, Any], shape_comp: Any) -> bool:
    dims = _leaf_dims(shape_comp)
    # Check all flat indices from 0 to product(dims)-1
    total = 1
    for d in dims:
        total *= d
    for i in range(total):
        if i not in storage or storage[i] is None:
            return False
    return True


_LIVE_TENSORS: dict[int, weakref.ReferenceType[Any]] = {}


def _track_live_tensor(tensor: Any) -> None:
    tensor_id = id(tensor)

    def _cleanup(_ref: weakref.ReferenceType[Any]) -> None:
        _LIVE_TENSORS.pop(tensor_id, None)

    _LIVE_TENSORS[tensor_id] = weakref.ref(tensor, _cleanup)


def invalidate_runtime_allocations(
    *,
    device_ptrs: Iterable[int] = (),
    host_ptrs: Iterable[int] = (),
) -> None:
    freed_device_ptrs = {int(ptr) for ptr in device_ptrs if int(ptr) != 0}
    freed_host_ptrs = {int(ptr) for ptr in host_ptrs if int(ptr) != 0}
    if not freed_device_ptrs and not freed_host_ptrs:
        return
    for tensor_ref in list(_LIVE_TENSORS.values()):
        tensor = tensor_ref()
        if tensor is None:
            continue
        if tensor.data_ptr in freed_device_ptrs:
            if tensor.stale:
                tensor._device_data_lost = True
            tensor.data_ptr = 0
            tensor.stale = False
        if tensor.host_ptr in freed_host_ptrs:
            tensor.host_ptr = 0


def _flatten_int_leaves_tree(tree: Any) -> list[int]:
    """Preorder flatten of positive-int leaves (same leaf order as :func:`catlass.core_api._flatten_tla_tuple`)."""
    if isinstance(tree, (tuple, list)):
        out: list[int] = []
        for x in tree:
            out.extend(_flatten_int_leaves_tree(x))
        return out
    if isinstance(tree, int):
        return [tree]
    raise TypeError(f"tensor shape tree expects int leaves, got {type(tree).__name__}")


def _tree_structure_mask(tree: Any) -> Any:
    """Shape-only mask: nested tuples match; leaves are ``None`` placeholders."""
    if isinstance(tree, (tuple, list)):
        return tuple(_tree_structure_mask(x) for x in tree)
    return None


def _try_remap_stride_coord_trees(
    comp: tuple[Any, ...],
    origin_components: tuple[Any, ...],
    dtype: str,
    layout_tok: str,
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    """Use :func:`catlass.core_api._remap_tensor_like_prefix_fields_for_layout_trees` for defaults.

    Returns ``(stride_tree, coord_tree)`` with ``None`` entries when remap is unavailable or
    when the remap stride tree does not match ``comp``'s nesting. The caller must raise if a
    ``None`` entry was needed as a default.
    """
    from .core_api import _remap_tensor_like_prefix_fields_for_layout_trees

    trees = _remap_tensor_like_prefix_fields_for_layout_trees(
        origin_components, dtype.strip().lower(), layout_tok.strip()
    )
    if trees is None:
        return (None, None)
    _, stride_tree, coord_tree, _ = trees
    stride_ok = _tree_structure_mask(stride_tree) == _tree_structure_mask(comp)
    return (stride_tree if stride_ok else None, coord_tree)


@dataclass
class Tensor:
    """Tensor annotation that lowers to a first-class ``!tla.tensor`` type.

    Non-symbolic tensors must use :func:`catlass.core_api.make_shape` for ``shape`` and
    for ``origin_shape`` (required), :func:`catlass.core_api.make_coord` for ``coord``, and
    :func:`catlass.core_api.make_stride` for ``stride``.
    When ``coord`` / ``stride`` are omitted, defaults come only from
    :func:`catlass.core_api._remap_tensor_like_prefix_fields_for_layout_trees` (requires a flat
    logical ``origin_shape`` tree ``M,N`` and a stride tree matching
    ``shape``); otherwise construction raises ``ValueError``—pass ``tla.make_coord`` /
    ``tla.make_stride`` explicitly.
    These ops must run under an active frontend session (e.g. :func:`catlass.runtime._eager_capture`).

    ``dtype`` must be a compile-time element class such as ``tla.Float16`` (not a string token).
    ``addrspace`` must be :class:`~catlass.address_space.AddressSpace` (e.g. ``tla.AddressSpace.gm``,
    ``tla.AddressSpace.l0a``); the sixth field of ``!tla.tensor<…>`` uses the same keyword
    (``gm``, ``l0a``, ``l1``, …).

    ``layout_tag`` must be a ``tla.arch`` layout sentinel (e.g. ``tla.arch.RowMajor``); omit it
    to default to row-major. Raw strings are not accepted.

    Tensor shape metadata is carried by Tla shape/coord/stride values, not Python
    strings.
    """

    shape: Any
    dtype: Any
    addrspace: Any = AddressSpace.gm
    data_ptr: int = 0
    host_ptr: int = 0
    stale: bool = False
    origin_shape: Iterable[Any] | None = None
    coord: Iterable[Any] | None = None
    stride: Any | None = None
    layout_tag: Any | None = None
    _shape_components: tuple[Any, ...] | None = field(
        default=None, repr=False, compare=False
    )
    _shape_tuple: tuple[int, ...] | None = field(
        default=None, repr=False, compare=False
    )
    _data_storage: dict[Any, Any] | None = field(
        default=None, repr=False, compare=False
    )
    _device_data_lost: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from .core_api import _Coord, _Shape, _Stride, _resolve_arch_layout_tag

        self.dtype = _coerce_host_tensor_dtype(self.dtype)
        self.addrspace = _coerce_host_tensor_addrspace(self.addrspace)
        self.layout_tag = _resolve_arch_layout_tag(self.layout_tag, for_op="Tensor")

        if not isinstance(self.shape, _Shape):
            raise TypeError(
                "Tensor(shape=...): shape must be the result of tla.make_shape(...)"
            )

        comp = self.shape._components
        self.shape = comp
        flat = tuple(_flatten_int_leaves_tree(comp))
        self._shape_tuple = flat
        self._shape_components = comp

        if self.origin_shape is None:
            raise TypeError(
                "Tensor(shape=tla.make_shape(...)): origin_shape is required; "
                "pass tla.make_shape(...) for logical bounds"
            )
        if isinstance(self.origin_shape, _Shape):
            self.origin_shape = self.origin_shape._components
        else:
            raise TypeError(
                "Tensor origin_shape=... must be the result of tla.make_shape(...)"
            )

        rs_stride, rs_coord = _try_remap_stride_coord_trees(
            comp, self.origin_shape, self.dtype, self.layout_tag
        )

        if self.coord is None:
            if rs_coord is None:
                raise ValueError(
                    "Tensor(coord=None): cannot derive coord from layout remap; "
                    "use a flat logical origin_shape (M,N) without nested parentheses, "
                    "or pass tla.make_coord(...)"
                )
            self.coord = rs_coord
        elif isinstance(self.coord, _Coord):
            self.coord = self.coord._components
        else:
            raise TypeError(
                "Tensor coord=... must be None or the result of tla.make_coord(...)"
            )

        if self.stride is None:
            if rs_stride is None:
                raise ValueError(
                    "Tensor(stride=None): cannot derive stride from layout remap "
                    "(remap stride tree must match shape tree); pass tla.make_stride(...)"
                )
            self.stride = rs_stride
        elif isinstance(self.stride, _Stride):
            sc = self.stride._components
            if _tree_structure_mask(sc) != _tree_structure_mask(comp):
                raise ValueError(
                    "Tensor stride component tree must match shape tree structure"
                )
            self.stride = sc
        else:
            raise TypeError(
                "Tensor stride=... must be None or the result of tla.make_stride(...)"
            )

        if self._data_storage is None:
            self._data_storage = {}
        _track_live_tensor(self)

    def _mark_host_data_authoritative(self) -> None:
        self._device_data_lost = False
        self.stale = False

    def _require_device_data_available(self) -> None:
        if self._device_data_lost:
            raise RuntimeError(
                "Tensor device-resident data was freed during tla.finalize(); "
                "repopulate host data before reuse."
            )

    def tla_tensor_type_descriptor(self) -> TlaTensorTypeDescriptor:
        """Structured ``!tla.tensor`` descriptor from host metadata."""
        st = self._shape_tuple
        addr_kw = (self.addrspace or "gm").strip()
        if st is None or self.stride is None or self.layout_tag is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        assert self._shape_components is not None
        return TlaTensorTypeDescriptor(
            layout=TlaLayoutDescriptor(
                shape=TlaIndexTreeType("shape", self._shape_components),
                stride=TlaIndexTreeType("stride", self.stride),
                origin_shape=TlaIndexTreeType("shape", self.origin_shape),
                layout_tag=str(self.layout_tag),
            ),
            coord=self.coord,
            element_type=str(self.dtype),
            addrspace=addr_kw,
            ptr_alignment=max(1, dtype_size_bytes(str(self.dtype))),
        )

    def __tla_type__(self) -> str:
        return str(self.tla_tensor_type_descriptor().to_mlir_type())

    def __c_pointers__(self) -> list[int]:
        return [int(self.data_ptr)]

    def __get_mlir_types__(
        self, context: mlir_ir.Context | None = None
    ) -> list[mlir_ir.Type]:
        return [self.tla_tensor_type_descriptor().to_mlir_type(context)]

    def __new_from_mlir_values__(self, values: list[Any]) -> "Tensor":
        del values
        return self

    @property
    def size_bytes(self) -> int:
        if self._shape_tuple is None:
            raise TypeError(
                "Tensor size is unavailable without concrete shape metadata."
            )
        if self.dtype not in _TENSOR_DTYPE_SIZES:
            raise ValueError(
                f"Unsupported tensor dtype for upload_data(): {self.dtype}"
            )
        elements = 1
        for dim in self._shape_tuple:
            elements *= dim
        return elements * _TENSOR_DTYPE_SIZES[self.dtype]

    def tobytes(self) -> bytes:
        self._require_device_data_available()
        if self._shape_tuple is None:
            raise TypeError(
                "Tensor bytes are unavailable without concrete shape metadata."
            )
        if self.dtype not in _TENSOR_DTYPE_TO_NUMPY:
            raise ValueError(f"Unsupported tensor dtype for tobytes(): {self.dtype}")
        np = _require_numpy()
        storage = self._data_storage
        if storage is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        assert self._shape_components is not None
        nested = _tensor_storage_to_nested_list(storage, self._shape_components)
        array = np.asarray(nested, dtype=_TENSOR_DTYPE_TO_NUMPY[self.dtype])
        return bytes(array.tobytes())

    def download_data(self) -> None:
        if self.data_ptr == 0:
            raise RuntimeError(
                "Tensor download_data() requires a prior upload_data() call."
            )
        if self._shape_tuple is None:
            raise TypeError(
                "Tensor download_data() is unavailable without concrete shape metadata."
            )
        if self.dtype not in _TENSOR_DTYPE_TO_NUMPY:
            raise ValueError(
                f"Unsupported tensor dtype for download_data(): {self.dtype}"
            )
        np = _require_numpy()
        acl = _load_acl()
        size_bytes = self.size_bytes
        if self.host_ptr == 0:
            host_ptr, ret = acl.rt.malloc_host(size_bytes)
            _require_acl_success(ret, "acl.rt.malloc_host")
            self.host_ptr = int(host_ptr)
            _register_runtime_pointer("host", self.host_ptr)
        ret = acl.rt.memcpy(self.host_ptr, size_bytes, self.data_ptr, size_bytes, 2)
        _require_acl_success(ret, "acl.rt.memcpy")
        bytes_ptr = acl.util.ptr_to_bytes(self.host_ptr, size_bytes)
        flat = np.frombuffer(bytes_ptr, dtype=_TENSOR_DTYPE_TO_NUMPY[self.dtype])
        assert self._shape_components is not None
        reshaped = flat.reshape(self._shape_tuple)
        self._device_data_lost = False
        self.stale = False
        self.data = reshaped

    def upload_data(self) -> None:
        self._require_device_data_available()
        acl = _load_acl()
        size_bytes = self.size_bytes
        if self.data_ptr == 0:
            dev_ptr, ret = acl.rt.malloc(size_bytes, 0)
            _require_acl_success(ret, "acl.rt.malloc")
            self.data_ptr = int(dev_ptr)
            _register_runtime_pointer("device", self.data_ptr)
        bytes_vector = self.tobytes()
        bytes_ptr = acl.util.bytes_to_ptr(bytes_vector)
        ret = acl.rt.memcpy(self.data_ptr, size_bytes, bytes_ptr, size_bytes, 1)
        _require_acl_success(ret, "acl.rt.memcpy")
        self._device_data_lost = False
        self.stale = True

    def prepare_for_launch(self) -> None:
        self._require_device_data_available()
        if self._shape_tuple is None:
            return
        if self.data_ptr == 0:
            storage = self._data_storage
            if storage is None:
                raise TypeError(
                    "Tensor data is unavailable without concrete shape metadata."
                )
            assert self._shape_components is not None
            if _has_tensor_host_data(storage, self._shape_components):
                self.upload_data()
                return
            acl = _load_acl()
            dev_ptr, ret = acl.rt.malloc(self.size_bytes, 0)
            _require_acl_success(ret, "acl.rt.malloc")
            self.data_ptr = int(dev_ptr)
            _register_runtime_pointer("device", self.data_ptr)
            self.stale = True
            return
        if not self.stale:
            self.upload_data()

    def _ensure_fresh_data(self) -> None:
        self._require_device_data_available()
        if self.stale:
            self.download_data()

    @property
    def data(self) -> list[Any] | _TensorDataList | None:
        self._ensure_fresh_data()
        storage = self._data_storage
        if storage is None:
            return None
        return _tensor_data_view(
            storage,
            self._shape_components,
            on_mutate=self._mark_host_data_authoritative,
        )

    @data.setter
    def data(self, value: Any) -> None:
        if not self._device_data_lost:
            self._ensure_fresh_data()
        storage = self._data_storage
        if storage is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        view = _tensor_data_view(
            storage,
            self._shape_components,
            on_mutate=self._mark_host_data_authoritative,
        )
        if view is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        _assign_tensor_data_view(view, value, layout_tag=self.layout_tag)
        self._mark_host_data_authoritative()

    def __len__(self) -> int:
        data = self.data
        if data is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        return len(data)

    def __iter__(self) -> Iterator[Any]:
        data = self.data
        if data is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        return iter(data)

    def __getitem__(self, index: int | slice) -> Any:
        data = self.data
        if data is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        return data[index]

    def __setitem__(self, index: int | slice, value: Any) -> None:
        data = self.data
        if data is None:
            raise TypeError(
                "Tensor data is unavailable without concrete shape metadata."
            )
        data[index] = value

    def __str__(self) -> str:
        return self.__tla_type__()

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"addrspace={self.addrspace!r}, data_ptr={self.data_ptr!r}, stale={self.stale!r}, "
            f"origin_shape={self.origin_shape!r}, coord={self.coord!r}, "
            f"stride={self.stride!r}, layout_tag={self.layout_tag!r})"
        )


@dataclass(frozen=True)
class Scalar:
    """Typed scalar wrapper for runtime argument passing."""

    value: Any
    dtype: str

    def __tla_type__(self) -> str:
        return self.dtype

    def __get_mlir_types__(
        self, context: mlir_ir.Context | None = None
    ) -> list[mlir_ir.Type]:
        with _tla_type_context(context):
            return [_elem_token_to_mlir_type(self.dtype)]

    def __c_pointers__(self) -> list[int]:
        dtype = self.dtype.lower()
        if dtype == "i1":
            packed = 1 if bool(self.value) else 0
            return [packed]
        if dtype == "i8":
            packed = ctypes.c_int8(int(self.value)).value & 0xFF
            return [packed]
        if dtype == "i16":
            packed = ctypes.c_int16(int(self.value)).value & 0xFFFF
            return [packed]
        if dtype == "i32":
            packed = ctypes.c_int32(int(self.value)).value & 0xFFFFFFFF
            return [packed]
        if dtype == "i64":
            packed = ctypes.c_int64(int(self.value)).value & 0xFFFFFFFFFFFFFFFF
            return [packed]
        if dtype == "u8":
            packed = int(self.value) & 0xFF
            return [packed]
        if dtype == "u16":
            packed = int(self.value) & 0xFFFF
            return [packed]
        if dtype == "u32":
            packed = int(self.value) & 0xFFFFFFFF
            return [packed]
        if dtype == "u64":
            packed = int(self.value) & 0xFFFFFFFFFFFFFFFF
            return [packed]
        if dtype == "f32":
            packed = struct.unpack("I", struct.pack("f", float(self.value)))[0]
            return [packed]
        if dtype == "f16":
            return [_float16_bits(float(self.value))]
        if dtype == "bf16":
            return [_bfloat16_bits(float(self.value))]
        if dtype == "f64":
            packed = struct.unpack("Q", struct.pack("d", float(self.value)))[0]
            return [packed]
        if dtype == "index":
            packed = ctypes.c_int64(int(self.value)).value & 0xFFFFFFFFFFFFFFFF
            return [packed]
        raise ValueError(f"Unsupported scalar dtype: {self.dtype}")

    def __new_from_mlir_values__(self, values: list[Any]) -> "Scalar":
        del values
        return self


def _float16_bits(value: float) -> int:
    f32 = struct.unpack("I", struct.pack("f", value))[0]
    sign = (f32 >> 31) & 0x1
    exp = (f32 >> 23) & 0xFF
    frac = f32 & 0x7FFFFF

    if exp == 0xFF:
        # Inf/NaN
        half_exp = 0x1F
        half_frac = 0x200 if frac != 0 else 0
    else:
        exp = exp - 127
        if exp > 15:
            # Overflow to Inf
            half_exp = 0x1F
            half_frac = 0
        elif exp < -14:
            # Subnormal or zero
            if exp < -24:
                half_exp = 0
                half_frac = 0
            else:
                shift = (-exp) - 14
                mant = (1 << 23) | frac
                half_frac = mant >> (13 + shift)
                # round to nearest even
                remainder = mant & ((1 << (13 + shift)) - 1)
                halfway = 1 << (12 + shift)
                if remainder > halfway or (remainder == halfway and (half_frac & 0x1)):
                    half_frac += 1
                half_exp = 0
        else:
            half_exp = exp + 15
            half_frac = frac >> 13
            remainder = frac & 0x1FFF
            if remainder > 0x1000 or (remainder == 0x1000 and (half_frac & 0x1)):
                half_frac += 1
                if half_frac == 0x400:
                    half_frac = 0
                    half_exp += 1
                    if half_exp >= 0x1F:
                        half_exp = 0x1F
                        half_frac = 0

    return (sign << 15) | ((half_exp & 0x1F) << 10) | (half_frac & 0x3FF)


def _bfloat16_bits(value: float) -> int:
    f32 = struct.unpack("I", struct.pack("f", value))[0]
    upper = f32 >> 16
    lower = f32 & 0xFFFF
    # round to nearest even
    if lower > 0x8000 or (lower == 0x8000 and (upper & 0x1)):
        upper = (upper + 1) & 0xFFFF
    return upper


from .base_dsl.typing import (
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Index,
    Float16,
    BFloat16,
    Float32,
    Float64,
)


__all__ = [
    "AddressSpace",
    "PtrType",
    "MemrefType",
    "TlaIndexTreeKind",
    "TlaIndexTreeLeaf",
    "TlaIndexTree",
    "TlaIndexTreeType",
    "TlaLayoutDescriptor",
    "TlaTensorTypeDescriptor",
    "TlaValue",
    "TlaTensor",
    "TlaTile",
    "TlaFlag",
    "TlaCrossFlag",
    "TlaIndex",
    "TlaShape",
    "TlaCoord",
    "TlaStride",
    "TlaLayout",
    "TlaRegion",
    "Tensor",
    "Scalar",
    "Numeric",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Index",
    "Float32",
    "Float64",
    "Float16",
    "BFloat16",
    "annotation_to_category",
    "dtype_size_bytes",
]
