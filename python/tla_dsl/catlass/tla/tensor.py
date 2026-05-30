"""Tla tensor IR values and view helpers."""

from __future__ import annotations

from typing import Any

from mlir import ir as mlir_ir  # type: ignore[assignment]

from .. import _tla_type_bridge
from .. import runtime as _runtime
from .typing import Tensor as TensorABC


class _Tensor(TensorABC):
    """Frontend proxy for an SSA ``!tla.tensor`` value."""

    def __init__(self, value: mlir_ir.Value) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Tensor value expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not _tla_type_bridge.type_is_tensor(value.type):
            raise TypeError(f"Tensor value expects !tla.tensor<...>, got {value.type}")
        self.value = value
        self.__tla_category__ = "tensor"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "tensor")
        _runtime._bind_frontend_category(value, "tensor")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    @property
    def shape(self) -> Any:
        return _tensor_metadata_field(self.value, "shape")

    @property
    def stride(self) -> Any:
        return _tensor_metadata_field(self.value, "stride")

    @property
    def coord(self) -> Any:
        return _tensor_metadata_field(self.value, "coord")

    @property
    def origin_shape(self) -> Any:
        return _tensor_metadata_field(self.value, "origin_shape")

    @property
    def dtype(self) -> str:
        return str(_tensor_metadata_field(self.value, "dtype"))

    @property
    def addrspace(self) -> str:
        return str(_tensor_metadata_field(self.value, "addrspace"))

    @property
    def layout_tag(self) -> str:
        return str(_tensor_metadata_field(self.value, "layout_tag"))


def _tensor_metadata_field(value: mlir_ir.Value, field: str) -> Any:
    from ..core_api import _tensor_metadata_field as _lookup

    return _lookup(value, field)


def _scale_coord_leaf(coord: Any, shape: Any) -> Any:
    if isinstance(coord, int):
        return coord * shape
    resolved = _runtime._resolve_frontend_bound_value(coord)
    if (
        resolved is not coord
        or _runtime._resolve_frontend_bound_category(coord) == "index"
    ):
        return _runtime._IndexExpr(_runtime._coerce_index_value(coord)) * shape
    return coord * shape


def scale_tile_coord_by_shape(coord_tree: Any, shape_tree: Any) -> Any:
    """Convert tile coordinates into element offsets using the tile shape."""

    if isinstance(coord_tree, tuple) and isinstance(shape_tree, tuple):
        if len(coord_tree) != len(shape_tree):
            raise ValueError(
                "tile-view coord/shape trees must have matching tuple profiles"
            )
        return tuple(
            scale_tile_coord_by_shape(coord_part, shape_part)
            for coord_part, shape_part in zip(coord_tree, shape_tree)
        )
    if isinstance(coord_tree, tuple) or isinstance(shape_tree, tuple):
        raise ValueError(
            "tile-view coord/shape trees must have matching tuple profiles"
        )
    return _scale_coord_leaf(coord_tree, shape_tree)


def normalize_tile_view_coord(
    *,
    shape_components: tuple[Any, ...],
    coord_components: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Convert ``tla.tile_view`` tile coordinates into element offsets."""

    if len(coord_components) != len(shape_components):
        raise ValueError("tile-view coord/shape ranks must match")
    return tuple(
        scale_tile_coord_by_shape(coord_part, shape_part)
        for coord_part, shape_part in zip(coord_components, shape_components)
    )


__all__ = [
    "_Tensor",
    "normalize_tile_view_coord",
    "scale_tile_coord_by_shape",
]
