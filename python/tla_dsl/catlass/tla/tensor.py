"""Tla tensor IR values and view helpers."""

from __future__ import annotations

from typing import Any

from mlir import ir as mlir_ir  # type: ignore[assignment]

from .. import _tla_type_bridge
from .._mlir_bindings import tla_ops_gen as _tla_ops_gen
from ..base_dsl.op import dsl_user_op
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

    @dsl_user_op
    def load(
        self,
        params: Any | None = None,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Load this tensor tile into vector SSA inside a tla.vec.func region."""
        from ..core_api import (
            VectorSSA,
            _as_value,
            _coerce_type,
            _register_tla_tensor_metadata,
            _register_tla_tensor_type,
            _require_frontend_state,
            _tla_tensor_type_for_mlir_value,
            _vector_lane_count,
        )
        from ..execution_lowering import TlaLoweringError
        from ..params import LoadDist, NormalLoadParams, PostMode, UnalignLoadParams
        from ..types import dtype_size_bytes

        loc = _normalize_user_loc(loc)
        _require_frontend_state("load")
        _runtime._require_enclosing_region("load", "vec.func")
        if params is None:
            params = NormalLoadParams()
        elif not isinstance(params, (NormalLoadParams, UnalignLoadParams)):
            raise TlaLoweringError(
                "load params must be NormalLoadParams or UnalignLoadParams, "
                f"got {type(params).__name__}"
            )

        if params.post_mode != PostMode.POST_MODE_NORMAL:
            raise NotImplementedError(
                f"currently unsupported post_mode {params.post_mode!r}"
            )
        if params.post_update_stride != 0:
            raise NotImplementedError(
                f"currently unsupported post_update_stride {params.post_update_stride}"
            )
        if isinstance(params, UnalignLoadParams) and params.is_pre:
            raise NotImplementedError(
                f"currently unsupported is_pre {params.is_pre}"
            )

        load_kwargs: dict[str, Any] = {"loc": loc}
        if isinstance(params, UnalignLoadParams):
            load_kwargs["unaligned_ub_access"] = True
        elif isinstance(params, NormalLoadParams) and params.load_dist != LoadDist.DIST_NORM:
            ctx = loc.context if loc is not None else mlir_ir.Context.current
            load_kwargs["load_dist"] = mlir_ir.Attribute.parse(
                f"#tla.load_dist<{params.load_dist}>",
                context=ctx,
            )

        source = _as_value(self)
        source_desc = _tla_tensor_type_for_mlir_value(source)
        result_desc = source_desc
        if (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_BRC_B32
        ):
            lanes = _vector_lane_count(dtype_size_bytes(source_desc.element_type))
            result_desc = source_desc.with_updates(
                shape=lanes, stride=1, origin_shape=lanes
            )

        result = _tla_ops_gen.load(_coerce_type(result_desc), source, **load_kwargs)
        _register_tla_tensor_type(result, result_desc)
        _register_tla_tensor_metadata(result, result_desc.metadata())
        return VectorSSA(result)

    @dsl_user_op
    def store(
        self,
        value: Any,
        *,
        mask: Any | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        """Store a vector SSA value into this tensor tile inside a tla.vec.func region.

        An optional ``mask`` (a ``MaskSSA`` from ``tla.create_mask`` or
        ``tla.update_mask``) controls which lanes are written; masked-out lanes
        are left untouched. Only a ``MaskSSA`` is accepted (validated below); a
        ``mask`` here is typed ``Any`` to avoid a circular import of ``MaskSSA``.
        """
        from ..core_api import (
            _as_value,
            _require_category,
            _require_frontend_state,
        )

        loc = _normalize_user_loc(loc)
        _require_category("store", "value", value, "vector_ssa", 1)
        if mask is not None:
            _require_category("store", "mask", mask, "mask_ssa", 2)
        _require_frontend_state("store")
        _runtime._require_enclosing_region("store", "vec.func")
        mask_val = _as_value(mask) if mask is not None else None
        _tla_ops_gen.store(_as_value(self), _as_value(value), mask=mask_val, loc=loc)


def _normalize_user_loc(loc: mlir_ir.Location | None) -> mlir_ir.Location | None:
    if loc is None and _runtime._current_frontend_state() is not None:
        from ..core_api import _capture_user_loc

        return _capture_user_loc()
    if loc is not None and not isinstance(loc, mlir_ir.Location):
        raise TypeError(f"loc must be mlir.ir.Location or None, got {type(loc).__name__}")
    return loc


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
