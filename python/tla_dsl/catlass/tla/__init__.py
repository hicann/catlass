"""Tla-specific semantic helpers."""

from .tensor import normalize_tile_view_coord, scale_tile_coord_by_shape

__all__ = [
    "normalize_tile_view_coord",
    "scale_tile_coord_by_shape",
]
