"""Tla dialect front-end."""

from .core import lower_copy
from .runtime import (
    DlpackBridgeError,
    export_dlpack_capsule,
    from_dlpack,
    make_fake_tensor,
)
from .tensor import (
    _Tensor,
    normalize_tile_view_coord,
    scale_tile_coord_by_shape,
)
from .typing import Tensor, TypedTensor

__all__ = [
    "Tensor",
    "TypedTensor",
    "DlpackBridgeError",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
    "_Tensor",
    "lower_copy",
    "normalize_tile_view_coord",
    "scale_tile_coord_by_shape",
]
