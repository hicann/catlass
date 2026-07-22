"""Runtime DLPack tensor binding."""

from __future__ import annotations

import ctypes
from typing import Any, Iterable, Iterator

from mlir import ir as mlir_ir  # type: ignore[assignment]

from ..address_space import AddressSpace
from ..base_dsl.runtime.dlpack_types import ASCEND_DEVICE_TYPES, DLTensor
from ..types import (
    RuntimeTensorError,
    TlaIndexTreeType,
    TlaLayoutDescriptor,
    TlaTensorTypeDescriptor,
    _TENSOR_DTYPE_SIZES,
    _coerce_host_tensor_addrspace,
    _coerce_host_tensor_dtype,
    _deduce_leading_dim,
    _flat_layout_leaves,
    _flatten_int_leaves_tree,
    _replace_flat_leaves_in_tree,
    _track_live_tensor,
    _tree_structure_mask,
    _try_remap_stride_coord_trees,
    dtype_size_bytes,
)
from .typing import Tensor as TensorABC


class DlpackBridgeError(RuntimeError):
    """Raised when DLPack export or parsing fails."""


def export_dlpack_capsule(tensor: Any, *, stream: int | None = -1) -> Any:
    """Call ``__dlpack__``, preferring ``stream=-1`` when supported."""
    if not hasattr(tensor, "__dlpack__"):
        raise DlpackBridgeError(
            f"object {type(tensor).__name__!r} does not implement __dlpack__()"
        )
    if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
        raise DlpackBridgeError(
            "tensor exposes __dlpack_device__ only; full __dlpack__() is required"
        )
    if stream is not None:
        try:
            return tensor.__dlpack__(stream=stream)  # type: ignore[attr-defined]
        except (TypeError, RuntimeError):
            # NumPy raises RuntimeError ("only supports stream=None"); others may
            # reject unknown stream via TypeError.
            pass
    return tensor.__dlpack__()  # type: ignore[attr-defined]


class _Tensor(TensorABC):
    """TLA runtime tensor with compile metadata and optional DLPack binding.

    Construct via ``tla.Tensor(...)`` or :func:`make_fake_tensor` inside
    :func:`~catlass.runtime._eager_capture` for metadata-only tensors, or via
    :func:`from_dlpack` to bind an Ascend/NPU buffer zero-copy.

    Non-symbolic tensors must use :func:`catlass.core_api.make_shape` for ``shape`` and
    ``origin_shape`` (required), :func:`catlass.core_api.make_coord` for ``coord``, and
    :func:`catlass.core_api.make_stride` for ``stride``.
    """

    @staticmethod
    def _parse_capsule(capsule: Any) -> dict[str, Any]:
        """Parse a ``dltensor`` capsule without invoking its deleter (borrowed fields)."""
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
            ctypes.py_object,
            ctypes.c_char_p,
        ]
        # Capsule holds DLManagedTensor*; dl_tensor is its first member (dlpack.h).
        managed_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
        dl = ctypes.cast(managed_ptr, ctypes.POINTER(DLTensor)).contents
        ndim = int(dl.ndim)
        shape = tuple(int(dl.shape[i]) for i in range(ndim))
        if not dl.strides:
            raise DlpackBridgeError(
                "DLPack tensor has null strides; explicit per-dimension strides are "
                "required (e.g. torch_npu / PyTorch export). NumPy C-contiguous "
                "buffers are not supported."
            )
        strides = tuple(int(dl.strides[i]) for i in range(ndim))
        data_ptr = int(dl.data or 0) + int(dl.byte_offset)
        return {
            "data_ptr": data_ptr,
            "device_type": int(dl.device.device_type),
            "device_id": int(dl.device.device_id),
            "shape": shape,
            "strides": strides,
            "dtype_code": int(dl.dtype.code),
            "dtype_bits": int(dl.dtype.bits),
            "dtype_lanes": int(dl.dtype.lanes),
        }

    def __init__(
        self,
        shape: Any,
        dtype: Any,
        *,
        addrspace: Any = AddressSpace.gm,
        data_ptr: int = 0,
        origin_shape: Iterable[Any] | None = None,
        coord: Iterable[Any] | None = None,
        stride: Any | None = None,
        layout_tag: Any | None = None,
    ) -> None:
        self._external_binding = False
        self._assumed_align: int | None = None
        self._shape_components: tuple[Any, ...] | None = None
        self._shape_tuple: tuple[int, ...] | None = None
        self._dynamic_shape_tree: Any | None = None
        self._dynamic_stride_tree: Any | None = None

        self.data_ptr = data_ptr
        self._initialize_metadata(
            shape,
            dtype,
            addrspace=addrspace,
            origin_shape=origin_shape,
            coord=coord,
            stride=stride,
            layout_tag=layout_tag,
        )

    def _initialize_metadata(
        self,
        shape: Any,
        dtype: Any,
        *,
        addrspace: Any,
        origin_shape: Iterable[Any] | None,
        coord: Iterable[Any] | None,
        stride: Any | None,
        layout_tag: Any | None,
    ) -> None:
        from ..core_api import _Coord, _Shape, _Stride, _resolve_arch_layout_tag

        self._dtype = _coerce_host_tensor_dtype(dtype)
        self.addrspace = _coerce_host_tensor_addrspace(addrspace)
        self.layout_tag = _resolve_arch_layout_tag(layout_tag, for_op="Tensor")

        if not isinstance(shape, _Shape):
            raise TypeError(
                "Tensor(shape=...): shape must be the result of tla.make_shape(...)"
            )

        comp = shape._components
        flat = tuple(_flatten_int_leaves_tree(comp))
        self._shape_tuple = flat
        self._shape_components = comp

        if origin_shape is None:
            raise TypeError(
                "Tensor(shape=tla.make_shape(...)): origin_shape is required; "
                "pass tla.make_shape(...) for logical bounds"
            )
        if isinstance(origin_shape, _Shape):
            self.origin_shape = origin_shape._components
        else:
            raise TypeError(
                "Tensor origin_shape=... must be the result of tla.make_shape(...)"
            )

        rs_stride, rs_coord = _try_remap_stride_coord_trees(
            comp, self.origin_shape, self._dtype, self.layout_tag
        )

        if coord is None:
            if rs_coord is None:
                raise ValueError(
                    "Tensor(coord=None): cannot derive coord from layout remap; "
                    "use a flat logical origin_shape (M,N) without nested parentheses, "
                    "or pass tla.make_coord(...)"
                )
            self.coord = rs_coord
        elif isinstance(coord, _Coord):
            self.coord = coord._components
        else:
            raise TypeError(
                "Tensor coord=... must be None or the result of tla.make_coord(...)"
            )

        if stride is None:
            if rs_stride is None:
                raise ValueError(
                    "Tensor(stride=None): cannot derive stride from layout remap "
                    "(remap stride tree must match shape tree); pass tla.make_stride(...)"
                )
            self._stride_components = rs_stride
        elif isinstance(stride, _Stride):
            sc = stride._components
            if _tree_structure_mask(sc) != _tree_structure_mask(comp):
                raise ValueError(
                    "Tensor stride component tree must match shape tree structure"
                )
            self._stride_components = sc
        else:
            raise TypeError(
                "Tensor stride=... must be None or the result of tla.make_stride(...)"
            )

        _track_live_tensor(self)

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def shape(self) -> Any:
        if self._shape_components is None:
            raise NotImplementedError(f"{type(self).__name__} does not expose shape")
        return self._shape_components

    @property
    def stride(self) -> Any:
        return self._stride_components

    def _require_bound(self) -> None:
        if not getattr(self, "_external_binding", False) or self.data_ptr == 0:
            raise RuntimeError(
                "Tensor buffer is not bound; use from_dlpack first."
            )

    def mark_layout_dynamic(self, leading_dim: int | None = None) -> "_Tensor":
        """Mark stride metadata dynamic."""
        flat_strides = _flat_layout_leaves(self.stride)
        shape_tuple = self._shape_tuple or ()
        if leading_dim is None:
            leading_dim = _deduce_leading_dim(shape_tuple, flat_strides)
        if leading_dim < 0 or leading_dim >= len(flat_strides):
            raise RuntimeTensorError(
                f"leading_dim={leading_dim} out of range for rank {len(flat_strides)}"
            )
        if int(flat_strides[leading_dim]) != 1:
            raise RuntimeTensorError(
                f"leading_dim={leading_dim} has stride {flat_strides[leading_dim]}, expected 1"
            )
        new_stride_leaves = [
            1 if index == leading_dim else None for index in range(len(flat_strides))
        ]
        self._dynamic_stride_tree = _replace_flat_leaves_in_tree(
            self.stride, new_stride_leaves
        )
        return self

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: tuple[int, ...] | None = None,
        divisibility: int = 1,
    ) -> "_Tensor":
        """Mark a shape mode dynamic."""
        flat_shape = list(self._shape_tuple or ())
        if mode < 0 or mode >= len(flat_shape):
            raise RuntimeTensorError(
                f"mode={mode} out of range for rank {len(flat_shape)}"
            )
        if int(divisibility) < 1:
            raise RuntimeTensorError(
                f"divisibility must be positive, got {divisibility}"
            )
        if stride_order is not None:
            expected = len(flat_shape)
            if len(stride_order) != expected or sorted(stride_order) != list(
                range(expected)
            ):
                raise RuntimeTensorError(
                    f"stride_order {stride_order!r} is not a permutation of "
                    f"range({expected})"
                )
        flat_shape[mode] = None
        self._dynamic_shape_tree = _replace_flat_leaves_in_tree(
            self._shape_components, flat_shape
        )
        return self

    def _layout_shape_components(self) -> tuple[Any, ...]:
        shape_components = self._shape_components
        if shape_components is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        return self._dynamic_shape_tree or shape_components

    def _layout_stride_components(self) -> Any:
        return self._dynamic_stride_tree or self.stride

    def _ptr_alignment(self) -> int:
        assumed = getattr(self, "_assumed_align", None)
        if assumed is not None:
            return max(1, int(assumed))
        return max(1, dtype_size_bytes(str(self.dtype)))

    def tla_tensor_type_descriptor(self) -> TlaTensorTypeDescriptor:
        """Structured ``!tla.tensor`` descriptor from host metadata."""
        st = self._shape_tuple
        addr_kw = (self.addrspace or "gm").strip()
        if st is None or self.stride is None or self.layout_tag is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        return TlaTensorTypeDescriptor(
            layout=TlaLayoutDescriptor(
                shape=TlaIndexTreeType("shape", self._layout_shape_components()),
                stride=TlaIndexTreeType("stride", self._layout_stride_components()),
                origin_shape=TlaIndexTreeType("shape", self.origin_shape),
                layout_tag=str(self.layout_tag),
            ),
            coord=self.coord,
            element_type=str(self.dtype),
            addrspace=addr_kw,
            ptr_alignment=self._ptr_alignment(),
        )

    def __tla_type__(self) -> str:
        return str(self.tla_tensor_type_descriptor().to_mlir_type())

    def __c_pointers__(self) -> list[int]:
        return [int(self.data_ptr)]

    def __get_mlir_types__(
        self, context: mlir_ir.Context | None = None
    ) -> list[mlir_ir.Type]:
        return [self.tla_tensor_type_descriptor().to_mlir_type(context)]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_Tensor":
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

    def prepare_for_launch(self) -> None:
        self._require_bound()

    def __setitem__(self, index: int | slice, value: Any) -> None:
        raise TypeError("runtime Tensor is not indexable")

    def __getitem__(self, index: int | slice) -> Any:
        raise TypeError("runtime Tensor is not indexable")

    def __len__(self) -> int:
        raise TypeError("runtime Tensor is not indexable")

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("runtime Tensor is not indexable")

    @property
    def data(self) -> Any:
        raise TypeError("runtime Tensor is not indexable")

    def __str__(self) -> str:
        return self.__tla_type__()

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"addrspace={self.addrspace!r}, data_ptr={self.data_ptr!r}, "
            f"origin_shape={self.origin_shape!r}, coord={self.coord!r}, "
            f"stride={self.stride!r}, layout_tag={self.layout_tag!r})"
        )

    @property  # type: ignore[misc]
    def __class__(self) -> type[TensorABC]:
        return TensorABC


def from_dlpack(
    tensor_dlpack: object,
    *,
    layout_tag: Any,
    origin_shape: Any | None = None,
    assumed_align: int | None = None,
    stream: int | None = -1,
) -> _Tensor:
    """Convert a DLPack object to a TLA runtime tensor (zero-copy).

    ``tensor_dlpack`` must export an Ascend/NPU tensor (e.g. ``torch_npu``). CPU /
    NumPy buffers are not supported. ``layout_tag`` must be a ``tla.arch`` layout
    sentinel (e.g. ``tla.arch.ColumnMajor``).

    When ``origin_shape`` is omitted, logical ``origin_shape`` is derived from the
    DLPack physical shape/strides and ``layout_tag``. For dense 2-D buffers the
    physical storage must match ``basic_matmul`` preparation: ``tensor.contiguous()``
    for ``row_major``, or ``tensor.permute(1, 0).contiguous()`` for ``column_major``
    (row-major physical on the permuted shape). When ``origin_shape`` is provided
    it is used directly and DLPack stride derivation is skipped. Shape / stride metadata
    come from the logical origin and ``layout_tag`` via layout remap, not from raw
    DLPack fields. Use :meth:`_Tensor.mark_layout_dynamic` /
    :meth:`_Tensor.mark_compact_shape_dynamic` when dynamic layout metadata is required.
    """
    from ..base_dsl.runtime.dlpack_types import DLDataTypeCode
    from ..base_dsl.typing import (
        BFloat16,
        Bool,
        Float16,
        Float32,
        Int16,
        Int32,
        Int64,
        Int8,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    )
    from ..core_api import (
        _Shape,
        _remap_tensor_like_prefix_fields_for_layout_trees,
        _resolve_arch_layout_tag,
        make_coord,
        make_shape,
        make_stride,
    )
    from ..runtime import _eager_capture

    try:
        dlpack_data = export_dlpack_capsule(tensor_dlpack, stream=stream)
    except DlpackBridgeError as exc:
        raise RuntimeTensorError(str(exc)) from exc

    try:
        parsed = _Tensor._parse_capsule(dlpack_data)
    except DlpackBridgeError as exc:
        raise RuntimeTensorError(str(exc)) from exc

    phys_shape = tuple(int(dim) for dim in parsed["shape"])
    phys_strides = tuple(int(stride) for stride in parsed["strides"])
    data_ptr = int(parsed["data_ptr"])

    device_type = int(parsed["device_type"])
    device_id = int(parsed["device_id"])
    if device_type not in ASCEND_DEVICE_TYPES:
        raise RuntimeTensorError(
            f"DLPack device_type={device_type} device_id={device_id} is not "
            f"an Ascend/NPU device; from_dlpack requires a device-resident buffer "
            f"(supported types: {sorted(ASCEND_DEVICE_TYPES)})."
        )

    lanes = int(parsed["dtype_lanes"])
    if lanes != 1:
        raise RuntimeTensorError(f"unsupported DLPack dtype lanes={lanes}")
    dtype_code = int(parsed["dtype_code"])
    dtype_bits = int(parsed["dtype_bits"])
    # DLPack bool is byte-sized storage (kDLBool, bits=8), matching torch.bool /
    # NumPy bool_. Map to TLA Bool (MLIR i1); tensor element size is still 1 byte.
    dtype_mapping: dict[tuple[int, int], type] = {
        (DLDataTypeCode.kDLInt, 8): Int8,
        (DLDataTypeCode.kDLInt, 16): Int16,
        (DLDataTypeCode.kDLInt, 32): Int32,
        (DLDataTypeCode.kDLInt, 64): Int64,
        (DLDataTypeCode.kDLUInt, 8): UInt8,
        (DLDataTypeCode.kDLUInt, 16): UInt16,
        (DLDataTypeCode.kDLUInt, 32): UInt32,
        (DLDataTypeCode.kDLUInt, 64): UInt64,
        (DLDataTypeCode.kDLFloat, 16): Float16,
        (DLDataTypeCode.kDLFloat, 32): Float32,
        (DLDataTypeCode.kDLBfloat, 16): BFloat16,
        (DLDataTypeCode.kDLBool, 8): Bool,
    }
    dtype = dtype_mapping.get((dtype_code, dtype_bits))
    if dtype is None:
        raise RuntimeTensorError(
            f"unsupported DLPack dtype code={dtype_code} bits={dtype_bits} lanes={lanes}"
        )

    if layout_tag is None:
        raise RuntimeTensorError(
            "from_dlpack requires layout_tag (e.g. tla.arch.RowMajor or "
            "tla.arch.ColumnMajor)"
        )
    resolved_layout = layout_tag
    dtype_token = str(getattr(dtype, "dtype", "")).strip().lower()
    layout_token = _resolve_arch_layout_tag(resolved_layout, for_op="from_dlpack")
    if origin_shape is None:
        row_major_compact = (
            len(phys_shape) == 2
            and (phys_shape[1] == 1 or phys_strides[1] == 1)
            and (phys_shape[0] == 1 or phys_strides[0] == phys_shape[1])
        )
        if (
            len(phys_shape) == 2
            and layout_token in ("row_major", "column_major")
            and (
                (layout_token == "row_major" and not row_major_compact)
                or (
                    layout_token == "column_major"
                    and (phys_strides[1] != 1 or phys_strides[0] != phys_shape[1])
                )
            )
        ):
            torch_hint = (
                "tensor.contiguous()"
                if layout_token == "row_major"
                else "tensor.permute(1, 0).contiguous()"
            )
            raise RuntimeTensorError(
                f"from_dlpack layout_tag={layout_token!r} requires a buffer prepared as "
                f"{torch_hint}; got shape={phys_shape}, strides={phys_strides}"
            )
        logical_origin = (
            (phys_shape[1], phys_shape[0])
            if len(phys_shape) == 2 and layout_token == "column_major"
            else phys_shape
        )
    else:
        if not isinstance(origin_shape, _Shape):
            raise TypeError(
                "from_dlpack origin_shape=... must be the result of tla.make_shape(...)"
            )
        logical_origin = origin_shape._components
    trees = _remap_tensor_like_prefix_fields_for_layout_trees(
        logical_origin, dtype_token, layout_token
    )
    if trees is None:
        raise RuntimeTensorError(
            f"from_dlpack cannot derive layout metadata for origin_shape={logical_origin!r} "
            f"layout={resolved_layout!r}"
        )
    shape_tree, stride_tree, coord_tree, origin_tree = trees

    with _eager_capture():
        tensor = _Tensor(
            make_shape(*shape_tree),
            dtype,
            origin_shape=make_shape(*origin_tree),
            coord=make_coord(*coord_tree),
            stride=make_stride(*stride_tree),
            layout_tag=resolved_layout,
            data_ptr=data_ptr,
        )
    if assumed_align is not None:
        tensor._assumed_align = int(assumed_align)
    tensor._external_binding = True
    return tensor


def make_fake_tensor(*args: Any, **kwargs: Any) -> _Tensor:
    """Create a metadata-only host tensor inside an eager-capture session."""
    from ..runtime import _eager_capture

    with _eager_capture():
        return _Tensor(*args, **kwargs)


__all__ = [
    "DlpackBridgeError",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
    "_Tensor",
]
