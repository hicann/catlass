"""Runtime DLPack tensor binding."""

from __future__ import annotations

import ctypes
import weakref
from typing import Any, Iterable, Iterator

from mlir import ir as mlir_ir  # type: ignore[assignment]

from ..address_space import AddressSpace
from ..base_dsl.runtime.dlpack_types import (
    ASCEND_DEVICE_TYPES,
    DLManagedTensor,
)
from ..types import (
    RuntimeTensorError,
    TlaIndexTreeType,
    TlaLayoutDescriptor,
    TlaTensorTypeDescriptor,
    _coerce_host_tensor_addrspace,
    _coerce_host_tensor_dtype,
    _deduce_compact_stride_order,
    _flat_layout_leaves,
    _flatten_int_leaves_tree,
    _replace_flat_leaves_in_tree,
    _tree_structure_mask,
    _try_remap_stride_coord_trees,
    dtype_size_bytes,
)
from .typing import Tensor as TensorABC


def _as_host_index_tree(value: Any, *, what: str) -> Any:
    """Normalize a Host index tree: bare ``int`` → ``(int,)``; otherwise require ``tuple``."""
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple):
        return value
    raise TypeError(
        f"{what} must be an int or nested int tuple, got {type(value).__name__}"
    )


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


def _consume_capsule(capsule: Any) -> None:
    """Take ownership of a ``dltensor`` capsule by renaming it to ``used_dltensor``.

    A ``dltensor`` capsule's destructor calls ``DLManagedTensor.deleter`` when the
    capsule dies unconsumed, which would release the buffer as soon as the local
    reference goes out of scope. Renaming is the DLPack handshake that says the
    consumer now owns the ``DLManagedTensor`` and will call the deleter itself
    (see :func:`_release_managed_tensor`); producers key their capsule destructor
    on the old name, so it becomes a no-op.
    """
    set_name = ctypes.pythonapi.PyCapsule_SetName
    set_name.restype = ctypes.c_int
    set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
    # The name is borrowed by the capsule, so it must outlive it: keep the bytes
    # object alive in a module-level constant rather than a temporary.
    if set_name(capsule, _USED_DLTENSOR_NAME) != 0:
        raise DlpackBridgeError("failed to rename DLPack capsule to 'used_dltensor'")


# PyCapsule_SetName borrows the pointer; this must never be garbage-collected.
_USED_DLTENSOR_NAME = b"used_dltensor"


def _release_managed_tensor(managed_ptr: int) -> None:
    """Call ``DLManagedTensor.deleter`` once, releasing the producer's allocation.

    Registered through :func:`weakref.finalize` on the owning tensor, so it must
    not close over that tensor (a strong reference would keep it alive forever).
    """
    managed = ctypes.cast(ctypes.c_void_p(managed_ptr), ctypes.POINTER(DLManagedTensor))
    deleter = managed.contents.deleter
    # A NULL deleter means the producer has nothing to release (DLPack allows it).
    if deleter:
        deleter(managed)


class _Tensor(TensorABC):
    """TLA runtime host tensor with compile metadata and optional DLPack binding.

    Not a Host public constructor. Use :func:`make_fake_tensor` (unbound sample) or
    :func:`from_dlpack` (NPU buffer). ``tla.Tensor`` is the shared ABC / annotation.
    """

    @staticmethod
    def _parse_capsule(capsule: Any) -> dict[str, Any]:
        """Read a ``dltensor`` capsule's fields; does not consume it (see :func:`_consume_capsule`)."""
        is_valid = ctypes.pythonapi.PyCapsule_IsValid
        is_valid.restype = ctypes.c_int
        is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
        # Check the name before reading it: PyCapsule_GetPointer on a renamed
        # capsule returns NULL *and* sets the error indicator, which would
        # otherwise surface later at an unrelated call site.
        if not is_valid(capsule, b"dltensor"):
            raise DlpackBridgeError(
                "not a fresh 'dltensor' capsule; a DLPack capsule can only be "
                "consumed once (an already-consumed capsule is renamed to "
                "'used_dltensor')"
            )
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
            ctypes.py_object,
            ctypes.c_char_p,
        ]
        # Capsule holds DLManagedTensor*; dl_tensor is its first member (dlpack.h).
        managed_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
        dl = ctypes.cast(
            managed_ptr, ctypes.POINTER(DLManagedTensor)
        ).contents.dl_tensor
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
            "managed_ptr": int(managed_ptr),
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
        data_ptr: int | None = 0,
        origin_shape: Iterable[Any] | None = None,
        coord: Iterable[Any] | None = None,
        stride: Any | None = None,
        layout_tag: Any | None = None,
    ) -> None:
        self._external_binding = False
        self._assumed_align: int | None = None
        # DLPack ownership, set by from_dlpack: the consumed DLManagedTensor is
        # released by _dlpack_release when this tensor dies, and _dlpack_source
        # keeps the producer object itself alive meanwhile.
        self._dlpack_source: Any = None
        self._dlpack_release: weakref.finalize | None = None
        self._shape_components: tuple[Any, ...] | None = None
        self._shape_tuple: tuple[int, ...] | None = None
        self._dynamic_shape_tree: Any | None = None
        self._dynamic_origin_shape_tree: Any | None = None
        self._dynamic_stride_tree: Any | None = None
        self._memref_launch_fields_cache: dict[str, int] | None = None

        # None is treated as unbound (compile-time placeholder tensors).
        self.data_ptr = 0 if data_ptr is None else int(data_ptr)
        # Non-zero data_ptr means the host already owns a device buffer (e.g. torch).
        if self.data_ptr != 0:
            self._external_binding = True
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
        from ..core_api import _resolve_arch_layout_tag

        self._dtype = _coerce_host_tensor_dtype(dtype)
        self.addrspace = _coerce_host_tensor_addrspace(addrspace)
        self.layout_tag = _resolve_arch_layout_tag(layout_tag, for_op="Tensor")

        comp = _as_host_index_tree(shape, what="shape")
        flat = tuple(_flatten_int_leaves_tree(comp))
        self._shape_tuple = flat
        self._shape_components = comp

        if origin_shape is None:
            raise TypeError("Tensor origin_shape is required")
        self.origin_shape = _as_host_index_tree(origin_shape, what="origin_shape")

        rs_stride, rs_coord = _try_remap_stride_coord_trees(
            comp, self.origin_shape, self._dtype, self.layout_tag
        )

        if coord is None:
            if rs_coord is None:
                raise ValueError(
                    "Tensor(coord=None): cannot derive coord from layout remap; "
                    "use a flat logical origin_shape (M,N) without nested parentheses, "
                    "or pass an explicit coord tree"
                )
            self.coord = rs_coord
        else:
            self.coord = _as_host_index_tree(coord, what="coord")

        if stride is None:
            if rs_stride is None:
                raise ValueError(
                    "Tensor(stride=None): cannot derive stride from layout remap "
                    "(remap stride tree must match shape tree); pass an explicit stride tree"
                )
            self._stride_components = rs_stride
        else:
            sc = _as_host_index_tree(stride, what="stride")
            if _tree_structure_mask(sc) != _tree_structure_mask(comp):
                raise ValueError(
                    "Tensor stride component tree must match shape tree structure"
                )
            self._stride_components = sc

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
            raise RuntimeError("Tensor buffer is not bound; use from_dlpack first.")

    def mark_layout_dynamic(self, leading_dim: int | None = None) -> "_Tensor":
        """Directory: Host Tensor / Dynamic Layout
        Description:
            Mark every shape mode dynamic so one compiled artifact can run at
            different extents. Strides become dynamic except the leading dimension
            (stride stays `1`). Broadcast strides of `0` are kept. Matching
            `origin_shape` leaves become dynamic so the compile type no longer
            depends on concrete DLPack extents.

            Parameters:
            - *`leading_dim`* (`int | None`): Index of the unit-stride (leading)
              dimension. Optional; default `None` (inferred from `layout_tag` or
              compact stride order).

            Constraints:
            - In-place; returns `self` (chainable).
            - All `coord` leaves must be `0`; sliced views are rejected.
            - `leading_dim` must have stride `1`.
            - For NZFamily layouts, each two-leaf physical shape group maps to one
              logical `origin_shape` axis.

            Example:
            ```python
            ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
            ta = ta.mark_layout_dynamic()
            artifact = tla.compile(my_kernel, ta, options="--npu-arch 3510")
            ```

        """
        # Dynamic GM ABI hard-codes root coord/offset 0 (same rule as
        # TlaLowerFuncPass::validateKernelTensorArg).
        coord_leaves = _flat_layout_leaves(self.coord, allow_dynamic=True)
        if any(leaf is None for leaf in coord_leaves) or not all(
            int(leaf) == 0 for leaf in coord_leaves
        ):
            raise RuntimeTensorError(
                "mark_*_dynamic requires a root tensor with zero coordinates; "
                f"got coord={self.coord!r}"
            )
        flat_strides = _flat_layout_leaves(self.stride)
        shape_tuple = self._shape_tuple or ()
        if leading_dim is None:
            from ..core_api import (
                _COLUMN_MAJOR_LAYOUT_TOKENS,
                is_row_major_layout,
            )

            # Prefer layout-tag semantics when unit strides are ambiguous
            # (e.g. ColumnMajor with shape[0]==1 → strides (1,1)).
            if is_row_major_layout(self.layout_tag):
                leading_dim = len(flat_strides) - 1
            elif self.layout_tag in _COLUMN_MAJOR_LAYOUT_TOKENS:
                leading_dim = 0
            else:
                leading_dim = _deduce_compact_stride_order(
                    shape_tuple, flat_strides, strict_unit_stride=True
                )[-1]
        if leading_dim < 0 or leading_dim >= len(flat_strides):
            raise RuntimeTensorError(
                f"leading_dim={leading_dim} out of range for rank {len(flat_strides)}"
            )
        if int(flat_strides[leading_dim]) != 1:
            raise RuntimeTensorError(
                f"leading_dim={leading_dim} has stride {flat_strides[leading_dim]}, expected 1"
            )
        new_stride_leaves = [
            1
            if index == leading_dim
            else (0 if int(flat_strides[index]) == 0 else None)
            for index in range(len(flat_strides))
        ]
        self._dynamic_stride_tree = _replace_flat_leaves_in_tree(
            self.stride, new_stride_leaves
        )
        self._mark_shape_modes_dynamic(range(len(shape_tuple)))
        return self

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: tuple[int, ...] | None = None,
    ) -> "_Tensor":
        """Directory: Host Tensor / Dynamic Layout
        Description:
            Mark one compact shape mode dynamic. Strides of modes major to `mode`
            (whose compact stride product includes that extent) become dynamic as
            well. Matching `origin_shape` leaves are marked so the compile type does
            not depend on the concrete size.

            Parameters:
            - *`mode`* (`int`): Flattened shape-leaf index to mark dynamic
              (0-based). Required.
            - *`stride_order`* (`tuple[int, ...] | None`): Compact stride order
              (outer → inner). Optional; inferred from current strides when omitted.

            Constraints:
            - In-place; returns `self`.
            - All `coord` leaves must be `0`.
            - `stride_order` must be a permutation of `range(rank)`.
            - For NZFamily layouts, physical modes 0/1 map to logical M and modes
              2/3 map to logical N.

            Example:
            ```python
            ta = from_dlpack(a.contiguous(), layout_tag=tla.arch.RowMajor)
            ta = ta.mark_compact_shape_dynamic(mode=0)
            ```

        """
        coord_leaves = _flat_layout_leaves(self.coord, allow_dynamic=True)
        if any(leaf is None for leaf in coord_leaves) or not all(
            int(leaf) == 0 for leaf in coord_leaves
        ):
            raise RuntimeTensorError(
                "mark_*_dynamic requires a root tensor with zero coordinates; "
                f"got coord={self.coord!r}"
            )
        flat_shape = list(self._shape_tuple or ())
        rank = len(flat_shape)
        if mode < 0 or mode >= rank:
            raise RuntimeTensorError(f"mode={mode} out of range for rank {rank}")
        if stride_order is None:
            stride_order = _deduce_compact_stride_order(
                flat_shape, _flat_layout_leaves(self.stride)
            )
        elif len(stride_order) != rank or sorted(stride_order) != list(range(rank)):
            raise RuntimeTensorError(
                f"stride_order {stride_order!r} is not a permutation of range({rank})"
            )

        self._mark_shape_modes_dynamic((mode,))

        # Modes major to ``mode`` (appear before it outer→inner) include its size
        # in their compact stride product → mark those strides dynamic.
        mode_pos = stride_order.index(mode)
        major_modes = set(stride_order[:mode_pos])
        flat_strides = _flat_layout_leaves(
            self._layout_stride_components(),
            allow_dynamic=True,
            expected_rank=rank,
        )
        for index in major_modes:
            if int(flat_strides[index] or 0) != 0:
                flat_strides[index] = None
        self._dynamic_stride_tree = _replace_flat_leaves_in_tree(
            self.stride, flat_strides
        )
        return self

    def _mark_shape_modes_dynamic(self, modes: Any) -> None:
        shape_components = self._shape_components
        if shape_components is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        rank = len(self._shape_tuple or ())
        mode_indices = tuple(int(mode) for mode in modes)
        if any(mode < 0 or mode >= rank for mode in mode_indices):
            raise RuntimeTensorError(
                f"dynamic shape mode out of range for rank {rank}: {mode_indices!r}"
            )
        shape_leaves = _flat_layout_leaves(
            self._layout_shape_components(),
            allow_dynamic=True,
            expected_rank=rank,
        )
        origin_leaves = _flat_layout_leaves(
            self._layout_origin_shape_components(),
            allow_dynamic=True,
        )
        if len(origin_leaves) == rank:
            origin_mode_indices = mode_indices
        else:
            # NZFamily shape/stride are physical 2x2 trees with four leaves,
            # while origin_shape remains the flat logical (M, N) pair. Both
            # leaves in the first physical group map to M; both leaves in the
            # second group map to N.
            from ..core_api import _NZ_FAMILY_LAYOUT_TOKENS

            is_nz_family_2x2 = (
                self.layout_tag in _NZ_FAMILY_LAYOUT_TOKENS
                and rank == 4
                and len(origin_leaves) == 2
                and _tree_structure_mask(self._layout_shape_components())
                == ((None, None), (None, None))
                and _tree_structure_mask(self._layout_origin_shape_components())
                == (None, None)
            )
            if not is_nz_family_2x2:
                raise RuntimeTensorError(
                    f"layout tree rank mismatch: expected {rank} leaves, "
                    f"got {len(origin_leaves)}"
                )
            origin_mode_indices = tuple(sorted({mode // 2 for mode in mode_indices}))

        for mode in mode_indices:
            shape_leaves[mode] = None
        for mode in origin_mode_indices:
            origin_leaves[mode] = None
        self._dynamic_shape_tree = _replace_flat_leaves_in_tree(
            shape_components, shape_leaves
        )
        self._dynamic_origin_shape_tree = _replace_flat_leaves_in_tree(
            self.origin_shape, origin_leaves
        )

    def build_memref_launch_fields(self) -> dict[str, int]:
        """Build unified schema-v4 GM launch fields (13 slots, pad unused with 1)."""
        self._require_bound()
        data_ptr = int(self.data_ptr)
        cached = self._memref_launch_fields_cache
        if cached is not None and cached["allocated"] == data_ptr:
            return cached

        shape = tuple(int(dim) for dim in (self._shape_tuple or ()))
        if not shape:
            raise RuntimeTensorError(
                "build_memref_launch_fields requires a concrete DLPack shape"
            )
        strides = [int(s) for s in _flat_layout_leaves(self.stride)]
        if len(strides) != len(shape):
            raise RuntimeTensorError(
                "build_memref_launch_fields shape/stride rank mismatch"
            )
        # Concrete origin from construction (mark_* only shadows the type tree).
        origin_leaves = [int(v) for v in _flat_layout_leaves(self.origin_shape)]
        if not origin_leaves:
            origin_leaves = list(shape)

        def _pad4(values: list[int]) -> list[int]:
            if len(values) > 4:
                raise RuntimeTensorError(
                    f"build_memref_launch_fields supports at most 4 shape/stride "
                    f"leaves, got {len(values)}"
                )
            return list(values) + [1] * (4 - len(values))

        sizes = _pad4(list(shape))
        stride_vals = _pad4(list(strides))
        origin0 = int(origin_leaves[0]) if len(origin_leaves) >= 1 else 1
        origin1 = int(origin_leaves[1]) if len(origin_leaves) >= 2 else 1

        fields = {
            "allocated": data_ptr,
            "aligned": data_ptr,
            "offset": 0,
            "size0": sizes[0],
            "size1": sizes[1],
            "size2": sizes[2],
            "size3": sizes[3],
            "stride0": stride_vals[0],
            "stride1": stride_vals[1],
            "stride2": stride_vals[2],
            "stride3": stride_vals[3],
            "originShape0": origin0,
            "originShape1": origin1,
        }
        self._memref_launch_fields_cache = fields
        return fields

    def _layout_shape_components(self) -> tuple[Any, ...]:
        shape_components = self._shape_components
        if shape_components is None:
            raise TypeError(
                "Tensor metadata is incomplete; construct tensors with tla.make_shape, "
                "origin_shape, coord, and stride metadata"
            )
        return self._dynamic_shape_tree or shape_components

    def _layout_origin_shape_components(self) -> Any:
        return self._dynamic_origin_shape_tree or self.origin_shape

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
                origin_shape=TlaIndexTreeType(
                    "shape", self._layout_origin_shape_components()
                ),
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
    element_type: type | None = None,
) -> _Tensor:
    """Directory: Host Tensor / Binding
    Description:
        Bind a DLPack NPU tensor to a TLA Host tensor (zero-copy). The returned
        object shares the device buffer of `tensor_dlpack`.

        Parameters:
        - *`tensor_dlpack`* (`object`): Object implementing `__dlpack__()`. Must
          be an Ascend/NPU buffer (e.g. `torch_npu`). CPU / NumPy are rejected.
          Required.
        - *`layout_tag`* (`tla.arch.*`): Layout tag such as `tla.arch.RowMajor`,
          `tla.arch.ColumnMajor`, `tla.arch.zN`. Required.
        - *`origin_shape`* (`tuple | int | None`): Logical origin as a Python int
          tree. Optional; derived from the DLPack physical shape and `layout_tag`
          when omitted. Not a Kernel `tla.make_shape`.
        - *`assumed_align`* (`int | None`): Reserved; currently unused.
        - *`stream`* (`int | None`): Passed to `__dlpack__(stream=...)`. Default
          `-1` (no stream sync). `None` omits the `stream` argument.
        - *`element_type`* (`type | None`): Optional override for the element
          type inferred from DLPack. Default `None` keeps the DLPack type.
          Use when DLPack cannot express the real type (e.g. fp8): pass
          `tla.Float8E4M3FN` / `Float8E5M2`. Must have the same per-element
          bit width as the exported buffer.

        Constraints:
        - Ownership follows the DLPack consumer contract: the capsule is consumed
          and its deleter runs when the returned tensor is destroyed. A reference
          to `tensor_dlpack` is also retained, so a temporary source such as
          `from_dlpack(x.contiguous().to(device), ...)` is safe.
        - A capsule is single-use; passing an already-consumed capsule raises
          `RuntimeTensorError`. Call `from_dlpack` again for another binding.
        - 2-D `RowMajor` requires `tensor.contiguous()`. 2-D `ColumnMajor`
          requires `tensor.permute(1, 0).contiguous()`. A mismatch raises
          `RuntimeTensorError`. Providing `origin_shape` skips that check.
        - Default layout is static. Call `mark_layout_dynamic` /
          `mark_compact_shape_dynamic` for dynamic extents.
        - When `element_type` is set, its per-element bit width must match the
          exported DLPack buffer.

        Example:
        ```python
        tx = from_dlpack(x.contiguous(), layout_tag=tla.arch.RowMajor)
        ty = from_dlpack(
            y.permute(1, 0).contiguous(),
            layout_tag=tla.arch.ColumnMajor,
        )
        ```
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
        _remap_tensor_like_prefix_fields_for_layout_trees,
        _resolve_arch_layout_tag,
    )

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

    if element_type is not None:
        # Read the widths directly rather than through a defaulted getattr: a
        # missing `width` would make both sides compare equal and turn the check
        # below into a no-op, which is exactly when it is most needed.
        try:
            exported_width = int(dtype.width)
            override_width = int(element_type.width)
        except (AttributeError, TypeError) as exc:
            raise RuntimeTensorError(
                "from_dlpack element_type override requires both the exported dtype and the "
                f"override to declare a storage width; got {dtype!r} and {element_type!r}"
            ) from exc
        # Narrowing to a sub-byte element that tiles the exported width is the
        # one width change that is well defined: the host has no 4-bit dtype, so
        # a packed fp4 buffer can only be exported as bytes, and the override is
        # what says how many elements each byte holds. origin_shape carries the
        # true element count. Anything else -- f32 read as i32, say -- is a
        # reinterpretation the caller almost never means.
        packs_into_exported = (
            override_width < exported_width and exported_width % override_width == 0
        )
        if override_width != exported_width and not packs_into_exported:
            raise RuntimeTensorError(
                f"from_dlpack element_type override {element_type.__name__} is "
                f"{override_width}-bit but the exported buffer is {exported_width}-bit; "
                "an override may narrow to a sub-byte element that tiles the "
                "exported width, but must not otherwise change it"
            )
        dtype = element_type

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
            and layout_token in ("RowMajor", "ColumnMajor")
            and (
                (layout_token == "RowMajor" and not row_major_compact)
                or (
                    layout_token == "ColumnMajor"
                    and (phys_strides[1] != 1 or phys_strides[0] != phys_shape[1])
                )
            )
        ):
            torch_hint = (
                "tensor.contiguous()"
                if layout_token == "RowMajor"
                else "tensor.permute(1, 0).contiguous()"
            )
            raise RuntimeTensorError(
                f"from_dlpack layout_tag={layout_token!r} requires a buffer prepared as "
                f"{torch_hint}; got shape={phys_shape}, strides={phys_strides}"
            )
        logical_origin = (
            (phys_shape[1], phys_shape[0])
            if len(phys_shape) == 2 and layout_token == "ColumnMajor"
            else phys_shape
        )
    else:
        if isinstance(origin_shape, tuple):
            logical_origin = origin_shape
        elif isinstance(origin_shape, int):
            logical_origin = (origin_shape,)
        else:
            raise TypeError(
                "from_dlpack origin_shape=... must be an int tree / tuple "
                "(Kernel tla.make_shape is not a Host API)"
            )
    trees = _remap_tensor_like_prefix_fields_for_layout_trees(
        logical_origin, dtype_token, layout_token
    )
    if trees is None:
        raise RuntimeTensorError(
            f"from_dlpack cannot derive layout metadata for origin_shape={logical_origin!r} "
            f"layout={resolved_layout!r}"
        )
    shape_tree, stride_tree, coord_tree, origin_tree = trees

    tensor = _Tensor(
        shape_tree,
        dtype,
        origin_shape=origin_tree,
        coord=coord_tree,
        stride=stride_tree,
        layout_tag=resolved_layout,
        data_ptr=data_ptr,
    )
    if assumed_align is not None:
        tensor._assumed_align = int(assumed_align)
    tensor._external_binding = True

    # Take ownership of the buffer the tensor is about to point at. Everything
    # that can reject the capsule has run by now, so a failure past this point
    # cannot leave a consumed-but-unowned DLManagedTensor behind.
    _consume_capsule(dlpack_data)
    tensor._dlpack_release = weakref.finalize(
        tensor, _release_managed_tensor, int(parsed["managed_ptr"])
    )
    # Also retain the producer object: its deleter may be a no-op while the
    # object itself owns the allocation, and a temporary source (e.g.
    # ``x.contiguous().to(device)``) would otherwise be freed on return and the
    # kernel would silently read reused memory.
    tensor._dlpack_source = tensor_dlpack
    return tensor


def make_fake_tensor(
    dtype: Any,
    shape: Any,
    stride: Any,
    *,
    layout_tag: Any | None = None,
    addrspace: Any = AddressSpace.gm,
    origin_shape: Iterable[Any] | None = None,
    coord: Iterable[Any] | None = None,
    assumed_align: int | None = None,
) -> _Tensor:
    """Directory: Host Tensor / Binding
    Description:
        Build a metadata-only Host tensor with no device buffer (`data_ptr == 0`).
        Use this as a `tla.compile` type sample when no NPU is needed. Bind real
        buffers with `from_dlpack`.

        Parameters:
        - *`dtype`*: Element type such as `tla.Float16` / `tla.Float32`. Required.
        - *`shape`* (`int | tuple`): Logical shape tree (nested tuples for zN
          physical layouts). Required.
        - *`stride`* (`int | tuple`): Stride tree; structure must match `shape`.
          Required.
        - *`layout_tag`*: `tla.arch` tag. Optional; default `tla.arch.RowMajor`.
        - *`addrspace`*: Address space. Optional; default `AddressSpace.gm`.
        - *`origin_shape`* (`int | tuple | None`): Logical origin. Optional;
          defaults to `shape`.
        - *`coord`* (`int | tuple | None`): Coordinate tree. Optional; derived
          from the layout when omitted (typically zeros).
        - *`assumed_align`* (`int | None`): Reserved; currently unused.

        Constraints:
        - `shape` / `stride` / `origin_shape` / `coord` are Python int trees,
          not Kernel `tla.make_shape` / `tla.make_stride` / `tla.make_coord`.
        - Always unbound; cannot be launched until replaced by `from_dlpack`.
        - Explicit `shape` / `stride` are kept as given (no layout remap).

        Example:
        ```python
        fa = make_fake_tensor(tla.Float16, (128, 64), (64, 1))
        fzn = make_fake_tensor(
            tla.Float16,
            ((16, 2), (16, 4)),
            ((16, 256), (1, 512)),
            layout_tag=tla.arch.zN,
            origin_shape=(32, 64),
        )
        ```
    """
    from ..runtime import _eager_capture

    if origin_shape is None:
        origin_shape = shape

    with _eager_capture():
        tensor = _Tensor(
            shape,
            dtype,
            addrspace=addrspace,
            data_ptr=0,
            origin_shape=origin_shape,
            coord=coord,
            stride=stride,
            layout_tag=layout_tag,
        )
    if assumed_align is not None:
        tensor._assumed_align = int(assumed_align)
    tensor.data_ptr = 0
    tensor._external_binding = False
    return tensor


__all__ = [
    "DlpackBridgeError",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
]
