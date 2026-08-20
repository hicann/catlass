"""Unit tests for DLPack helpers in :mod:`catlass.tla.runtime`."""

from __future__ import annotations

import ctypes
from typing import Any

import pytest

import catlass.tla as tla
from catlass.base_dsl.runtime.dlpack_types import DLDataType, DLDataTypeCode, DLDevice, DLTensor
from catlass.tla.runtime import from_dlpack
from catlass.types import RuntimeTensorError

np = pytest.importorskip("numpy")

# Keep ctypes-backed objects alive for as long as tests hold the capsule.
_CAPSULE_PINS: list[Any] = []


def _wrap_as_dltensor_capsule(dl: DLTensor, *, keep: list[Any]) -> Any:
    """Put a ``DLTensor`` behind a ``dltensor`` capsule (parse path only reads that header)."""
    create = ctypes.pythonapi.PyCapsule_New
    create.restype = ctypes.py_object
    # Intentionally leave argtypes unset so ctypes converts from ``c_void_p``.
    payload = ctypes.cast(ctypes.pointer(dl), ctypes.c_void_p)
    capsule = create(payload, b"dltensor", None)
    if capsule is None:
        raise MemoryError("failed to allocate dltensor capsule")
    _CAPSULE_PINS.append((capsule, tuple(keep)))
    return capsule


def _null_stride_capsule() -> Any:
    """Host-only fixture: rank-2 DLTensor whose stride pointer is NULL."""
    elems = np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    extents = (ctypes.c_int64 * 2)(2, 3)
    header = DLTensor()
    header.data = ctypes.cast(elems.ctypes.data, ctypes.c_void_p)
    header.device = DLDevice(1, 0)
    header.ndim = 2
    header.dtype = DLDataType(int(DLDataTypeCode.kDLFloat), 32, 1)
    header.shape = ctypes.cast(extents, ctypes.POINTER(ctypes.c_int64))
    header.strides = None
    header.byte_offset = 0
    return _wrap_as_dltensor_capsule(header, keep=[elems, extents, header])


def test_parse_rejects_null_dlpack_strides() -> None:
    class _Exporter:
        def __dlpack__(self, stream: int | None = None):
            del stream
            return _null_stride_capsule()

    with pytest.raises(RuntimeTensorError, match="null strides"):
        from_dlpack(_Exporter(), layout_tag=tla.arch.RowMajor)
