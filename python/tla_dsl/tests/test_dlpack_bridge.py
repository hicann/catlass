"""Unit tests for DLPack helpers in :mod:`catlass.runtime`."""

from __future__ import annotations

import ctypes
from typing import Any

import pytest

from catlass.base_dsl.runtime.dlpack_types import DLDataType, DLDataTypeCode, DLDevice, DLTensor
from catlass.runtime import DlpackBridgeError, _Tensor, export_dlpack_capsule

np = pytest.importorskip("numpy")


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("deleter", ctypes.c_void_p),
        ("manager_ctx", ctypes.c_void_p),
    ]


def _null_strides_dltensor_capsule() -> Any:
    """Build a ``dltensor`` capsule with ``strides == NULL`` (NumPy 2+ often exports explicit strides)."""
    storage = np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    shape_arr = (ctypes.c_int64 * 2)(2, 3)
    managed = _DLManagedTensor()
    managed.dl_tensor = DLTensor(
        data=ctypes.cast(storage.ctypes.data, ctypes.c_void_p),
        device=DLDevice(int(1), 0),
        ndim=2,
        dtype=DLDataType(int(DLDataTypeCode.kDLFloat), 32, 1),
        shape=ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64)),
        strides=None,
        byte_offset=0,
    )
    managed.deleter = None
    managed.manager_ctx = None
    # Keep backing arrays alive for the lifetime of the capsule consumer.
    managed._storage = storage  # type: ignore[attr-defined]
    managed._shape_arr = shape_arr  # type: ignore[attr-defined]
    ptr = ctypes.pointer(managed)
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.cast(ptr, ctypes.c_void_p),
        b"dltensor",
        None,
    )
    if not capsule:
        raise MemoryError("PyCapsule_New failed")
    return capsule


def test_numpy_dlpack_export_ignores_non_none_stream() -> None:
    arr = np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    export_dlpack_capsule(arr, stream=-1)


def test_parse_rejects_null_dlpack_strides() -> None:
    with pytest.raises(DlpackBridgeError, match="null strides"):
        _Tensor._parse_capsule(_null_strides_dltensor_capsule())


def test_parse_numpy_dlpack_explicit_strides() -> None:
    """NumPy exports explicit strides for C-contiguous buffers on recent versions."""
    arr = np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    parsed = _Tensor._parse_capsule(export_dlpack_capsule(arr))
    assert parsed["shape"] == (2, 3)
    assert parsed["strides"] == (3, 1)

    arr_f = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    parsed_f = _Tensor._parse_capsule(export_dlpack_capsule(arr_f))
    assert parsed_f["shape"] == (2, 3)
    assert parsed_f["strides"] == (1, 2)
