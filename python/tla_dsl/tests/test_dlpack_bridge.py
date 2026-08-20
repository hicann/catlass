"""Unit tests for DLPack helpers in :mod:`catlass.tla.runtime`."""

from __future__ import annotations

import ctypes
import gc
import weakref
from typing import Any

import pytest

import catlass.tla as tla
from catlass.base_dsl.runtime.dlpack_types import (
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    DLManagedTensorDeleter,
)
from catlass.tla.runtime import from_dlpack
from catlass.types import RuntimeTensorError

np = pytest.importorskip("numpy")

# The capsule is passed as a raw PyObject* rather than ctypes.py_object: a
# destructor runs when the refcount has already reached zero, and py_object
# conversion would incref/decref it and free it a second time.
_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.c_void_p]

# ctypes trampolines and the structs the capsules point at must outlive the
# producer object: the deleter runs when the *consumer* drops the tensor, which
# in some tests is after the producer is gone. A real exporter has this for free
# (its deleter lives in the extension module), so pin them for the module's life.
_PINS: list[Any] = []
_DELETER_CALLS: dict[int, int] = {}


class _ManagedExport:
    """A minimal DLPack producer that behaves like a conforming C exporter.

    The capsule owns a real ``DLManagedTensor`` and carries a destructor that
    calls its deleter while the capsule is still named ``dltensor`` — exactly
    what a producer whose allocation is owned through ``manager_ctx`` does. The
    deleter records its calls in :attr:`deleter_calls`, so a test can tell
    "released" from "still valid" instead of assuming it.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (2, 3),
        strides: tuple[int, int] | None = (3, 1),
    ) -> None:
        self._shape = shape
        self._strides = strides
        # Identify this producer by value, so the callbacks below do not close
        # over `self` and keep it alive past the point a test drops it.
        self._id = len(_DELETER_CALLS)
        _DELETER_CALLS[self._id] = 0

    @property
    def deleter_calls(self) -> int:
        return _DELETER_CALLS[self._id]

    def _build_managed(self) -> Any:
        elems = np.ascontiguousarray(
            np.arange(int(np.prod(self._shape)), dtype=np.float32).reshape(self._shape)
        )
        extents = (ctypes.c_int64 * len(self._shape))(*self._shape)
        managed = DLManagedTensor()
        managed.dl_tensor.data = ctypes.cast(elems.ctypes.data, ctypes.c_void_p)
        managed.dl_tensor.device = DLDevice(int(DLDeviceType.kDLNpuCandidate1), 0)
        managed.dl_tensor.ndim = len(self._shape)
        managed.dl_tensor.dtype = DLDataType(int(DLDataTypeCode.kDLFloat), 32, 1)
        managed.dl_tensor.shape = ctypes.cast(extents, ctypes.POINTER(ctypes.c_int64))
        if self._strides is None:
            managed.dl_tensor.strides = None
        else:
            stride_buf = (ctypes.c_int64 * len(self._strides))(*self._strides)
            managed.dl_tensor.strides = ctypes.cast(
                stride_buf, ctypes.POINTER(ctypes.c_int64)
            )
            _PINS.append(stride_buf)
        managed.dl_tensor.byte_offset = 0

        producer_id = self._id

        def _deleter(_managed_ptr: Any) -> None:
            _DELETER_CALLS[producer_id] += 1

        deleter = DLManagedTensorDeleter(_deleter)
        managed.deleter = deleter
        managed.manager_ctx = None
        _PINS.extend([elems, extents, managed, deleter])
        return managed

    def __dlpack__(self, stream: int | None = None) -> Any:
        del stream
        managed = self._build_managed()

        producer_id = self._id

        def _destroy(capsule_ptr: int) -> None:
            # A conforming destructor releases only an unconsumed capsule; a
            # consumer that took ownership renames it to 'used_dltensor'.
            if ctypes.pythonapi.PyCapsule_GetName(capsule_ptr) == b"dltensor":
                _DELETER_CALLS[producer_id] += 1

        destructor = _CAPSULE_DESTRUCTOR(_destroy)
        create = ctypes.pythonapi.PyCapsule_New
        create.restype = ctypes.py_object
        # Intentionally leave argtypes unset so ctypes converts from ``c_void_p``.
        payload = ctypes.cast(ctypes.pointer(managed), ctypes.c_void_p)
        capsule = create(payload, b"dltensor", destructor)
        if capsule is None:
            raise MemoryError("failed to allocate dltensor capsule")
        _PINS.append(destructor)
        return capsule


def test_parse_rejects_null_dlpack_strides() -> None:
    exporter = _ManagedExport(strides=None)

    with pytest.raises(RuntimeTensorError, match="null strides"):
        from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    # A rejected capsule was never consumed, so the producer still owns it and
    # its destructor frees it — from_dlpack must not leak on the error path.
    gc.collect()
    assert exporter.deleter_calls == 1


def test_from_dlpack_defers_the_deleter_until_the_tensor_dies() -> None:
    """The tensor owns the DLManagedTensor: released on its death, not before."""
    exporter = _ManagedExport()
    tensor = from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    # from_dlpack's own capsule reference is gone by now. Had the capsule not
    # been consumed, its destructor would already have freed the buffer that
    # tensor.data_ptr points at.
    gc.collect()
    assert exporter.deleter_calls == 0

    del tensor
    gc.collect()
    assert exporter.deleter_calls == 1, "the tensor did not release its DLManagedTensor"


def test_from_dlpack_keeps_producer_alive() -> None:
    """Producers whose deleter is a no-op still own the buffer via the object."""
    exporter = _ManagedExport()
    alive = weakref.ref(exporter)
    tensor = from_dlpack(exporter, layout_tag=tla.arch.RowMajor)

    # Drop the caller's only reference, as `from_dlpack(x.contiguous().to(dev))` does.
    del exporter
    gc.collect()

    assert alive() is not None, "from_dlpack dropped the DLPack producer"
    assert tensor._dlpack_source is alive()

    del tensor
    gc.collect()
    assert alive() is None, "producer outlived the tensor that borrowed it"


def test_from_dlpack_rejects_an_already_consumed_capsule() -> None:
    """A DLPack capsule is single-use; the second consumer must be told, not fed a stale pointer."""
    exporter = _ManagedExport()
    capsule = exporter.__dlpack__()

    class _Replay:
        def __dlpack__(self, stream: int | None = None) -> Any:
            del stream
            return capsule

    replay = _Replay()
    tensor = from_dlpack(replay, layout_tag=tla.arch.RowMajor)
    with pytest.raises(RuntimeTensorError, match="consumed once"):
        from_dlpack(replay, layout_tag=tla.arch.RowMajor)

    del tensor
    gc.collect()
    assert exporter.deleter_calls == 1
