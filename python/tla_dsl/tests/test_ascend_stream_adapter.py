from __future__ import annotations


import pytest

from catlass.base_dsl.runtime.ascend_stream_adapter import as_stream
from catlass.execution import TlaRuntimeUnavailableError


class _FakeTorchStream:
    def __init__(self, handle: int, device: int | None = None) -> None:
        self.npu_stream = handle
        if device is not None:
            self.device = device


def test_as_stream_accepts_int() -> None:
    assert as_stream(42, device=0) == 42


def test_as_stream_accepts_torch_like_stream() -> None:
    assert as_stream(_FakeTorchStream(99), device=0) == 99


def test_as_stream_rejects_stream_from_another_device() -> None:
    with pytest.raises(TlaRuntimeUnavailableError, match="stream belongs to device 2"):
        as_stream(_FakeTorchStream(99, device=2), device=1)
