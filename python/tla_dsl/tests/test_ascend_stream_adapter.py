from __future__ import annotations


from catlass.base_dsl.runtime.ascend_stream_adapter import as_stream


class _FakeTorchStream:
    def __init__(self, handle: int) -> None:
        self.npu_stream = handle


def test_as_stream_accepts_int() -> None:
    assert as_stream(42, device=0) == 42


def test_as_stream_accepts_torch_like_stream() -> None:
    assert as_stream(_FakeTorchStream(99), device=0) == 99
