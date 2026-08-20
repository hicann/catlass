from __future__ import annotations

import pytest

import catlass.tla as tla


@tla.kernel
def manual_sync_kernel() -> None:
    pass


@tla.kernel(auto_sync="v0")
def auto_sync_kernel() -> None:
    pass


@tla.kernel()
def explicit_manual_sync_kernel() -> None:
    pass


def test_kernel_auto_sync_defaults_to_disabled() -> None:
    assert manual_sync_kernel.options == {}
    assert "tla.auto_sync" not in manual_sync_kernel.dump_mlir()


def test_kernel_empty_call_preserves_manual_sync_ir() -> None:
    assert explicit_manual_sync_kernel.options == {}
    assert "tla.auto_sync" not in explicit_manual_sync_kernel.dump_mlir()


def test_kernel_auto_sync_v0_propagates_to_tla_func() -> None:
    assert auto_sync_kernel.options == {"auto_sync": "v0"}
    mlir = auto_sync_kernel.dump_mlir()
    assert 'tla.auto_sync = "v0"' in mlir


@pytest.mark.parametrize("value", ["", "auto", "V0", "v1", 1])
def test_kernel_rejects_invalid_auto_sync_version(value: object) -> None:
    with pytest.raises(ValueError, match="auto_sync must be 'v0' or None"):
        tla.kernel(auto_sync=value)  # type: ignore[arg-type]
