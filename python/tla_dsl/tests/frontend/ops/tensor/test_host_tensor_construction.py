"""Host tensor construction: ``make_fake_tensor`` / ``from_dlpack``."""
from __future__ import annotations

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.core_api import _category
from catlass.tla.runtime import make_fake_tensor


def test_make_fake_tensor_host_sample() -> None:
    host = make_fake_tensor(
        tla.Float32,
        (4, 8),
        (8, 1),
        origin_shape=(4, 8),
        layout_tag=tla.arch.RowMajor,
    )
    assert isinstance(host, tla.Tensor)
    assert host.data_ptr == 0
    assert host._external_binding is False
    assert host.stride == (8, 1)
    types = host.__get_mlir_types__()
    assert len(types) == 1
    assert "tla.tensor" in str(types[0])


def test_make_fake_tensor_defaults_layout_tag_to_row_major() -> None:
    host = make_fake_tensor(tla.Float32, (4, 8), (8, 1))
    assert host.layout_tag == "RowMajor"
    assert host.origin_shape == (4, 8)


def test_make_fake_tensor_requires_stride() -> None:
    with pytest.raises(TypeError, match="stride"):
        make_fake_tensor(tla.Float32, (4, 8))  # type: ignore[call-arg]


def test_make_fake_tensor_zn_explicit_trees() -> None:
    host = make_fake_tensor(
        tla.Float16,
        ((16, 2), (16, 4)),
        ((16, 256), (1, 512)),
        layout_tag=tla.arch.zN,
        origin_shape=(32, 64),
        coord=(0, 0),
    )
    assert host.origin_shape == (32, 64)
    assert host.coord == (0, 0)
    assert host.shape != (32, 64)
    assert isinstance(host.shape[0], tuple)


def test_host_runtime_does_not_export_tensor_ctor() -> None:
    assert not hasattr(runtime_mod, "_Tensor")
    assert "_Tensor" not in getattr(runtime_mod, "__all__", ())
    assert not hasattr(tla, "_Tensor")


def test_host_make_shape_coord_stride_require_frontend() -> None:
    with pytest.raises(tla.TlaIRNotExecutableError):
        tla.make_shape(1, 2)
    with pytest.raises(tla.TlaIRNotExecutableError):
        tla.make_coord(0, 0)
    with pytest.raises(tla.TlaIRNotExecutableError):
        tla.make_stride(1, 1)


def test_host_tensor_tile_view_still_requires_kernel_context() -> None:
    mem = make_fake_tensor(tla.Float16, (1, 2), (2, 1), origin_shape=(1, 2))
    with pytest.raises(tla.TlaIRNotExecutableError):
        _ = tla.tile_view(mem, tla.make_shape(1, 2), tla.make_coord(0, 0))


def test_region_helpers_keep_region_stub_behavior() -> None:
    stub = tla.cube()
    assert stub.__class__.__name__ == "_RegionStub"
    assert _category(stub) == "region"
