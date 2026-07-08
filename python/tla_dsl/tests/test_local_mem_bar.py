from __future__ import annotations

import pytest

import catlass as tla


@tla.kernel
def local_mem_bar_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            tla.local_mem_bar(tla.params.MemType.VEC_LOAD, tla.params.MemType.VEC_STORE)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_STORE)
            tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.SCALAR_ALL)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.SCALAR_LOAD)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.SCALAR_STORE)
            tla.local_mem_bar(tla.params.MemType.VEC_LOAD, tla.params.MemType.SCALAR_STORE)
            tla.local_mem_bar(tla.params.MemType.SCALAR_ALL, tla.params.MemType.VEC_ALL)
            tla.local_mem_bar(tla.params.MemType.SCALAR_STORE, tla.params.MemType.VEC_LOAD)
            tla.local_mem_bar(tla.params.MemType.SCALAR_STORE, tla.params.MemType.VEC_STORE)
            tla.local_mem_bar(tla.params.MemType.SCALAR_LOAD, tla.params.MemType.VEC_STORE)


@tla.kernel
def local_mem_bar_unsupported_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_LOAD, tla.params.MemType.SCALAR_LOAD)


@tla.kernel
def local_mem_bar_outside_vector_kernel() -> None:
    tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL)


def test_local_mem_bar_emits_tla_op() -> None:
    mlir = local_mem_bar_kernel.dump_mlir()
    assert "tla.local_mem_bar" in mlir
    assert "tla.local_mem_bar 0" in mlir  # VEC_ALL × VEC_ALL → 0
    assert "tla.local_mem_bar 1" in mlir  # VEC_STORE × VEC_LOAD → 1
    assert "tla.local_mem_bar 2" in mlir  # VEC_LOAD × VEC_STORE → 2
    assert "tla.local_mem_bar 3" in mlir  # VEC_STORE × VEC_STORE → 3
    assert "tla.local_mem_bar 4" in mlir  # VEC_ALL × SCALAR_ALL → 4
    assert "tla.local_mem_bar 5" in mlir  # VEC_STORE × SCALAR_LOAD → 5
    assert "tla.local_mem_bar 6" in mlir  # VEC_LOAD × SCALAR_STORE → 6
    assert "tla.local_mem_bar 7" in mlir  # VEC_STORE × SCALAR_STORE → 7
    assert "tla.local_mem_bar 8" in mlir  # SCALAR_ALL × VEC_ALL → 8
    assert "tla.local_mem_bar 9" in mlir  # SCALAR_STORE × VEC_LOAD → 9
    assert "tla.local_mem_bar 10" in mlir # SCALAR_LOAD × VEC_STORE → 10
    assert "tla.local_mem_bar 11" in mlir # SCALAR_STORE × VEC_STORE → 11


def test_local_mem_bar_rejects_unsupported_pair() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported"):
        local_mem_bar_unsupported_kernel.dump_mlir()


def test_local_mem_bar_requires_vector_region() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError, match="must be nested inside tla.vec.func"
    ):
        local_mem_bar_outside_vector_kernel.dump_mlir()
