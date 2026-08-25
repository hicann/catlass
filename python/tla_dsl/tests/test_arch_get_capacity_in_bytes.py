from __future__ import annotations

import pytest

import catlass.tla as tla


def test_get_capacity_in_bytes_host_side_accepts_arch_tokens() -> None:
    assert tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1) == 512 * 1024
    assert tla.arch.get_capacity_in_bytes(tla.AddressSpace.l0a) == 64 * 1024
    assert tla.arch.get_capacity_in_bytes(tla.AddressSpace.l0b) == 64 * 1024
    assert tla.arch.get_capacity_in_bytes(tla.AddressSpace.l0c) == 256 * 1024
    assert tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub) == 248 * 1024


def test_get_capacity_in_bytes_rejects_non_token_input() -> None:
    for bad in ("gm", tla.AddressSpace.gm, 1024, None):
        with pytest.raises(tla.TlaCoreAPIError, match="tla.arch memory-scope token"):
            tla.arch.get_capacity_in_bytes(bad)


@tla.kernel
def capacity_in_bytes_range_kernel() -> None:
    ub_blocks = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub) // (62 * 1024)
    for i in tla.range(ub_blocks):
        tla.make_coord(i, 0)


def test_get_capacity_in_bytes_kernel_side_folds_to_constant() -> None:
    mlir = capacity_in_bytes_range_kernel.dump_mlir()
    assert "tla.range" in mlir or "scf.for" in mlir
    # 248 KiB UB / 62 KiB = 4 iterations, folded as a constant bound.
    assert "%c4_i32" in mlir
    assert "tla.get_capacity_in_bytes" not in mlir  # no runtime op emitted
