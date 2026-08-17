from __future__ import annotations

import pytest

arch_mod = pytest.importorskip("catlass.base_dsl.arch", exc_type=ImportError)


def test_resolve_npu_arch_maps_public_chip_name() -> None:
    assert arch_mod.resolve_npu_arch("3510") == "c310"
    assert arch_mod.Arch.from_string("3510") is arch_mod.Arch.C310


def test_arch_scope_for_target_resolves_supported_pairs() -> None:
    assert (
        arch_mod.arch_scope_for_target(target_arch="c310", core_type="aiv")
        == "aiv.c310"
    )
    assert (
        arch_mod.arch_scope_for_target(target_arch="c310", core_type="aic")
        == "aic.c310"
    )


def test_parse_arch_scope_round_trips_supported_scopes() -> None:
    assert arch_mod.parse_arch_scope("aiv.c310") == ("c310", "aiv")
    assert arch_mod.parse_arch_scope("aic.c310") == ("c310", "aic")


def test_get_localmem_capacity_bytes_accepts_arch_override() -> None:
    assert arch_mod.get_localmem_capacity_bytes("L1", arch="c310") == 512 * 1024
    assert arch_mod.get_localmem_capacity_bytes("UB", arch="c310") == 248 * 1024


def test_c220_is_no_longer_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported (--npu-arch|target architecture)"):
        arch_mod.arch_scope_for_target(target_arch="c220", core_type="aiv")
