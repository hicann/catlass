from __future__ import annotations


def test_star_import_includes_dynamic_op_helpers() -> None:
    namespace: dict[str, object] = {}
    exec("from catlass import *", namespace, namespace)
    assert "tile_view" in namespace
