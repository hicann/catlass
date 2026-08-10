from __future__ import annotations

import catlass.tla as tla


def test_tla_namespace_exposes_dynamic_op_helpers() -> None:
    assert callable(tla.tile_view)
    assert callable(tla.make_shape)
    assert callable(tla.kernel)
