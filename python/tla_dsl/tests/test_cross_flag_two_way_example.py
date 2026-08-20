from examples.end_to_end.cross_flag_two_way.cross_flag_two_way import dump_tlair


def test_two_way_cross_flag_example_emits_complete_handshake() -> None:
    mlir = dump_tlair()

    assert mlir.count('tla.cross_flag "aic_to_aiv" -> !tla.cross_flag<4>') == 1
    assert mlir.count('tla.cross_flag "aiv_to_aic" -> !tla.cross_flag<4>') == 1
    assert mlir.count("tla.cross_core_set_flag") == 6
    assert mlir.count("tla.cross_core_wait_flag") == 6
    assert mlir.count("aiv_id = 0 : i64") == 6
    assert mlir.count("aiv_id = 1 : i64") == 6
