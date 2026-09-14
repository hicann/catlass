// RUN: %tla_compile %s -o - | %filecheck %s

// Repeated sets of the same mode 2 flag from distinct pipes each emit their
// own INTRA.BLOCKI set with the respective pipe encoding (migrated from
// test_tla_lower_flag_barrier_to_hivm.py
// test_tla_sync_to_hivm_emits_repeated_sets_from_distinct_pipes).

module {
  tla.func @mode2_aiv_repeated_sets() {
    "tla.vector"() ({
      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<mte3>} : !tla.cross_flag<2>
    }) : () -> ()
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @mode2_aiv_repeated_sets_mix_aiv
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 5 : i64, sync_id = 0 : i64}>
// CHECK-NOT: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"()
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
