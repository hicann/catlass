// RUN: %tla_compile %s -o - | %filecheck %s

// Mode 2 set/wait inside the cube region is duplicated across both cube
// halves: sync_id 0 and 16 (migrated from test_tla_lower_flag_barrier_to_hivm.py
// test_tla_sync_to_hivm_duplicates_static_mode2_on_aic).

module {
  tla.func @mode2_cube_duplicate() {
    "tla.cube"() ({
      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<mte3>} : !tla.cross_flag<2>
      tla.cross_core_wait_flag %flag {pipe = #tla.pipe<mte1>} : !tla.cross_flag<2>
    }) : () -> ()
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @mode2_cube_duplicate_mix_aic
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 5 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 5 : i64, sync_id = 16 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
