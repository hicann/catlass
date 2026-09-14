// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @bidirectional_cross_flag() {
    %flag = tla.cross_flag "bidirectional" -> !tla.cross_flag<4>
    "tla.cube"() ({
      tla.cross_core_set_flag %flag {aiv_id = 0 : i64, pipe = #tla.pipe<mte3>} : !tla.cross_flag<4>
      tla.cross_core_set_flag %flag {aiv_id = 1 : i64, pipe = #tla.pipe<mte3>} : !tla.cross_flag<4>
      tla.cross_core_wait_flag %flag {aiv_id = 0 : i64, pipe = #tla.pipe<mte1>} : !tla.cross_flag<4>
      tla.cross_core_wait_flag %flag {aiv_id = 1 : i64, pipe = #tla.pipe<mte1>} : !tla.cross_flag<4>
    }) : () -> ()
    "tla.vector"() ({
      tla.cross_core_set_flag %flag {aiv_id = 0 : i64, pipe = #tla.pipe<fix>} : !tla.cross_flag<4>
      tla.cross_core_set_flag %flag {aiv_id = 1 : i64, pipe = #tla.pipe<fix>} : !tla.cross_flag<4>
      tla.cross_core_wait_flag %flag {aiv_id = 0 : i64, pipe = #tla.pipe<vector>} : !tla.cross_flag<4>
      tla.cross_core_wait_flag %flag {aiv_id = 1 : i64, pipe = #tla.pipe<vector>} : !tla.cross_flag<4>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @bidirectional_cross_flag_mix_aic
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 5 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 5 : i64, sync_id = 16 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 3 : i64, sync_id = 16 : i64}>
// CHECK-LABEL: func.func @bidirectional_cross_flag_mix_aiv
// CHECK: hivm.hir.get_sub_block_idx
// CHECK-COUNT-2: llvm.icmp "eq"
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 1 : i64, sync_id = 0 : i64}>
// CHECK-NOT: hivm.hir.sync_block
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
