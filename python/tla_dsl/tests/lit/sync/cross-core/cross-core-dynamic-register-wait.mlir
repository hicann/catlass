// RUN: %tla_compile %s -o - | %filecheck %s

// A cross_flag selected through scf.if loses its static id, so mode 0 lowers
// to the register form of the device wait intrinsic with shift/or id math
// (migrated from test_tla_lower_flag_barrier_to_hivm.py
// test_tla_sync_to_hivm_lowers_dynamic_id_to_register_form, mode 0 branch).

module {
  tla.func @dynamic_mode0_register_wait() {
    "tla.vector"() ({
      %condition = arith.constant true
      %a = tla.cross_flag "a" -> !tla.cross_flag<0>
      %b = tla.cross_flag "b" -> !tla.cross_flag<0>
      %selected = scf.if %condition -> (!tla.cross_flag<0>) {
        scf.yield %a : !tla.cross_flag<0>
      } else {
        scf.yield %b : !tla.cross_flag<0>
      }
      tla.cross_core_set_flag %selected {pipe = #tla.pipe<fix>} : !tla.cross_flag<0>
      tla.cross_core_wait_flag %selected {pipe = #tla.pipe<vector>} : !tla.cross_flag<0>
    }) : () -> ()
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @dynamic_mode0_register_wait_mix_aiv
// CHECK: llvm.shl
// CHECK: llvm.or
// CHECK: "hivm.intr.hivm.SET.CROSS.CORE"(
// CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.REG"(
// CHECK-NOT: WAIT.FLAG.DEV.PIPE.IMM
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
