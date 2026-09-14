// RUN: %tla_compile %s -o - | %filecheck %s

// Dynamic (scf.if selected) cross flags lowered inside the cube region:
// mode 2 duplicates the register-form set across both cube halves, and
// mode 4 with aiv_id = 1 offsets its single register-form set by 16.
// Flag names are module-global, so they differ from the vector-side files.
// (Migrated from test_tla_lower_flag_barrier_to_hivm.py
// test_tla_sync_to_hivm_duplicates_dynamic_mode2_on_aic and
// test_tla_sync_to_hivm_offsets_dynamic_mode4_aiv1_on_aic.)

module {
  tla.func @dynamic_mode2_cube_duplicate() {
    "tla.cube"() ({
      %condition = arith.constant true
      %m2a = tla.cross_flag "m2a" -> !tla.cross_flag<2>
      %m2b = tla.cross_flag "m2b" -> !tla.cross_flag<2>
      %selected = scf.if %condition -> (!tla.cross_flag<2>) {
        scf.yield %m2a : !tla.cross_flag<2>
      } else {
        scf.yield %m2b : !tla.cross_flag<2>
      }
      tla.cross_core_set_flag %selected {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
    }) : () -> ()
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }

  tla.func @dynamic_mode4_aiv1_cube() {
    "tla.cube"() ({
      %condition = arith.constant true
      %m4a = tla.cross_flag "m4a" -> !tla.cross_flag<4>
      %m4b = tla.cross_flag "m4b" -> !tla.cross_flag<4>
      %selected = scf.if %condition -> (!tla.cross_flag<4>) {
        scf.yield %m4a : !tla.cross_flag<4>
      } else {
        scf.yield %m4b : !tla.cross_flag<4>
      }
      tla.cross_core_set_flag %selected {aiv_id = 1 : i64, pipe = #tla.pipe<fix>} : !tla.cross_flag<4>
    }) : () -> ()
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @dynamic_mode2_cube_duplicate_mix_aic
// CHECK: llvm.mlir.constant(16 : i64)
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(
// CHECK: llvm.add
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(
// CHECK-NOT: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core

// CHECK-LABEL: func.func @dynamic_mode4_aiv1_cube_mix_aic
// CHECK: llvm.mlir.constant(16 : i64)
// CHECK: llvm.add
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(
// CHECK-NOT: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"(
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
