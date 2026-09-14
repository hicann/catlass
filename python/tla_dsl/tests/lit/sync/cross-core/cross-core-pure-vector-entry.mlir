// RUN: %tla_compile %s -o - | %filecheck %s

// A pure-vector entry retains hivm.func_core_type = AIV (it is no longer
// stripped by tla-lower-func), so a cross_flag inside it lowers directly
// instead of erroring on a missing core type (migrated from
// test_tla_lower_flag_barrier_to_hivm.py
// test_tla_sync_to_hivm_lowers_cross_flag_in_pure_vector_entry).

module {
  tla.func @pure_vector_entry() {
    "tla.vector"() ({
      %flag = tla.cross_flag "flag" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %flag {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @pure_vector_entry
// CHECK: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"()
// CHECK-NOT: hivm.hir.sync_block_set
// CHECK-NOT: hivm.hir.sync_block_wait
// CHECK-NOT: tla.cross_flag
// CHECK-NOT: tla.cross_core
