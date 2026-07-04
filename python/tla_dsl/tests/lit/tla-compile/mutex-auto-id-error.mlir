// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

module {
  tla.func @mutex_auto_id_error() {
    "tla.cube"() ({
      %mutex = tla.mutex "l0a_ping" {id = -1 : i64} -> !tla.mutex
      tla.mutex_lock %mutex [#tla.pipe<mte2>] : !tla.mutex
    }) : () -> ()
    tla.return
  }
}

// CHECK: mutex id auto allocation is not implemented for bitcode call lowering
