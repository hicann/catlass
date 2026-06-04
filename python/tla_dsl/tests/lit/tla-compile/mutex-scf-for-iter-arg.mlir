// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @mutex_for_iter_arg() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %mutex = tla.mutex "l1a0" {id = 5 : i64} -> !tla.mutex

    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%m = %mutex) -> (!tla.mutex) {
      tla.mutex_lock %m [#tla.pipe<mte2>] : !tla.mutex
      tla.mutex_unlock %m [#tla.pipe<mte2>] : !tla.mutex
      scf.yield %m : !tla.mutex
    }

    tla.mutex_lock %result [#tla.pipe<mte2>] : !tla.mutex
    tla.return
  }
}

// CHECK-DAG: func.func private @get_buf_mte2(i8)
// CHECK-DAG: func.func private @rls_buf_mte2(i8)
// CHECK-LABEL: func.func @mutex_for_iter_arg()
// CHECK-DAG: [[ID:%.*]] = arith.constant 5 : i8
// CHECK: call @get_buf_mte2([[ID]])
// CHECK: call @rls_buf_mte2([[ID]])
// CHECK: call @get_buf_mte2([[ID]])
// CHECK-NOT: !tla.mutex
// CHECK-NOT: tla.mutex
// CHECK: return
