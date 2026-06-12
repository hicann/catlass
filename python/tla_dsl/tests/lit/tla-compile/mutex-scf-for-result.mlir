// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @mutex_for_result() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %mutex = tla.mutex "l1a1" {id = 6 : i64} -> !tla.mutex

    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%m = %mutex) -> (!tla.mutex) {
      scf.yield %m : !tla.mutex
    }

    tla.mutex_unlock %result [#tla.pipe<mte3>] : !tla.mutex
    tla.return
  }
}

// CHECK-DAG: func.func private @rls_buf_mte3
// CHECK-LABEL: func.func @mutex_for_result()
// CHECK-DAG: [[ID:%.*]] = llvm.mlir.constant(6 : i8) : i8
// CHECK: call @rls_buf_mte3([[ID]])
// CHECK-NOT: !tla.mutex
// CHECK-NOT: tla.mutex
// CHECK: return
