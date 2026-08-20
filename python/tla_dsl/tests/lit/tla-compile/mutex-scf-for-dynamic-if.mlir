// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @mutex_for_iter_arg_dynamic_if() {
    "tla.cube"() ({
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %cond = arith.cmpi slt, %c0, %c1 : index
      %mutex0 = tla.mutex "l1a0" {id = 4 : i64} -> !tla.mutex
      %mutex1 = tla.mutex "l1a1" {id = 5 : i64} -> !tla.mutex

      %chosen = scf.if %cond -> (!tla.mutex) {
        scf.yield %mutex0 : !tla.mutex
      } else {
        scf.yield %mutex1 : !tla.mutex
      }

      %result = scf.for %i = %c0 to %c2 step %c1
          iter_args(%m = %chosen) -> (!tla.mutex) {
        tla.mutex_lock %m [#tla.pipe<mte1>] : !tla.mutex
        tla.mutex_unlock %m [#tla.pipe<mte1>] : !tla.mutex
        scf.yield %m : !tla.mutex
      }

      tla.mutex_lock %result [#tla.pipe<cube>] : !tla.mutex
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @get_buf_mte1(i8)
// CHECK-DAG: func.func private @rls_buf_mte1(i8)
// CHECK-DAG: func.func private @get_buf_m(i8)
// CHECK-LABEL: func.func @mutex_for_iter_arg_dynamic_if()
// CHECK-DAG: [[ID0:%.*]] = llvm.mlir.constant(4 : i8) : i8
// CHECK-DAG: [[ID1:%.*]] = llvm.mlir.constant(5 : i8) : i8
// CHECK: cf.cond_br {{%.*}}, ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: cf.br ^[[MERGE:bb[0-9]+]]([[ID0]] : i8)
// CHECK: ^[[ELSE]]:
// CHECK: cf.br ^[[MERGE]]([[ID1]] : i8)
// CHECK: ^[[MERGE]]([[BUFID:%.*]]: i8):
// CHECK: call @get_buf_mte1([[BUFID]])
// CHECK: call @rls_buf_mte1([[BUFID]])
// CHECK: call @get_buf_m([[BUFID]])
// CHECK-NOT: !tla.mutex
// CHECK-NOT: tla.mutex
// CHECK: return
