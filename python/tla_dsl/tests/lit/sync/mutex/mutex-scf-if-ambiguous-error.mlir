// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @mutex_if_dynamic_id() {
    "tla.cube"() ({
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cond = arith.cmpi slt, %c0, %c1 : index
      %mutex0 = tla.mutex "l1a0" {id = 1 : i64} -> !tla.mutex
      %mutex1 = tla.mutex "l1a1" {id = 2 : i64} -> !tla.mutex

      %chosen = scf.if %cond -> (!tla.mutex) {
        scf.yield %mutex0 : !tla.mutex
      } else {
        scf.yield %mutex1 : !tla.mutex
      }

      tla.mutex_lock %chosen [#tla.pipe<mte2>] : !tla.mutex
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @get_buf_mte2(i8)
// CHECK-LABEL: func.func @mutex_if_dynamic_id()
// CHECK-DAG: [[ID0:%.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK-DAG: [[ID1:%.*]] = llvm.mlir.constant(2 : i8) : i8
// CHECK: cf.cond_br {{%.*}}, ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: cf.br ^[[MERGE:bb[0-9]+]]([[ID0]] : i8)
// CHECK: ^[[ELSE]]:
// CHECK: cf.br ^[[MERGE]]([[ID1]] : i8)
// CHECK: ^[[MERGE]]([[BUFID:%.*]]: i8):
// CHECK: call @get_buf_mte2([[BUFID]])
// CHECK-NOT: !tla.mutex
// CHECK-NOT: tla.mutex
// CHECK: return
