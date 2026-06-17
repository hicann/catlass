// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @mutex_if_result() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cond = arith.cmpi slt, %c0, %c1 : index
    %mutex = tla.mutex "l1a0" {id = 7 : i64} -> !tla.mutex

    %chosen = scf.if %cond -> (!tla.mutex) {
      scf.yield %mutex : !tla.mutex
    } else {
      scf.yield %mutex : !tla.mutex
    }

    tla.mutex_lock %chosen [#tla.pipe<fix>] : !tla.mutex
    tla.return
  }
}

// CHECK-DAG: func.func private @get_buf_fix
// CHECK-LABEL: func.func @mutex_if_result()
// CHECK-DAG: [[ID:%.*]] = llvm.mlir.constant(7 : i8) : i8
// CHECK: cf.cond_br {{%.*}}, ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: cf.br ^[[MERGE:bb[0-9]+]]([[ID]] : i8)
// CHECK: ^[[ELSE]]:
// CHECK: cf.br ^[[MERGE]]([[ID]] : i8)
// CHECK: ^[[MERGE]]([[BUFID:%.*]]: i8):
// CHECK: call @get_buf_fix([[BUFID]])
// CHECK-NOT: !tla.mutex
// CHECK-NOT: tla.mutex
// CHECK: return
