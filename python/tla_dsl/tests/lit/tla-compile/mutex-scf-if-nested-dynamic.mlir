// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @mutex_if_nested_dynamic_id() {
    "tla.cube"() ({
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %outer_cond = arith.cmpi slt, %c0, %c1 : index
      %mutex0 = tla.mutex "ub0" {id = 3 : i64} -> !tla.mutex
      %mutex1 = tla.mutex "ub1" {id = 4 : i64} -> !tla.mutex
      %mutex2 = tla.mutex "ub2" {id = 5 : i64} -> !tla.mutex

      %chosen = scf.if %outer_cond -> (!tla.mutex) {
        %inner_cond = arith.cmpi eq, %c0, %c1 : index
        %chosen01 = scf.if %inner_cond -> (!tla.mutex) {
          scf.yield %mutex0 : !tla.mutex
        } else {
          scf.yield %mutex1 : !tla.mutex
        }
        scf.yield %chosen01 : !tla.mutex
      } else {
        scf.yield %mutex2 : !tla.mutex
      }

      tla.mutex_lock %chosen [#tla.pipe<mte2>] : !tla.mutex
      tla.mutex_unlock %chosen [#tla.pipe<mte2>] : !tla.mutex
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @get_buf_mte2(i8)
// CHECK-DAG: func.func private @rls_buf_mte2(i8)
// CHECK-LABEL: func.func @mutex_if_nested_dynamic_id()
// CHECK-DAG: llvm.mlir.constant(3 : i8) : i8
// CHECK-DAG: llvm.mlir.constant(4 : i8) : i8
// CHECK-DAG: llvm.mlir.constant(5 : i8) : i8
// CHECK-COUNT-2: cf.cond_br
// CHECK: call @get_buf_mte2({{%.*}})
// CHECK: call @rls_buf_mte2({{%.*}})
// CHECK-NOT: !tla.mutex
// CHECK-NOT: tla.mutex
// CHECK: return
