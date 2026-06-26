// RUN: %tla_compile %s -o - | %filecheck %s

module {

  tla.func @mutex_cube_pipe() {
    %mutex = tla.mutex "l0c_ping" {id = 3 : i64} -> !tla.mutex
    tla.mutex_lock %mutex [#tla.pipe<cube>] : !tla.mutex
    tla.mutex_unlock %mutex [#tla.pipe<cube>] : !tla.mutex
    tla.return
  }

  tla.func @mutex_roundtrip() {
    %mutex = tla.mutex "l0a_ping" {id = 7 : i64} -> !tla.mutex
    tla.mutex_lock %mutex [#tla.pipe<mte2>] : !tla.mutex
    tla.mutex_unlock %mutex [#tla.pipe<mte2>] : !tla.mutex
    tla.return
  }

  tla.func @mutex_vector_pipe_suffix() {
    %mutex = tla.mutex "ub_ping" {id = 6 : i64} -> !tla.mutex
    tla.mutex_lock %mutex [#tla.pipe<vector>] : !tla.mutex
    tla.mutex_unlock %mutex [#tla.pipe<vector>] : !tla.mutex
    tla.mutex_lock %mutex [#tla.pipe<mte3>] : !tla.mutex
    tla.mutex_unlock %mutex [#tla.pipe<mte3>] : !tla.mutex
    tla.return
  }
}

// CHECK-DAG: func.func private @get_buf_mte2
// CHECK-DAG: func.func private @rls_buf_mte2
// CHECK-DAG: func.func private @get_buf_mte3
// CHECK-DAG: func.func private @rls_buf_mte3
// CHECK-DAG: func.func private @get_buf_v
// CHECK-DAG: func.func private @rls_buf_v
// CHECK-DAG: func.func private @get_buf_m
// CHECK-DAG: func.func private @rls_buf_m
// CHECK-LABEL: func.func @mutex_cube_pipe()
// CHECK-DAG: [[CUBE_ID:%.*]] = llvm.mlir.constant(3 : i8) : i8
// CHECK: call @get_buf_m([[CUBE_ID]])
// CHECK: call @rls_buf_m([[CUBE_ID]])
// CHECK-LABEL: func.func @mutex_roundtrip()
// CHECK-DAG: [[ID:%.*]] = llvm.mlir.constant(7 : i8) : i8
// CHECK: call @get_buf_mte2([[ID]])
// CHECK: call @rls_buf_mte2([[ID]])
// CHECK-LABEL: func.func @mutex_vector_pipe_suffix()
// CHECK-DAG: [[VECTOR_ID:%.*]] = llvm.mlir.constant(6 : i8) : i8
// CHECK: call @get_buf_v([[VECTOR_ID]])
// CHECK: call @rls_buf_v([[VECTOR_ID]])
// CHECK: call @get_buf_mte3([[VECTOR_ID]])
// CHECK: call @rls_buf_mte3([[VECTOR_ID]])
// CHECK-NOT: tla.mutex
// CHECK: return
