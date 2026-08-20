// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-mutex-to-std -o %t 2>&1 | %filecheck %s

// !tla.mutex is a first-class SSA type before tla-lower-mutex-to-std. The
// lowering must convert every structural carrier together instead of rebuilding
// a selector independently at each lock/unlock use.

module {
  func.func @mutex_control_flow(%cond: i1, %lb: index, %ub: index, %step: index) {
    %mutex0 = tla.mutex "ub0" {id = 3 : i64} -> !tla.mutex
    %mutex1 = tla.mutex "ub1" {id = 4 : i64} -> !tla.mutex
    %selected = scf.if %cond -> (!tla.mutex) {
      scf.yield %mutex0 : !tla.mutex
    } else {
      scf.yield %mutex1 : !tla.mutex
    }
    %loop_result = scf.for %i = %lb to %ub step %step
        iter_args(%carried = %selected) -> (!tla.mutex) {
      %next = scf.if %cond -> (!tla.mutex) {
        scf.yield %mutex1 : !tla.mutex
      } else {
        scf.yield %carried : !tla.mutex
      }
      scf.yield %next : !tla.mutex
    }
    %false = arith.constant false
    %while_result = scf.while (%carried = %loop_result) : (!tla.mutex) -> !tla.mutex {
      scf.condition(%false) %carried : !tla.mutex
    } do {
    ^bb0(%body_mutex: !tla.mutex):
      %next = func.call @mutex_identity(%body_mutex) : (!tla.mutex) -> !tla.mutex
      scf.yield %next : !tla.mutex
    }
    tla.mutex_lock %while_result [#tla.pipe<mte2>] : !tla.mutex
    func.call @mutex_sink(%while_result) : (!tla.mutex) -> ()
    return
  }

  func.func @mutex_cfg(%cond: i1, %lhs_input: !tla.mutex,
                       %rhs_input: !tla.mutex) {
    cf.cond_br %cond, ^bb1(%lhs_input : !tla.mutex),
                      ^bb2(%rhs_input : !tla.mutex)
  ^bb1(%lhs: !tla.mutex):
    cf.br ^bb3(%lhs : !tla.mutex)
  ^bb2(%rhs: !tla.mutex):
    cf.br ^bb3(%rhs : !tla.mutex)
  ^bb3(%merged: !tla.mutex):
    func.call @mutex_sink(%merged) : (!tla.mutex) -> ()
    return
  }

  func.func private @mutex_identity(!tla.mutex) -> !tla.mutex
  func.func private @mutex_sink(!tla.mutex)
}

// CHECK-LABEL: func.func @mutex_control_flow(
// CHECK-DAG: %[[ID0:.*]] = arith.constant 3 : i8
// CHECK-DAG: %[[ID1:.*]] = arith.constant 4 : i8
// CHECK: scf.if {{.*}} -> (i8)
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i8)
// CHECK: scf.if {{.*}} -> (i8)
// CHECK: scf.while {{.*}} : (i8) -> i8
// CHECK: func.call @mutex_identity({{.*}}) : (i8) -> i8
// CHECK: call @get_buf_mte2({{.*}}) : (i8) -> ()
// CHECK: call @mutex_sink({{.*}}) : (i8) -> ()
// CHECK-NOT: !tla.mutex

// CHECK-LABEL: func.func @mutex_cfg({{.*}}i8)
// CHECK: cf.cond_br {{.*}} i8), {{.*}} i8)
// CHECK: ^{{bb[0-9]+}}({{.*}}: i8):
// CHECK: call @mutex_sink({{.*}}) : (i8) -> ()
// CHECK: func.func private @mutex_identity(i8) -> i8
// CHECK: func.func private @mutex_sink(i8)
