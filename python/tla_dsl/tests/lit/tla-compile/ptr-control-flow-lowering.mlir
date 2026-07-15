// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-lower-ptr 2>&1 | %filecheck %s

// !tla.ptr is a first-class SSA type before tla-lower-ptr. The lowering must
// convert every structural carrier together: nested/multi-result scf.if,
// scf.for/while loop-carried values, and function call/signature boundaries.
// An alloc nested in a loop denotes one static kernel-lifetime slot, so its
// lowered address is the same constant on every dynamic iteration.

module {
  func.func @ptr_control_flow(%cond: i1, %lb: index, %ub: index, %step: index) {
    %seed = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f32, l1, 512>
    %selected, %tag = scf.if %cond -> (!tla.ptr<f32, l1, 512>, index) {
      %offset = "tla.ptr_add"(%seed, %lb) : (!tla.ptr<f32, l1, 512>, index) -> !tla.ptr<f32, l1, 512>
      scf.yield %offset, %lb : !tla.ptr<f32, l1, 512>, index
    } else {
      %branch_slot = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f32, l1, 512>
      %nested = scf.if %cond -> (!tla.ptr<f32, l1, 512>) {
        scf.yield %branch_slot : !tla.ptr<f32, l1, 512>
      } else {
        scf.yield %seed : !tla.ptr<f32, l1, 512>
      }
      scf.yield %nested, %ub : !tla.ptr<f32, l1, 512>, index
    }
    %loop_result = scf.for %i = %lb to %ub step %step iter_args(%carried = %selected) -> (!tla.ptr<f32, l1, 512>) {
      %loop_slot = "tla.alloc_ptr"() {size_bytes = 512 : i64} : () -> !tla.ptr<f32, l1, 512>
      %next = scf.if %cond -> (!tla.ptr<f32, l1, 512>) {
        scf.yield %loop_slot : !tla.ptr<f32, l1, 512>
      } else {
        scf.yield %carried : !tla.ptr<f32, l1, 512>
      }
      scf.yield %next : !tla.ptr<f32, l1, 512>
    }
    %false = arith.constant false
    %while_result = scf.while (%carried = %loop_result) : (!tla.ptr<f32, l1, 512>) -> !tla.ptr<f32, l1, 512> {
      scf.condition(%false) %carried : !tla.ptr<f32, l1, 512>
    } do {
    ^bb0(%body_ptr: !tla.ptr<f32, l1, 512>):
      %next = func.call @ptr_identity(%body_ptr) : (!tla.ptr<f32, l1, 512>) -> !tla.ptr<f32, l1, 512>
      scf.yield %next : !tla.ptr<f32, l1, 512>
    }
    func.call @ptr_sink(%while_result, %tag) : (!tla.ptr<f32, l1, 512>, index) -> ()
    return
  }
  func.func @ptr_cfg(%cond: i1, %input: !tla.ptr<f32, l1, 512>) {
    %c0 = arith.constant 0 : index
    cf.cond_br %cond, ^bb1(%input : !tla.ptr<f32, l1, 512>),
                      ^bb2(%input : !tla.ptr<f32, l1, 512>)
  ^bb1(%lhs: !tla.ptr<f32, l1, 512>):
    cf.br ^bb3(%lhs : !tla.ptr<f32, l1, 512>)
  ^bb2(%rhs: !tla.ptr<f32, l1, 512>):
    cf.br ^bb3(%rhs : !tla.ptr<f32, l1, 512>)
  ^bb3(%merged: !tla.ptr<f32, l1, 512>):
    func.call @ptr_sink(%merged, %c0) : (!tla.ptr<f32, l1, 512>, index) -> ()
    return
  }
  func.func private @ptr_identity(!tla.ptr<f32, l1, 512>) -> !tla.ptr<f32, l1, 512>
  func.func private @ptr_sink(!tla.ptr<f32, l1, 512>, index)
}

// CHECK-LABEL: func.func @ptr_control_flow(
// CHECK: %[[SEED:.*]] = arith.constant {{.*}}0 : i64
// CHECK: scf.if {{.*}} -> (i64, index)
// CHECK: arith.index_cast {{.*}} : index to i64
// CHECK: arith.constant 4 : i64
// CHECK: arith.muli {{.*}} : i64
// CHECK: arith.addi %[[SEED]], {{.*}} : i64
// CHECK: arith.constant {{.*}}512 : i64
// CHECK: scf.if {{.*}} -> (i64)
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i64)
// CHECK: arith.constant {{.*}}1024 : i64
// CHECK: scf.while {{.*}} : (i64) -> i64
// CHECK: func.call @ptr_identity({{.*}}) : (i64) -> i64
// CHECK: call @ptr_sink({{.*}}) : (i64, index) -> ()
// CHECK-NOT: !tla.ptr
// CHECK-NOT: tla.alloc_ptr
// CHECK-NOT: tla.ptr_add
// CHECK: func.func @ptr_cfg({{.*}}i64)
// CHECK: cf.cond_br {{.*}} i64), {{.*}} i64)
// CHECK: cf.br {{.*}} i64)
// CHECK: func.func private @ptr_identity(i64) -> i64
// CHECK: func.func private @ptr_sink(i64, index)
