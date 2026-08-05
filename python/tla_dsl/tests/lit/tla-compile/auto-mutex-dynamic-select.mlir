// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s

!layout = !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>
!coord = !tla.coord<0>
!ub = !tla.tensor<!layout, !coord, !tla.ptr<f32, ub, 256>>
!gm = !tla.tensor<!layout, !coord, !tla.ptr<f32, gm, 4>>

module {
  tla.func @auto_mutex_dynamic_select(%cond0: i1, %cond1: i1, %gm: !gm) attributes {tla.auto_sync = "v0"} {
    %shape = tla.make_shape -> !tla.shape<64>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<64>, !tla.stride<1> -> !layout
    %coord = tla.make_coord -> !coord
    %p0 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %p1 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %p2 = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %selected012 = scf.if %cond1 -> (!tla.ptr<f32, ub, 256>) {
      %false = arith.constant false
      %inner_cond = arith.xori %cond0, %false : i1
      %selected01 = scf.if %inner_cond -> (!tla.ptr<f32, ub, 256>) {
        scf.yield %p0 : !tla.ptr<f32, ub, 256>
      } else {
        scf.yield %p1 : !tla.ptr<f32, ub, 256>
      }
      scf.yield %selected01 : !tla.ptr<f32, ub, 256>
    } else {
      scf.yield %p2 : !tla.ptr<f32, ub, 256>
    }
    %local = tla.make_tensor %selected012, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !ub
    "tla.vector"() ({
      tla.copy %local, %gm : !ub, !gm
    }) : () -> ()
    tla.return
  }
}

// ID assignment is by UB base: p0=0, p1=1, p2=2.
// CHECK-LABEL: func.func @auto_mutex_dynamic_select
// CHECK-DAG: %[[M0:.*]] = tla.mutex "auto_ub_0_256" {id = 0 : i64} -> !tla.mutex
// CHECK-DAG: %[[M1:.*]] = tla.mutex "auto_ub_256_256" {id = 1 : i64} -> !tla.mutex
// CHECK-DAG: %[[M2:.*]] = tla.mutex "auto_ub_512_256" {id = 2 : i64} -> !tla.mutex
// The mutex is carried as a parallel result of the original pointer control
// flow. This remains valid even when an inner selector condition is defined in
// an outer branch and therefore cannot be recreated next to the instruction.
// CHECK: %[[CHOSEN012:.*]]:2 = scf.if %{{.*}} -> (!tla.ptr<f32, ub, 256>, !tla.mutex) {
// CHECK: %[[CHOSEN01:.*]]:2 = scf.if %{{.*}} -> (!tla.ptr<f32, ub, 256>, !tla.mutex) {
// CHECK: scf.yield %{{.*}}, %[[M0]] : !tla.ptr<f32, ub, 256>, !tla.mutex
// CHECK: scf.yield %{{.*}}, %[[M1]] : !tla.ptr<f32, ub, 256>, !tla.mutex
// CHECK: scf.yield %[[CHOSEN01]]#0, %[[CHOSEN01]]#1 : !tla.ptr<f32, ub, 256>, !tla.mutex
// CHECK: scf.yield %{{.*}}, %[[M2]] : !tla.ptr<f32, ub, 256>, !tla.mutex
// CHECK: "tla.vector"() ({
// CHECK-NEXT: tla.mutex_lock %[[CHOSEN012]]#1[<mte2>] : !tla.mutex
// CHECK-NEXT: tla.copy
// CHECK-NEXT: tla.mutex_unlock %[[CHOSEN012]]#1[<mte2>] : !tla.mutex
