// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-insert-auto-mutex 2>&1 | %filecheck %s

!layout = !tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>
!coord = !tla.coord<0>
!ub = !tla.tensor<!layout, !coord, !tla.ptr<f32, ub, 256>>

module {
  tla.func @auto_mutex_vec_func() attributes {tla.auto_sync = "v0"} {
    %shape = tla.make_shape -> !tla.shape<64>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride : !tla.shape<64>, !tla.stride<1> -> !layout
    %coord = tla.make_coord -> !coord
    %src_ptr = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %dst_ptr = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %src = tla.make_tensor %src_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !ub
    %dst = tla.make_tensor %dst_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !layout, !coord -> !ub
    %length = arith.constant 4 : i64
    %scalar = arith.constant 7 : i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %value = tla.load %src : !ub -> !tla.vector<64xf32>
        tla.store %dst, %value : !ub, !tla.vector<64xf32>
      }) : () -> ()
      tla.print_tensor %src length = %length shape = [64] : !ub, i64
      tla.debug_print %scalar : i32
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @auto_mutex_vec_func
// CHECK: %[[SRC:.*]] = tla.mutex "auto_ub_0_256" {id = 0 : i64} -> !tla.mutex
// CHECK-NEXT: %[[DST:.*]] = tla.mutex "auto_ub_256_256" {id = 1 : i64} -> !tla.mutex
// CHECK: "tla.vector"() ({
// CHECK-NEXT: tla.mutex_lock %[[SRC]][<vector>] : !tla.mutex
// CHECK-NEXT: tla.mutex_lock %[[DST]][<vector>] : !tla.mutex
// CHECK-NEXT: "tla.vec.func"
// CHECK: tla.load
// CHECK: tla.store
// CHECK: tla.mutex_unlock %[[DST]][<vector>] : !tla.mutex
// CHECK-NEXT: tla.mutex_unlock %[[SRC]][<vector>] : !tla.mutex
// print_tensor and debug_print are deliberately outside the instruction lock
// scope and do not receive their own automatic synchronization.
// CHECK-NEXT: tla.print_tensor
// CHECK-NEXT: tla.debug_print
// CHECK-NOT: tla.mutex_lock
// CHECK: return
