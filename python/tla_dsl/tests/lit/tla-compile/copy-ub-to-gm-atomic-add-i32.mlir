// RUN: %tla_compile %s -o - | %filecheck %s
// 
// Test lowering of tla.copy from UB to GM with atomic add operation.
// The copy should be enclosed by the set_ctrl sequence for atomic add i32,
// followed by the set_ctrl sequence that disables atomic writes.

module {
  tla.func @atomic_add_ub2gm(%arg0: !tla.tensor<!tla.layout<!tla.shape<64,64>, !tla.stride<64,1>, !tla.shape<64,64>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>) {
    %0 = tla.make_shape -> !tla.shape<16,16>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.make_coord -> !tla.coord<0,0>
    %3 = tla.tile_view %arg0, %0, %2 : !tla.tensor<!tla.layout<!tla.shape<64,64>, !tla.stride<64,1>, !tla.shape<64,64>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.shape<16,16>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>
    %4 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, ub, 256> -> !tla.ptr<i32, ub, 256>
    %6 = tla.make_shape -> !tla.shape<16,16>
    %7 = tla.make_stride -> !tla.stride<16,1>
    %8 = tla.make_layout %6, %7 : !tla.shape<16,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>
    %9 = tla.make_coord -> !tla.coord<0,0>
    %10 = tla.make_tensor %5, %8, %9 : !tla.ptr<i32, ub, 256>, !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
    "tla.vector"() ({
      tla.copy %3, %10 {atomic_mode = #tla.atomic_mode<add>} : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<64,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, ub, 256>>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_ub_row_major_to_gm_row_major_int32_t
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @atomic_add_ub2gm
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl true at ctrl[8]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[9]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[10]
// CHECK: call @copy_ub_row_major_to_gm_row_major_int32_t
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[7]
// CHECK-NEXT: hivm.hir.set_ctrl false at ctrl[8]
// CHECK-NOT: tla.copy
// CHECK-NOT: tla.make_tensor
