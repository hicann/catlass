// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @make_tensor_copy_kernel(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %0 = tla.make_shape -> !tla.shape<16,16>
    %1 = tla.make_coord -> !tla.coord<0,0>
    %2 = tla.make_coord -> !tla.coord<0,0>
    %3 = tla.tile_view %arg0, %0, %2 : !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<16,16>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    %4 = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %6 = tla.make_shape -> !tla.shape<16,16>
    %7 = tla.make_stride -> !tla.stride<16,1>
    %8 = tla.make_layout %6, %7 : !tla.shape<16,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>
    %9 = tla.make_coord -> !tla.coord<0,0>
    %10 = tla.make_tensor %5, %8, %9 : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
    "tla.vector"() ({
      tla.copy %3, %10 : !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<128,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_ub_row_major_to_gm_row_major_float
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @make_tensor_copy_kernel
// CHECK: hivm.hir.pointer_cast{{.*}} : memref<256xf32, #hivm.address_space<ub>>
// CHECK: call @copy_ub_row_major_to_gm_row_major_float
// CHECK-NOT: tla.make_tensor
// CHECK-NOT: memref.reinterpret_cast
