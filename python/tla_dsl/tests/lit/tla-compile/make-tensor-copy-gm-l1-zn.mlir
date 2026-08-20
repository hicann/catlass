// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @make_tensor_copy_gm_l1_zn(
      %arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f32, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(8,4)>
    %stride = tla.make_stride -> !tla.stride<(8,128),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,32>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = "zN"} :
      !tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)> origin !tla.shape<32,32> ->
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>
    %coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %coord :
      !tla.ptr<f32, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.cube"() ({
      tla.copy %dst, %arg0 :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @copy_gm_row_major_to_l1_zN_float
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @make_tensor_copy_gm_l1_zn
// CHECK: hivm.hir.pointer_cast{{.*}}memref<1024xf32, #hivm.address_space<cbuf>>
// CHECK: call @copy_gm_row_major_to_l1_zN_float
// CHECK-NOT: tla.make_tensor
// CHECK-NOT: tla.copy
