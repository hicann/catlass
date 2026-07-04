// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @copy_kernel(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    %1 = "tla.make_coord"() : () -> !tla.coord<32,32>
    %2 = "tla.tile_view"(%arg0, %0, %1)
        : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<32,32>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>
    %4 = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l1, 512>
    %5 = "tla.recast_ptr"(%4) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f32, l1, 512>
    %6 = "tla.make_tensor_like"(%5, %2) {layoutTag = "zN"}
        : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,999)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>
    "tla.cube"() ({
      "tla.copy"(%6, %2)
          : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,999)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK: func.func private @copy_gm_row_major_to_cbuf_zN_float
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-LABEL: func.func @copy_kernel
// CHECK-DAG: [[SRC:%.*]] = memref.cast %arg0 : memref<128x128xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK-DAG: [[DST_BASE:%.*]] = hivm.hir.pointer_cast{{.*}} : memref<1024xf32, #hivm.address_space<cbuf>>
// CHECK-DAG: [[DST:%.*]] = memref.cast [[DST_BASE]] : memref<1024xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK-DAG: [[PACKED_STRIDE:%.*]] = llvm.mlir.constant(999 : i64) : i64
// CHECK-NOT: memref.subview
// CHECK: call @copy_gm_row_major_to_cbuf_zN_float([[SRC]], [[DST]]
// CHECK-SAME: [[PACKED_STRIDE]]
// CHECK-NOT: memref.subview
// CHECK: return
// CHECK-NOT: "tla.copy"
// CHECK-NOT: "tla.alloc_ptr"
// CHECK-NOT: "tla.recast_ptr"
