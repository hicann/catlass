// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region 2>&1 | %filecheck %s

// CHECK-LABEL: func.func @ub_scalar_access_in_vec_func
// CHECK: call @vector_region_
// CHECK-NOT: tla.vec.func
// CHECK-NOT: tla.scalar_load
// CHECK-NOT: tla.scalar_store

// Vector tensor operands and scalar-access memrefs that point at the same
// storage reuse the same helper argument.
// CHECK-LABEL: func.func private @vector_region_
// CHECK-SAME: (%{{.*}}: memref<64xf32, #hivm.address_space<ub>>,
// CHECK-SAME: %{{.*}}: memref<64xf32, #hivm.address_space<ub>>)
// CHECK: ave.hir.masked_store
// CHECK: ave.hir.membar
// CHECK: %[[VALUE:.*]] = memref.load
// CHECK: memref.store %[[VALUE]]

// CHECK-LABEL: func.func @ub_scalar_only_vec_func
// CHECK: %[[SCALAR:.*]] = memref.load
// CHECK: memref.store %[[SCALAR]]
// CHECK-NOT: tla.vec.func
// CHECK-NOT: tla.scalar_load
// CHECK-NOT: tla.scalar_store

!vec64 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @ub_scalar_access_in_vec_func(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0]
      : memref<64xf32, #hivm.address_space<ub>> -> !vec64
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0]
      : memref<64xf32, #hivm.address_space<ub>> -> !vec64
    "tla.vector"() ({
      "tla.vec.func"() ({
        %vector_value = tla.load %src : !vec64 -> !tla.vector<64xf32>
        tla.store %dst, %vector_value : !vec64, !tla.vector<64xf32>
        tla.local_mem_bar 5
        %scalar_value = tla.scalar_load %dst[%c0] : !vec64 -> f32
        tla.scalar_store %src[%c1], %scalar_value : !vec64, f32
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  func.func @ub_scalar_only_vec_func(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0]
      : memref<64xf32, #hivm.address_space<ub>> -> !vec64
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0]
      : memref<64xf32, #hivm.address_space<ub>> -> !vec64
    "tla.vector"() ({
      "tla.vec.func"() ({
        %scalar_value = tla.scalar_load %src[%c0] : !vec64 -> f32
        tla.scalar_store %dst[%c1], %scalar_value : !vec64, f32
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
