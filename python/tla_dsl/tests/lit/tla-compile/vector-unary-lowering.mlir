// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!ivec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @vector_unary_f32(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %mask = "tla.create_mask"() {dtype = f32, pattern = "M4"} : () -> !tla.mask
      %v0 = tla.load %src_tile : !fvec -> !fvec
      %v1 = tla.exp %v0 : !fvec -> !fvec
      %v2 = tla.log %v1 : !fvec -> !fvec
      %v3 = tla.sqrt %v2 : !fvec -> !fvec
      %v4 = tla.neg %v3 : !fvec -> !fvec
      %v5 = tla.abs %v4 mask %mask : !fvec mask !tla.mask -> !fvec
      tla.store %dst_tile, %v5 : !fvec, !fvec
    }) : () -> ()
    return
  }

  func.func @vector_unary_i32(
      %src_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xi32, #hivm.address_space<ub>> to !ivec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xi32, #hivm.address_space<ub>> to !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %v0 = tla.load %src_tile : !ivec -> !ivec
      %v1 = tla.abs %v0 : !ivec -> !ivec
      %v2 = tla.neg %v1 : !ivec -> !ivec
      tla.store %dst_tile, %v2 : !ivec, !ivec
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vexp
// CHECK: ave.hir.vln
// CHECK: ave.hir.vsqrt
// CHECK: ave.hir.vneg
// CHECK: ave.hir.vabs
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vabs
// CHECK: ave.hir.vneg
// CHECK-NOT: tla.exp
// CHECK-NOT: tla.log
// CHECK-NOT: tla.sqrt
// CHECK-NOT: tla.abs
// CHECK-NOT: tla.neg
