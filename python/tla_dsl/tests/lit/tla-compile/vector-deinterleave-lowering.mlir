// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_deinterleave_f32(
      %src0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %src1_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst1_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src0 = builtin.unrealized_conversion_cast %src0_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    %src1 = builtin.unrealized_conversion_cast %src1_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    %dst0 = builtin.unrealized_conversion_cast %dst0_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    %dst1 = builtin.unrealized_conversion_cast %dst1_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src0_tile = "tla.tile_view"(%src0, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %src1_tile = "tla.tile_view"(%src1, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst0_tile = "tla.tile_view"(%dst0, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst1_tile = "tla.tile_view"(%dst1, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %v0 = tla.load %src0_tile : !fvec -> !fvec
      %v1 = tla.load %src1_tile : !fvec -> !fvec
      %r0, %r1 = tla.deinterleave %v0, %v1 : !fvec, !fvec -> !fvec, !fvec
      tla.store %dst0_tile, %r0 : !fvec, !fvec
      tla.store %dst1_tile, %r1 : !fvec, !fvec
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vdintlv
// CHECK-NOT: tla.deinterleave
