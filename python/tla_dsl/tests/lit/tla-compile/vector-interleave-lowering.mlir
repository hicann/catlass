// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_interleave_f32(
      %src0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %src1_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst1_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src0_c0 = arith.constant 0 : index
    %src0_c1 = arith.constant 1 : index
    %src0_c64 = arith.constant 64 : index
    %src0 = tla.tensor_desc %src0_memref[%src0_c0, %src0_c0, %src0_c64, %src0_c1, %src0_c1, %src0_c64, %src0_c1, %src0_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %src1_c0 = arith.constant 0 : index
    %src1_c1 = arith.constant 1 : index
    %src1_c64 = arith.constant 64 : index
    %src1 = tla.tensor_desc %src1_memref[%src1_c0, %src1_c0, %src1_c64, %src1_c1, %src1_c1, %src1_c64, %src1_c1, %src1_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst0_c0 = arith.constant 0 : index
    %dst0_c1 = arith.constant 1 : index
    %dst0_c64 = arith.constant 64 : index
    %dst0 = tla.tensor_desc %dst0_memref[%dst0_c0, %dst0_c0, %dst0_c64, %dst0_c1, %dst0_c1, %dst0_c64, %dst0_c1, %dst0_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst1_c0 = arith.constant 0 : index
    %dst1_c1 = arith.constant 1 : index
    %dst1_c64 = arith.constant 64 : index
    %dst1 = tla.tensor_desc %dst1_memref[%dst1_c0, %dst1_c0, %dst1_c64, %dst1_c1, %dst1_c1, %dst1_c64, %dst1_c1, %dst1_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src0_tile = "tla.tile_view"(%src0, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %src1_tile = "tla.tile_view"(%src1, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst0_tile = "tla.tile_view"(%dst0, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst1_tile = "tla.tile_view"(%dst1, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %v0 = tla.load %src0_tile : !fvec -> !fvec
      %v1 = tla.load %src1_tile : !fvec -> !fvec
      %r0, %r1 = tla.interleave %v0, %v1 : !fvec, !fvec -> !fvec, !fvec
      tla.store %dst0_tile, %r0 : !fvec, !fvec
      tla.store %dst1_tile, %r1 : !fvec, !fvec
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vintlv
// CHECK-NOT: tla.interleave
