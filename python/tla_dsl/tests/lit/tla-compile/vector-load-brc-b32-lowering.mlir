// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_load_brc_b32(
      %src_memref: memref<1xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<1xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c1, %src_c1, %src_c1, %src_c1, %src_c1, %src_c1] : (memref<1xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c1, %dst_c1, %dst_c1, %dst_c1, %dst_c1, %dst_c1] : (memref<1xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<1>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<1>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<1>, !tla.coord<0>) -> !fvec
      %loaded = tla.load %src_tile {load_dist = #tla.load_dist<brc_b32>} : !fvec -> !fvec
      tla.store %dst_tile, %loaded : !fvec, !fvec
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload <BRC_B32>
