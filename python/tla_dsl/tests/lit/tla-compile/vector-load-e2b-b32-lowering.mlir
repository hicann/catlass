// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

// E2B_B32 (i32, VL=64): element-to-DataBlock broadcast. Reads VL/DB=8 b32
// elements (DB = 32-byte DataBlock = 8 b32 lanes) from the tile-view base
// and replicates each element across its destination DataBlock to fill a
// VL-wide register.

!ivec = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 2>>

module {
  func.func @vector_load_e2b_b32(
      %src_memref: memref<1xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<1xi32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c1, %src_c1, %src_c1] stride [%src_c1, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c1] coord [%src_c0, %src_c0] : memref<1xi32, #hivm.address_space<ub>> -> !ivec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c1, %dst_c1, %dst_c1] stride [%dst_c1, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c1] coord [%dst_c0, %dst_c0] : memref<1xi32, #hivm.address_space<ub>> -> !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<1>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ivec, !tla.shape<1>, !tla.coord<0>) -> !ivec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ivec, !tla.shape<1>, !tla.coord<0>) -> !ivec
      %loaded = tla.load %src_tile {load_dist = #tla.load_dist<e2b_b32>} : !ivec -> !tla.vector<64xi32>
      tla.store %dst_tile, %loaded : !ivec, !tla.vector<64xi32>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload <E2B_B32>
// CHECK-SAME: into vector<64xi32>
