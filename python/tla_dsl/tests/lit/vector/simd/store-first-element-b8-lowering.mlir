// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!ub_i8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>

module {
  func.func @store_first_element_b8(
      %extent: index,
      %src_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !ub_i8
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %extent] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !ub_i8
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ub_i8, !tla.shape<64>, !tla.coord<0>) -> !ub_i8
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ub_i8, !tla.shape<64>, !tla.coord<0>) -> !ub_i8
      %loaded = tla.load %src_tile : !ub_i8 -> !tla.vector<64xi8>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<first_element_b8>} : !ub_i8, !tla.vector<64xi8>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <ALL>
// CHECK: ave.hir.vload
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store <ONEPT_B8>
// CHECK-SAME: vector<256xi8>
// CHECK-NOT: tla.store
