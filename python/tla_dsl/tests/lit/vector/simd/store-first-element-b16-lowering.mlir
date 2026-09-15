// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!ub_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  func.func @store_first_element_b16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !ub_f16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !ub_f16
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ub_f16, !tla.shape<64>, !tla.coord<0>) -> !ub_f16
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ub_f16, !tla.shape<64>, !tla.coord<0>) -> !ub_f16
      %loaded = tla.load %src_tile : !ub_f16 -> !tla.vector<64xf16>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<first_element_b16>} : !ub_f16, !tla.vector<64xf16>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload
// CHECK: ave.hir.masked_store <ONEPT_B16>
// CHECK-SAME: vector<128xf16>
// CHECK-NOT: tla.store
