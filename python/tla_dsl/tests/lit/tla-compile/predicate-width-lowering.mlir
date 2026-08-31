// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!bvec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>

module {
  func.func @predicate_width_b16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>,
      %true_shape: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %value = tla.load %src_tile : !hvec -> !tla.vector<128xf16>
      %pattern = "tla.create_mask"() {pattern = "H", dtype = f16} : () -> !tla.mask<128>
      %tail, %next = tla.update_mask %true_shape, f16 : !tla.mask<128>, index
      %sum = tla.add %value, %value mask %pattern : !tla.vector<128xf16>, !tla.vector<128xf16> mask !tla.mask<128> -> !tla.vector<128xf16>
      tla.store %dst_tile, %sum mask %tail : !hvec, !tla.vector<128xf16> mask !tla.mask<128>
    }) : () -> ()
    return
  }

  func.func @predicate_width_b8(
      %src_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>,
      %true_shape: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !bvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !bvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %value = tla.load %src_tile : !bvec -> !tla.vector<256xi8>
      %pattern = "tla.create_mask"() {pattern = "H", dtype = i8} : () -> !tla.mask<256>
      %tail, %next = tla.update_mask %true_shape, i8 : !tla.mask<256>, index
      %sum = tla.add %value, %value mask %pattern : !tla.vector<256xi8>, !tla.vector<256xi8> mask !tla.mask<256> -> !tla.vector<256xi8>
      tla.store %dst_tile, %sum mask %tail : !bvec, !tla.vector<256xi8> mask !tla.mask<256>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @predicate_width_b16
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <H> {element_alignment_bit_width = 16 : i32} : vector<256xi1>
// CHECK: ave.hir.plt {{.*}} {element_alignment_bit_width = 16 : i32} : vector<256xi1>, index

// CHECK-LABEL: func.func @predicate_width_b8
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <H> : vector<256xi1>
// CHECK-NOT: element_alignment_bit_width
// CHECK: ave.hir.plt {{.*}} : vector<256xi1>, index
// CHECK-NOT: element_alignment_bit_width
