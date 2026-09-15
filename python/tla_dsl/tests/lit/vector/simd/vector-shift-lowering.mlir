// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.shift --implicit-check-not=hivm.vshl.u --implicit-check-not=hivm.vshr.u --implicit-check-not=hivm.vshls.u --implicit-check-not=hivm.vshrs.u

!ivec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>

module {
  func.func @signed_shift_lowering(
      %src_memref: memref<128xi16, #hivm.address_space<ub>>,
      %amt_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !ivec
    %amt = tla.tensor_desc %amt_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !ivec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ivec, !tla.shape<128>, !tla.coord<0>) -> !ivec
      %amt_tile = "tla.tile_view"(%amt, %shape, %coord) : (!ivec, !tla.shape<128>, !tla.coord<0>) -> !ivec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ivec, !tla.shape<128>, !tla.coord<0>) -> !ivec
      %mask = "tla.create_mask"() {pattern = "ALL", dtype = i16} : () -> !tla.mask<128>
      %src_vec = tla.load %src_tile : !ivec -> !tla.vector<128xi16>
      %amt_vec = tla.load %amt_tile : !ivec -> !tla.vector<128xi16>
      %sl = tla.shift_left %src_vec, %amt_vec mask %mask : !tla.vector<128xi16>, !tla.vector<128xi16> mask !tla.mask<128> -> !tla.vector<128xi16>
      %sr = tla.shift_right %sl, %amt_vec mask %mask : !tla.vector<128xi16>, !tla.vector<128xi16> mask !tla.mask<128> -> !tla.vector<128xi16>
      %c4 = arith.constant 4 : i16
      %sls = tla.shift_lefts %sr, %c4 mask %mask : !tla.vector<128xi16>, i16 mask !tla.mask<128> -> !tla.vector<128xi16>
      %srs = tla.shift_rights %sls, %c4 mask %mask : !tla.vector<128xi16>, i16 mask !tla.mask<128> -> !tla.vector<128xi16>
      tla.store %dst_tile, %srs mask %mask : !ivec, !tla.vector<128xi16> mask !tla.mask<128>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK-DAG: hivm.vshl.s.x
// CHECK-DAG: hivm.vshr.s.x
// CHECK-DAG: hivm.vshls.s.x
// CHECK-DAG: hivm.vshrs.s.x
