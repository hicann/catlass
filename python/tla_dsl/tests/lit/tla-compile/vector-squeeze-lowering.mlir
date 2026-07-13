// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!ivec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @vector_squeeze_f32(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf32, #hivm.address_space<ub>> to !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f32} : () -> !tla.mask
      %v0 = "tla.load"(%src_tile) : (!fvec) -> !fvec
      %v1 = "tla.squeeze"(%v0, %mask) : (!fvec, !tla.mask) -> !fvec
      "tla.store"(%dst_tile, %v1) : (!fvec, !fvec) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_f16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<128xf16, #hivm.address_space<ub>> to !hvec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<128xf16, #hivm.address_space<ub>> to !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f16} : () -> !tla.mask
      %v0 = "tla.load"(%src_tile) : (!hvec) -> !hvec
      %v1 = "tla.squeeze"(%v0, %mask) : (!hvec, !tla.mask) -> !hvec
      "tla.store"(%dst_tile, %v1) : (!hvec, !hvec) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_i32(
      %src_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xi32, #hivm.address_space<ub>> to !ivec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xi32, #hivm.address_space<ub>> to !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = i32} : () -> !tla.mask
      %v0 = "tla.load"(%src_tile) : (!ivec) -> !ivec
      %v1 = "tla.squeeze"(%v0, %mask) : (!ivec, !tla.mask) -> !ivec
      "tla.store"(%dst_tile, %v1) : (!ivec, !ivec) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: func.func private @vsqueeze_int32_t
// CHECK: func.func private @vsqueeze_half
// CHECK: func.func private @vsqueeze_float
// CHECK: call @vsqueeze_float
// CHECK: call @vsqueeze_half
// CHECK: call @vsqueeze_int32_t
// CHECK-NOT: _mlir_ciface_vsqueeze_
// CHECK-NOT: tla.squeeze
