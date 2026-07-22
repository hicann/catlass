// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!ivec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @vector_squeeze_f32(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c64, %src_c1, %src_c1, %src_c64, %src_c1, %src_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c64, %dst_c1, %dst_c1, %dst_c64, %dst_c1, %dst_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f32} : () -> !tla.mask
      %v0 = "tla.load"(%src_tile) : (!fvec) -> !tla.vector<64xf32>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<64xf32>, !tla.mask) -> !tla.vector<64xf32>
      "tla.store"(%dst_tile, %v1) : (!fvec, !tla.vector<64xf32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_f16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c128, %src_c1, %src_c1, %src_c128, %src_c1, %src_c128] : (memref<128xf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !hvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c128, %dst_c1, %dst_c1, %dst_c128, %dst_c1, %dst_c128] : (memref<128xf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f16} : () -> !tla.mask
      %v0 = "tla.load"(%src_tile) : (!hvec) -> !tla.vector<128xf16>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<128xf16>, !tla.mask) -> !tla.vector<128xf16>
      "tla.store"(%dst_tile, %v1) : (!hvec, !tla.vector<128xf16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_i32(
      %src_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c64, %src_c1, %src_c1, %src_c64, %src_c1, %src_c64] : (memref<64xi32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !ivec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c64, %dst_c1, %dst_c1, %dst_c64, %dst_c1, %dst_c64] : (memref<64xi32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = i32} : () -> !tla.mask
      %v0 = "tla.load"(%src_tile) : (!ivec) -> !tla.vector<64xi32>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<64xi32>, !tla.mask) -> !tla.vector<64xi32>
      "tla.store"(%dst_tile, %v1) : (!ivec, !tla.vector<64xi32>) -> ()
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
