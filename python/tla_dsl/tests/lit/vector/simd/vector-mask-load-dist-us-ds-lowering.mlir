// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

// MaskLoadDist US/DS must lower via AIV bc (func.call), not ave.hir.vload:
// NPU-IR i1→plds hardcodes DIST_NORM.
// b8 and b32 UB both work (AscendC LoadAlign<T, MaskDist> T typing).

!mask_i8_64 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!mask_i32_64 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @mask_load_dist_ds_i8(
      %mask_memref: memref<64xi8, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c64 = arith.constant 64 : index
    %mask = tla.tensor_desc %mask_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xi8, #hivm.address_space<ub>> -> !mask_i8_64
    %src = tla.tensor_desc %src_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst = tla.tensor_desc %dst_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %shape, %coord) : (!mask_i8_64, !tla.shape<64>, !tla.coord<0>) -> !mask_i8_64
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %preg = tla.load %mask_tile {load_dist = #tla.load_dist<ds>} : !mask_i8_64 -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %zero = tla.sub %vec, %vec : !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      %sel = tla.where %preg, %vec, %zero : !tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sel : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  func.func @mask_load_dist_us_i8(
      %mask_memref: memref<64xi8, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c64 = arith.constant 64 : index
    %mask = tla.tensor_desc %mask_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xi8, #hivm.address_space<ub>> -> !mask_i8_64
    %src = tla.tensor_desc %src_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst = tla.tensor_desc %dst_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %shape, %coord) : (!mask_i8_64, !tla.shape<64>, !tla.coord<0>) -> !mask_i8_64
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %preg = tla.load %mask_tile {load_dist = #tla.load_dist<us>} : !mask_i8_64 -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %zero = tla.sub %vec, %vec : !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      %sel = tla.where %preg, %vec, %zero : !tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sel : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  func.func @mask_load_dist_ds_i32(
      %mask_memref: memref<64xi32, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c64 = arith.constant 64 : index
    %mask = tla.tensor_desc %mask_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xi32, #hivm.address_space<ub>> -> !mask_i32_64
    %src = tla.tensor_desc %src_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst = tla.tensor_desc %dst_memref shape [%m_c1, %m_c64, %m_c1, %m_c1] stride [%m_c64, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c64] coord [%m_c0, %m_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %shape, %coord) : (!mask_i32_64, !tla.shape<64>, !tla.coord<0>) -> !mask_i32_64
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %preg = tla.load %mask_tile {load_dist = #tla.load_dist<ds>} : !mask_i32_64 -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %zero = tla.sub %vec, %vec : !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      %sel = tla.where %preg, %vec, %zero : !tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sel : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @mask_load_dist_ds_i8
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @mask_load_ds_b8
// CHECK-NOT: ave.hir.vload <DS>
// CHECK-NOT: ave.hir.vload <US>

// CHECK-LABEL: func.func @mask_load_dist_us_i8
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @mask_load_us_b8
// CHECK-NOT: ave.hir.vload <US>

// CHECK-LABEL: func.func @mask_load_dist_ds_i32
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @mask_load_ds_b32
// CHECK-NOT: ave.hir.vload <DS>
