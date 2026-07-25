// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.load_mask --implicit-check-not=tla.store_mask

!mask_bytes = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, row_major>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @load_store_mask_lowering(
      %mask_memref: memref<8xi8, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c8 = arith.constant 8 : index
    %mask = tla.tensor_desc %mask_memref[%m_c0, %m_c0, %m_c8, %m_c1, %m_c1, %m_c8, %m_c1, %m_c8] : (memref<8xi8, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !mask_bytes
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c64, %src_c1, %src_c1, %src_c64, %src_c1, %src_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c64, %dst_c1, %dst_c1, %dst_c64, %dst_c1, %dst_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    "tla.vec.func"() ({
      %mask_shape = "tla.make_shape"() : () -> !tla.shape<8>
      %data_shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %mask_shape, %coord) : (!mask_bytes, !tla.shape<8>, !tla.coord<0>) -> !mask_bytes
      %src_tile = "tla.tile_view"(%src, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %pattern = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      tla.store %mask_tile, %pattern : !mask_bytes, !tla.mask<64>
      %loaded = tla.load %mask_tile : !mask_bytes -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %sum = tla.add %vec, %vec mask %loaded : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sum mask %loaded : !fvec, !tla.vector<64xf32> mask !tla.mask<64>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @load_store_mask_lowering
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <H>
// CHECK: ave.hir.masked_store <NORM_B8>
// CHECK-SAME: memref<64xi1
// CHECK: ave.hir.vload <NORM>
// CHECK-SAME: memref<64xi1
// CHECK-SAME: into vector<64xi1>
// CHECK: ave.hir.vadd
// CHECK: ave.hir.masked_store
