// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_scalar_binary_lowering(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c64, %src_c1, %src_c1, %src_c64, %src_c1, %src_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c64, %dst_c1, %dst_c1, %dst_c64, %dst_c1, %dst_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    %cst = arith.constant 5.000000e+00 : f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %mask = "tla.create_mask"() {pattern = "M4", dtype = f32} : () -> !tla.mask
        %v = tla.load %src_tile : !vec -> !tla.vector<64xf32>
        %mul = tla.muls %v, %cst mask %mask : !tla.vector<64xf32>, f32 mask !tla.mask -> !tla.vector<64xf32>
        %max = tla.maxs %mul, %cst : !tla.vector<64xf32>, f32 -> !tla.vector<64xf32>
        %out = tla.mins %max, %cst : !tla.vector<64xf32>, f32 -> !tla.vector<64xf32>
        tla.store %dst_tile, %out : !vec, !tla.vector<64xf32>
      }) : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm.pge
// CHECK: hivm.vmuls.s.x
// CHECK: hivm.vmaxs.s.x
// CHECK: hivm.vmins.s.x
// CHECK-NOT: tla.muls
// CHECK-NOT: tla.maxs
// CHECK-NOT: tla.mins
