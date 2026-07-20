// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region 2>&1 | %filecheck %s

// CHECK: ave.hir.scalar_broadcast

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @scalar_lane_broadcast(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !t
    %dst = tla.tensor_desc %dst_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %reg = "tla.load"(%src_tile) : (!t) -> !t
        %zero = arith.constant 0.000000e+00 : f32
        %full = "tla.full"(%zero) : (f32) -> !t
        %out = "tla.add"(%reg, %full) : (!t, !t) -> !t
        "tla.store"(%dst_tile, %out) : (!t, !t) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
