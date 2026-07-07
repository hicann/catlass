// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-to-vector 2>&1 | %filecheck %s

// CHECK: ave.hir.vci

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @vector_arange_lowering(
      %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xi32, #hivm.address_space<ub>> to !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %start = arith.constant 3 : i32
        %idx = "tla.arange"(%start) {order = "increase"} : (i32) -> !t
        "tla.store"(%dst_tile, %idx) : (!t, !t) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
