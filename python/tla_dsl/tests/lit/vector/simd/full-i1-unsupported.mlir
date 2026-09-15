// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// tla.full does not support an i1 value: the AVE broadcast op it lowers to does
// not accept an i1 source, so i1 is dropped from the op's accepted value types.
// CHECK: 'tla.full' op operand #0 {{.*}}but got 'i1'

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @full_i1_unsupported(
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf32, #hivm.address_space<ub>> to !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %b = arith.constant true
        %full = "tla.full"(%b) : (i1) -> !t
        "tla.store"(%dst_tile, %full) : (!t, !t) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
