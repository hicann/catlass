// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// tla.full broadcasts a scalar or a *one-lane* vector fragment (a tla.reduce
// result). A full-width vector source lowers to the same AVE vector_broadcast,
// which keeps lane 0 and drops the rest, so the verifier rejects it rather than
// letting the lowering truncate silently. Checked here at the IR level: the
// frontend refuses it first, so nothing else exercises the verifier.
// CHECK: 'tla.full' op vector source must be a one-lane fragment, got 64 valid lanes

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @full_full_width_vector_source(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xf32, #hivm.address_space<ub>> to !t
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf32, #hivm.address_space<ub>> to !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %v = "tla.load"(%src_tile) : (!t) -> !tla.vector<64xf32>
        %full = "tla.full"(%v) : (!tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %full) : (!t, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
