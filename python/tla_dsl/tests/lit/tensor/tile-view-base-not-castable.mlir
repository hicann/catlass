// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s
//
// A tile_view of a memref-backed buffer captured by a vector region:
// materializeBaseMemref hands back the *base* (16 elements) while the captured
// view bridges to an 8-element slice at a dynamic offset. That adaptation is not
// representable, so the lowering must refuse it. It must NOT build the
// memref.cast anyway -- that used to escape as an unanchored verifier error
// ("operand type 'memref<16xi32,...>' and result type 'memref<8xi32, strided...>'
// are cast incompatible") from an op the user never wrote.

!full  = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!chunk = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<8>, !tla.ptr<i32, ub, 4>>

module {
  func.func @tile_view_base_not_castable(%ub: memref<16xi32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %full = tla.tensor_desc %ub shape [%c16, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c16, %c1] coord [%c0, %c0] : memref<16xi32, #hivm.address_space<ub>> -> !full
    %ts = tla.make_shape -> !tla.shape<8>
    %tc = tla.make_coord -> !tla.coord<8>
    %chunk = tla.tile_view %full, %ts, %tc : !full, !tla.shape<8>, !tla.coord<8> -> !chunk
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.load %chunk : !chunk -> !tla.vector<8xi32>
        tla.store %chunk, %v : !chunk, !tla.vector<8xi32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// The diagnostic is anchored on the user's own op, not on a synthesized cast.
// CHECK: error: failed to legalize operation 'tla.load'
// CHECK-NOT: are cast incompatible
