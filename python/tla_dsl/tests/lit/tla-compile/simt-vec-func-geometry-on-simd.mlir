// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// The frontend rejects malformed SIMT geometry, but hand-written IR reaches the
// verifier directly -- which is the only check that always runs.

!fscalar = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @geometry_on_simd(%src_memref: memref<1xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0] : memref<1xf32, #hivm.address_space<gm>> -> !fscalar
    "tla.vector"() ({
      "tla.vec.func"() ({
        %i = arith.constant 0 : index
        %x = tla.simt_load %src[%i] : <!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
      }) {mode = "simd", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}
// CHECK: 'thread_block_dim' is only valid with mode = "simt"
