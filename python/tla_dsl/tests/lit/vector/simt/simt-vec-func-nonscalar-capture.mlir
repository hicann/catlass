// RUN: %tla_compile %s -o - 2>&1 | %filecheck %s

// NOTE: no `not` on the RUN line, unlike the other negative SIMT tests. This
// diagnostic is emitted from inside the pass, and tla-compile still exits 0
// after emitting it -- only verifier-level failures set a non-zero status. The
// test therefore checks the message rather than the exit code; if the exit
// status is ever fixed, add `not` here.

// The SIMT launch ABI forwards buffers (as tensors) and scalars. Anything else
// read from outside the region has no launch-argument form, and must be
// diagnosed rather than silently dropped or miscompiled -- here a raw memref,
// used directly inside the region instead of through a tla tensor descriptor.

module {
  func.func @simt_nonscalar_capture(%raw: memref<128xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = memref.load %raw[%c0] : memref<128xf32, #hivm.address_space<gm>>
        %s = tla.simt_add %v, %v : f32
        memref.store %s, %raw[%c0] : memref<128xf32, #hivm.address_space<gm>>
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: error: {{.*}}'tla.vec.func' op a SIMT tla.vec.func can only capture scalar runtime values
// CHECK-SAME: but this region reads one of type 'memref<128xf32
// CHECK-SAME: Buffers must be captured as tensors
