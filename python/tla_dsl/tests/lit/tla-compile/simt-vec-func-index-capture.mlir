// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// An index-typed captured scalar passes TlaCompile but dies in hivmc with
// "LLVM Translation failed for operation: builtin.unrealized_conversion_cast",
// because the SIMT launch ABI carries integers and floats but not index. It is
// therefore rejected here, where the message can say what to do about it,
// rather than surfacing as an opaque backend failure.

!gm_f32 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  func.func @simt_index_capture(%gm_memref: memref<128xf32, #hivm.address_space<gm>>,
                                %sx: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %gm = tla.tensor_desc %gm_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf32, #hivm.address_space<gm>> -> !gm_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %gm[%sx] : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        tla.simt_store %gm[%c0], %v : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: error: {{.*}}cannot capture an index-typed runtime value
// CHECK-SAME: Cast it to i32 outside the region first
