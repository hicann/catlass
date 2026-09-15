// RUN: %tla_compile %s -o - | %filecheck %s

// A SIMT tla.vec.func forwards runtime scalars computed outside it as extra
// launch arguments, after the buffers. The ABI is positional, so all three of
// these matter and none is checked by the frontend:
//   * the scalar kinds the ABI actually carries survive -- integer and float
//     (both verified on device; index is rejected, see
//     simt-vec-func-index-capture.mlir);
//   * their types are preserved on the outlined function's parameters;
//   * their relative order is preserved between launch and callee.
// Getting the order wrong silently swaps two same-typed values at runtime.

!gm_f32 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!gm_i32 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>

module {
  func.func @simt_scalar_capture(%gm_memref: memref<128xf32, #hivm.address_space<gm>>,
                                 %gi_memref: memref<128xi32, #hivm.address_space<gm>>,
                                 %si: i32, %sj: i32, %sf: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %gm = tla.tensor_desc %gm_memref shape [%c128, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c128, %c1] coord [%c0, %c0] : memref<128xf32, #hivm.address_space<gm>> -> !gm_f32
    %gi = tla.tensor_desc %gi_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi32, #hivm.address_space<gm>> -> !gm_i32
    "tla.vector"() ({
      "tla.vec.func"() ({
        // Use each captured scalar, in this order: i32, i32, then f32.
        // Both i32 scalars are genuinely used: an unused capture is elided
        // before it ever reaches the ABI, which would silently weaken the
        // ordering check below to a single i32.
        %i2 = tla.simt_add %si, %sj : i32
        tla.simt_store %gi[%c0], %i2 : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>, i32
        %f = tla.simt_load %gm[%c0] : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        %f2 = tla.simt_add %f, %sf : f32
        tla.simt_store %gm[%c0], %f2 : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The launch passes the buffer pointer first, then the scalars in use order.
// CHECK-LABEL: func.func @simt_scalar_capture(
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_scalar_capture_vf_simt
// CHECK-SAME:      !llvm.ptr<1>, i32, i32, f32

// The outlined function's parameters mirror that exactly: buffer, then the
// three scalars, same types, same order.
// CHECK-LABEL: func.func @simt_scalar_capture_vf_simt
// CHECK-SAME:    memref<128xf32, 1>
// CHECK-SAME:    i32
// CHECK-SAME:    i32
// CHECK-SAME:    f32
