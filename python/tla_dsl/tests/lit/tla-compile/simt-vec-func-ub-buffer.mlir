// RUN: %tla_compile %s -o - | %filecheck %s

// A SIMT tla.vec.func may read and write a UB buffer as well as GM: that is how
// threads of one block share scratch (tla.arch.sync_threads between them). The
// two address spaces must lower to *different* integer address spaces on the
// outlined function's memref parameters -- gm to 1, ub to 6 -- and the same
// spaces must appear on the launch's !llvm.ptr operands, or hivmc-a5 rejects the
// call signature.

!gm_f32 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!ub_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_ub_buffer(%gm_memref: memref<128xf32, #hivm.address_space<gm>>,
                            %ub_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %gm = tla.tensor_desc %gm_memref shape [%c128, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c128, %c1] coord [%c0, %c0] : memref<128xf32, #hivm.address_space<gm>> -> !gm_f32
    %ub = tla.tensor_desc %ub_memref shape [%c64, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c64, %c1] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !ub_f32
    "tla.vector"() ({
      "tla.vec.func"() ({
        // Read GM, publish into the shared UB buffer, read it back and store.
        %g = tla.simt_load %gm[%c0] : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>> -> f32
        tla.simt_store %ub[%c0], %g : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>, f32
        %s = tla.simt_load %ub[%c0] : <!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>> -> f32
        %sum = tla.simt_add %g, %s : f32
        tla.simt_store %gm[%c0], %sum : <!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 64, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The launch resolves each buffer to a pointer in its own address space and
// passes both to the outlined function.
// CHECK-LABEL: func.func @simt_ub_buffer(
// CHECK:         llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
// CHECK:         llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<6>
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_ub_buffer_vf_simt
// CHECK-SAME:      !llvm.ptr<1>, !llvm.ptr<6>

// The outlined function takes both, each in its own integer address space --
// gm as 1, ub as 6. They must differ, and must match the pointers above.
// CHECK-LABEL: func.func @simt_ub_buffer_vf_simt
// CHECK-SAME:    %{{[a-z0-9_]+}}: memref<128xf32, 1>
// CHECK-SAME:    %{{[a-z0-9_]+}}: memref<64xf32, 6>

// Both accesses are plain memref load/store on the parameters: the UB buffer is
// addressed directly, with no descriptor view materialized inside the body.
// CHECK:         memref.load %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}] : memref<128xf32, 1>
// CHECK:         memref.store %{{.*}} : memref<64xf32, 6>
// CHECK:         memref.load %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}] : memref<64xf32, 6>
// CHECK:         memref.store %{{.*}} : memref<128xf32, 1>
