// RUN: %tla_compile %s -o - | %filecheck %s

// A rank-2 tensor captured into a SIMT region. Only a bare pointer crosses the
// launch ABI, so the parameter stays flat and the two indices are folded into
// one: m[i, j] becomes flat[i*stride0 + j*stride1]. The parameter is sized to
// *span* the view, which for a contiguous 2x4 is 1*4 + 3*1 + 1 = 8.

!ub_2d = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<4,1>, !tla.shape<2,4>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_rank2(%ub_memref: memref<8xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %m = tla.tensor_desc %ub_memref shape [%c2, %c4, %c1, %c1] stride [%c4, %c1, %c1, %c1] origin_shape [%c2, %c4] coord [%c0, %c0] : memref<8xf32, #hivm.address_space<ub>> -> !ub_2d
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %m[%c1, %c2] : <!tla.layout<!tla.shape<2,4>, !tla.stride<4,1>, !tla.shape<2,4>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>> -> f32
        tla.simt_store %m[%c0, %c1], %v : <!tla.layout<!tla.shape<2,4>, !tla.stride<4,1>, !tla.shape<2,4>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @simt_rank2(
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_rank2_vf_simt

// Flat parameter spanning the view, and single-index accesses: the indices were
// folded with the tensor's own strides before outlining.
// CHECK-LABEL: func.func @simt_rank2_vf_simt
// CHECK-SAME:    memref<8xf32, 6>
// CHECK:         memref.load %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}] : memref<8xf32, 6>
// CHECK:         memref.store %{{.*}} : memref<8xf32, 6>
