// RUN: %tla_compile %s -o - | %filecheck %s

// A rank-1 view that starts at element 4 and takes every second element:
// offset *and* strided. Both halves are handled, and they compose --
//   * the launch folds the view's start into the pointer (4*2 = 8 elements in,
//     since the coordinate is scaled by the view's own stride), and
//   * the index is scaled by that stride, so t[1] reaches the element two
//     along rather than the neighbour.

!ub_strided = !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<2>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_strided_view(%ub_memref: memref<16xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %strided = tla.tensor_desc %ub_memref shape [%c1, %c4, %c1, %c1] stride [%c16, %c2, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c4] : memref<16xf32, #hivm.address_space<ub>> -> !ub_strided
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %strided[%c1] : <!tla.layout<!tla.shape<4>, !tla.stride<2>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>> -> f32
        tla.simt_store %strided[%c2], %v : <!tla.layout<!tla.shape<4>, !tla.stride<2>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 4, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The start is folded into the pointer rather than left to the callee.
// CHECK-LABEL: func.func @simt_strided_view(
// CHECK:         llvm.getelementptr
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_strided_view_vf_simt

// CHECK-LABEL: func.func @simt_strided_view_vf_simt
// CHECK:         memref.load
// CHECK:         memref.store
