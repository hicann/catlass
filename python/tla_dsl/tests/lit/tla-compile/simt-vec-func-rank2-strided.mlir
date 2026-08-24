// RUN: %tla_compile %s -o - | %filecheck %s

// A 2x4 tile of a wider (2x8) tensor: the rows are 8 apart, not 4. A rank-2
// memref parameter could not describe that -- its strides would be the implicit
// packed (4, 1) -- so the indices are folded instead:
//
//     t[i, j]  ->  flat[i*8 + j*1]
//
// The parameter spans the view rather than counting its elements: the last row
// starts at (rows-1)*8, so the span is 1*8 + 3*1 + 1 = 12, not 2*4 = 8.

!ub_tile = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<8,1>, !tla.shape<2,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_rank2_strided(%ub_memref: memref<16xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %m = tla.tensor_desc %ub_memref shape [%c2, %c4, %c1, %c1] stride [%c8, %c1, %c1, %c1] origin_shape [%c2, %c8] coord [%c0, %c0] : memref<16xf32, #hivm.address_space<ub>> -> !ub_tile
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %m[%c1, %c2] : <!tla.layout<!tla.shape<2,4>, !tla.stride<8,1>, !tla.shape<2,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>> -> f32
        tla.simt_store %m[%c0, %c1], %v : <!tla.layout<!tla.shape<2,4>, !tla.stride<8,1>, !tla.shape<2,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @simt_rank2_strided(
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_rank2_strided_vf_simt

// Spans the strided view (12), not its element count (8), so the folded index
// cannot run off the end of the parameter.
// CHECK-LABEL: func.func @simt_rank2_strided_vf_simt
// CHECK-SAME:    memref<12xf32, 6>
// CHECK:         memref.load %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}] : memref<12xf32, 6>
// CHECK:         memref.store %{{.*}} : memref<12xf32, 6>
