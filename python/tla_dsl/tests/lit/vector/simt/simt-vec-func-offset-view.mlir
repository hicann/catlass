// RUN: %tla_compile %s -o - | %filecheck %s

// Two views of one UB allocation, one starting at element 4. The SIMT launch
// passes bare pointers -- there is no descriptor on the other side -- so the
// view's start must be folded into the pointer, or both views collapse to the
// same address and a write through the offset one silently lands at element 0.
//
// The offset is computed the way TlaLowerScalarAccessPass computes it for the
// general path (coord[0]*stride[0] + coord[1]*stride[1], in elements); the two
// must agree or a tensor means different things inside and outside the region.

!ub_full = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!ub_tail = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>>

module {
  func.func @simt_offset_view(%ub_memref: memref<16xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %full = tla.tensor_desc %ub_memref shape [%c1, %c16, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c0] : memref<16xf32, #hivm.address_space<ub>> -> !ub_full
    %tail = tla.tensor_desc %ub_memref shape [%c1, %c8, %c1, %c1] stride [%c16, %c1, %c1, %c1] origin_shape [%c1, %c16] coord [%c0, %c4] : memref<16xf32, #hivm.address_space<ub>> -> !ub_tail
    "tla.vector"() ({
      "tla.vec.func"() ({
        %v = tla.simt_load %full[%c0] : <!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>> -> f32
        tla.simt_store %tail[%c0], %v : <!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<16>, RowMajor>, !tla.coord<4>, !tla.ptr<f32, ub, 4>>, f32
      }) {mode = "simt", thread_block_dim = array<i64: 1, 1, 1>} : () -> ()
    }) : () -> ()
    return
  }
}

// The whole-allocation view is passed as the bare base pointer; the offset view
// is passed as that pointer advanced by 4 elements.
// CHECK-LABEL: func.func @simt_offset_view(
// CHECK:         llvm.getelementptr
// CHECK:         hivm_regbaseintrins.intrins.launch_func @simt_offset_view_vf_simt

// The parameters are bounded by their own views, not by the allocation: the
// offset one is 8 elements, so an index cannot run off the end of the buffer.
// CHECK-LABEL: func.func @simt_offset_view_vf_simt
// CHECK-SAME:    memref<16xf32, 6>
// CHECK-SAME:    memref<8xf32, 6>
