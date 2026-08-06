// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

!tile = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<4,1>, !tla.shape<2,4>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @reject_non_unit_linear_tail(
      %base: memref<2x4xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %bad = tla.tensor_desc %base
      shape[%c2, %c4, %c2, %c1]
      stride[%c4, %c1, %c1, %c1]
      origin_shape[%c2, %c4]
      coord[%c0, %c0]
      : memref<2x4xf32, #hivm.address_space<ub>> -> !tile
    return
  }

  func.func @reject_non_unit_linear_stride_tail(
      %base: memref<2x4xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %bad = tla.tensor_desc %base
      shape[%c2, %c4, %c1, %c1]
      stride[%c4, %c1, %c2, %c1]
      origin_shape[%c2, %c4]
      coord[%c0, %c0]
      : memref<2x4xf32, #hivm.address_space<ub>> -> !tile
    return
  }
}

// CHECK-COUNT-2: error: 'tla.tensor_desc' op linear layout shape[2:4] and stride[2:4] must be constant 1
