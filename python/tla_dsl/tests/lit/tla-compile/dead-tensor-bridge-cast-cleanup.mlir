// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  func.func @dead_tensor_bridge_cast(%arg0: memref<4x4xf32>) {
    %tensor = builtin.unrealized_conversion_cast %arg0
        : memref<4x4xf32> to !tla.tensor<!tla.layout<!tla.shape<4,4>, !tla.stride<4,1>, !tla.shape<4,4>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
    func.return
  }
}

// CHECK-LABEL: func.func @dead_tensor_bridge_cast
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
