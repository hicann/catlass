// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @consume_tile(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) {
    func.return
  }

  func.func @caller(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %shape = "tla.make_shape"() : () -> !tla.shape<32,32>
    %coord = "tla.make_coord"() : () -> !tla.coord<32,32>
    %tile = "tla.tile_view"(%arg0, %shape, %coord)
        : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<32,32>, !tla.coord<32,32>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>
    call @consume_tile(%tile)
        : (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, row_major>, !tla.coord<32,32>, !tla.ptr<f32, gm, 4>>) -> ()
    func.return
  }
}

// CHECK-LABEL: func.func @consume_tile(
// CHECK-SAME: %{{.*}}: memref<32x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %[[ARG0:.*]]: memref<128x128xf32, #hivm.address_space<gm>>
// CHECK: %[[BASE:.*]], %{{.*}}, %{{.*}}:2, %{{.*}}:2 = memref.extract_strided_metadata %[[ARG0]]
// CHECK: %[[TILE:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [4128], sizes: [32, 32], strides: [128, 1]
// CHECK: %[[CAST:.*]] = memref.cast %[[TILE]]
// CHECK-SAME: to memref<32x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK: call @consume_tile(%[[CAST]])
// CHECK-NOT: !tla.tensor
