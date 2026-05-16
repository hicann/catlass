// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @consume_root(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    func.return
  }

  func.func @caller(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    call @consume_root(%arg0)
        : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> ()
    func.return
  }
}

// CHECK-LABEL: func.func @consume_root(
// CHECK-SAME: %{{.*}}: memref<128x128xf32, #hivm.address_space<gm>>
// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %[[ARG0:.*]]: memref<128x128xf32, #hivm.address_space<gm>>
// CHECK: call @consume_root(%[[ARG0]]) : (memref<128x128xf32, #hivm.address_space<gm>>) -> ()
// CHECK-NOT: !tla.tensor
