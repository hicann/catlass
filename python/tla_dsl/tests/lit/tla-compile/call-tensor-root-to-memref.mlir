// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

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

// CHECK: error:
// CHECK-SAME: tensor-typed func.call is not supported
