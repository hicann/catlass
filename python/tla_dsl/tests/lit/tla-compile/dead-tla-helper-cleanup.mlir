// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  func.func @dead_tla_helpers() {
    %shape = "tla.make_shape"() : () -> !tla.shape<4,4>
    %coord = "tla.make_coord"() : () -> !tla.coord<0,0>
    %stride = "tla.make_stride"() : () -> !tla.stride<4,1>
    %layout = "tla.make_layout"(%shape, %stride)
        : (!tla.shape<4,4>, !tla.stride<4,1>) -> !tla.layout<!tla.shape<4,4>, !tla.stride<4,1>, !tla.shape<4,4>, row_major>
    func.return
  }
}

// CHECK-LABEL: func.func @dead_tla_helpers()
// CHECK-NOT: tla.make_shape
// CHECK-NOT: tla.make_coord
// CHECK-NOT: tla.make_stride
// CHECK-NOT: tla.make_layout
// CHECK: return
