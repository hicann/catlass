// RUN: %tla_compile %s -o %t --mlir-print-ir-before=tla-lower-to-std 2>&1 | %filecheck %s

module {
  func.func @structured_type_roundtrip(
      %gm: !tla.memref<16x?xf16, gm>,
      %lhs: !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>,
      %rhs: !tla.tensor<!tla.layout<!tla.shape<(8,2),(16,1)>, !tla.stride<(1,256),(8,128)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>) {
    %shape = "tla.make_shape"() : () -> !tla.shape<(?,16),8>
    %stride = "tla.make_stride"() : () -> !tla.stride<(16,1),128>
    %origin = "tla.make_shape"() : () -> !tla.shape<(32,16),8>
    %layout = "tla.make_layout"(%shape, %stride, %origin)
        : (!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>)
          -> !tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, row_major>
    func.return
  }
}

// CHECK-LABEL: func.func @structured_type_roundtrip
// CHECK-SAME: !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
// CHECK-SAME: !tla.tensor<!tla.layout<!tla.shape<(8,2),(16,1)>, !tla.stride<(1,256),(8,128)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>
// CHECK: !tla.memref<16x?xf16, gm>
// CHECK: !tla.shape<(?,16),8>
// CHECK: !tla.stride<(16,1),128>
// CHECK: !tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, row_major>
