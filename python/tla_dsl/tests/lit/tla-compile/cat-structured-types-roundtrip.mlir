// RUN: %tla_compile %s -o %t --mlir-print-ir-before=tla-vector-region 2>&1 | %filecheck %s

module {
  func.func @structured_type_roundtrip(
      %gm: memref<16x?xf16, #hivm.address_space<gm>>) {
    %parent_raw = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<i8, l1, 512>
    %parent_ptr = tla.recast_ptr %parent_raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %parent_shape = tla.make_shape -> !tla.shape<16,16>
    %parent_stride = tla.make_stride -> !tla.stride<16,1>
    %parent_layout = tla.make_layout %parent_shape, %parent_stride : !tla.shape<16,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>
    %parent_coord = tla.make_coord -> !tla.coord<0,0>
    %parent = tla.make_tensor %parent_ptr, %parent_layout, %parent_coord : !tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %lhs_raw = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<i8, l1, 512>
    %lhs_ptr = tla.recast_ptr %lhs_raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %lhs = tla.make_tensor_like %lhs_ptr like %parent layoutTag("zN") : !tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %rhs_raw = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<i8, l0b, 512>
    %rhs_ptr = tla.recast_ptr %rhs_raw : !tla.ptr<i8, l0b, 512> -> !tla.ptr<f16, l0b, 512>
    %rhs = tla.make_tensor_like %rhs_ptr like %parent layoutTag("nZ") : !tla.ptr<f16, l0b, 512>, !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>> -> !tla.tensor<!tla.layout<!tla.shape<(8,2),(16,1)>, !tla.stride<(1,256),(8,128)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>
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
// CHECK-SAME: memref<16x?xf16, #hivm.address_space<gm>>
// CHECK: !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
// CHECK: !tla.tensor<!tla.layout<!tla.shape<(8,2),(16,1)>, !tla.stride<(1,256),(8,128)>, !tla.shape<16,16>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 512>>
// CHECK: !tla.shape<(?,16),8>
// CHECK: !tla.stride<(16,1),128>
// CHECK: !tla.layout<!tla.shape<(?,16),8>, !tla.stride<(16,1),128>, !tla.shape<(32,16),8>, row_major>
