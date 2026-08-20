// RUN: %tla_compile %s -o - | %filecheck %s

module {
  "tla.func"() ({
    %raw = "tla.alloc_ptr"() {size_bytes = 4096 : i64} : () -> !tla.ptr<i8, l1, 512>
    %ptr = "tla.recast_ptr"(%raw) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f16, l1, 512>
    %shape = "tla.make_shape"() : () -> !tla.shape<(16,2),(16,2)>
    %unalign_shape = "tla.make_shape"() : () -> !tla.shape<(32,1),(16,2)>
    %origin = "tla.make_shape"() : () -> !tla.shape<32,32>
    %coord = "tla.make_coord"() : () -> !tla.coord<0,0>

    %zn_stride = "tla.make_stride"() : () -> !tla.stride<(16,256),(1,512)>
    %zn_layout = "tla.make_layout"(%shape, %zn_stride, %origin) {layoutTag = "zN"} :
      (!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>) ->
      !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, zN>
    %zn = "tla.make_tensor"(%ptr, %zn_layout, %coord) :
      (!tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>) ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>

    %nz_stride = "tla.make_stride"() : () -> !tla.stride<(1,512),(16,256)>
    %nz_layout = "tla.make_layout"(%shape, %nz_stride, %origin) {layoutTag = "nZ"} :
      (!tla.shape<(16,2),(16,2)>, !tla.stride<(1,512),(16,256)>, !tla.shape<32,32>) ->
      !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,512),(16,256)>, !tla.shape<32,32>, nZ>
    %nz = "tla.make_tensor"(%ptr, %nz_layout, %coord) :
      (!tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,512),(16,256)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>) ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,512),(16,256)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>

    %zz_stride = "tla.make_stride"() : () -> !tla.stride<(16,512),(1,256)>
    %zz_layout = "tla.make_layout"(%shape, %zz_stride, %origin) {layoutTag = "zZ"} :
      (!tla.shape<(16,2),(16,2)>, !tla.stride<(16,512),(1,256)>, !tla.shape<32,32>) ->
      !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,512),(1,256)>, !tla.shape<32,32>, zZ>
    %zz = "tla.make_tensor"(%ptr, %zz_layout, %coord) :
      (!tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,512),(1,256)>, !tla.shape<32,32>, zZ>, !tla.coord<0,0>) ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,512),(1,256)>, !tla.shape<32,32>, zZ>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>

    %l0c_layout = "tla.make_layout"(%shape, %zn_stride, %origin) {layoutTag = "L0Clayout"} :
      (!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>) ->
      !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>
    %l0c = "tla.make_tensor"(%ptr, %l0c_layout, %coord) :
      (!tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>) ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>

    %unalign_stride = "tla.make_stride"() : () -> !tla.stride<(16,512),(1,512)>
    %unalign_layout = "tla.make_layout"(%unalign_shape, %unalign_stride, %origin) {layoutTag = "zNUnAlign"} :
      (!tla.shape<(32,1),(16,2)>, !tla.stride<(16,512),(1,512)>, !tla.shape<32,32>) ->
      !tla.layout<!tla.shape<(32,1),(16,2)>, !tla.stride<(16,512),(1,512)>, !tla.shape<32,32>, zNUnAlign>
    %unalign = "tla.make_tensor"(%ptr, %unalign_layout, %coord) :
      (!tla.ptr<f16, l1, 512>, !tla.layout<!tla.shape<(32,1),(16,2)>, !tla.stride<(16,512),(1,512)>, !tla.shape<32,32>, zNUnAlign>, !tla.coord<0,0>) ->
      !tla.tensor<!tla.layout<!tla.shape<(32,1),(16,2)>, !tla.stride<(16,512),(1,512)>, !tla.shape<32,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>

    "tla.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "make_tensor_packed_layouts"} : () -> ()
}

// CHECK-LABEL: func.func @make_tensor_packed_layouts
// CHECK-NOT: tla.make_tensor
// CHECK-NOT: tla.tensor_desc
