// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s
//
// A RowMajor tile has no fractal structure, so InitConstValue has no C0 axis to
// align the region start to -- the tag must be zN or nZ.

module {
  tla.func @fill_rowmajor_tile() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<32,64>
    %stride = tla.make_stride -> !tla.stride<64,1>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<RowMajor>} :
      !tla.shape<32,64>, !tla.stride<64,1> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<32,64>, !tla.stride<64,1>, !tla.shape<32,64>, RowMajor>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f16, l1, 512>,
      !tla.layout<!tla.shape<32,64>, !tla.stride<64,1>, !tla.shape<32,64>, RowMajor>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<32,64>, !tla.stride<64,1>, !tla.shape<32,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<32,64>, !tla.stride<64,1>, !tla.shape<32,64>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }
}

// CHECK: 'tla.fill' op fill destination must be tagged zN or nZ
