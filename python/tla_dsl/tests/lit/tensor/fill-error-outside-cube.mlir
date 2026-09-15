// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s
//
// tla.fill drives L1 buffer upkeep from the cube pipeline, so like tla.copy it
// is only meaningful inside a cube region.

module {
  tla.func @fill_outside_cube() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<f16, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(16,4)>
    %stride = tla.make_stride -> !tla.stride<(16,256),(1,256)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,256)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,256)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<f16, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,256)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    tla.fill %dst, %zero, %fill_coord, %fill_shape :
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,4)>, !tla.stride<(16,256),(1,256)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>,
      i32, !tla.coord<0,32>, !tla.shape<32,32>
    tla.return
  }
}

// CHECK: 'tla.fill' op must be nested inside a tla.cube region
