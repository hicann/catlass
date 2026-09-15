// RUN: %tla_compile %s -o - | %filecheck %s --check-prefix=FP4
//
// Packed fp4 on the fill route: the tile's element type selects the x2-encoded
// wrapper over byte storage (same rule as the GM->L1 fp4 copy), so the callee
// is named after the encoding even though the storage is bytes. At runtime the
// MX pad region is always swallowed by the C0=64 alignment (start == end), so
// the wrapper early-returns; the lowering contract itself is what this pins.

module {
  tla.func @fill_l1_zn_fp4() {
    %raw = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<i8, l1, 512>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, l1, 512> -> !tla.ptr<!tla.f4e2m1, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,2),(64,1)>
    %stride = tla.make_stride -> !tla.stride<(64,512),(1,1024)>
    %origin = tla.make_shape -> !tla.shape<32,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,2),(64,1)>, !tla.stride<(64,512),(1,1024)> origin !tla.shape<32,64> ->
      !tla.layout<!tla.shape<(16,2),(64,1)>, !tla.stride<(64,512),(1,1024)>, !tla.shape<32,64>, zN>
    %zero_coord = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %ptr, %layout, %zero_coord :
      !tla.ptr<!tla.f4e2m1, l1, 512>,
      !tla.layout<!tla.shape<(16,2),(64,1)>, !tla.stride<(64,512),(1,1024)>, !tla.shape<32,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,2),(64,1)>, !tla.stride<(64,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<!tla.f4e2m1, l1, 512>>
    %zero = arith.constant 0 : i32
    %fill_coord = tla.make_coord -> !tla.coord<0,32>
    %fill_shape = tla.make_shape -> !tla.shape<32,32>
    "tla.cube"() ({
      tla.fill %dst, %zero, %fill_coord, %fill_shape :
        !tla.tensor<!tla.layout<!tla.shape<(16,2),(64,1)>, !tla.stride<(64,512),(1,1024)>, !tla.shape<32,64>, zN>, !tla.coord<0,0>, !tla.ptr<!tla.f4e2m1, l1, 512>>,
        i32, !tla.coord<0,32>, !tla.shape<32,32>
    }) : () -> ()
    tla.return
  }
}

// FP4: func.func private @fill_l1_zN_float4_e2m1x2_t(
// FP4-SAME: hacc.always_inline
// FP4: call @fill_l1_zN_float4_e2m1x2_t
