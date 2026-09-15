// RUN: not %tla_compile %s -o %t 2>&1 | %filecheck %s

!ub = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!gm = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>

module {
  tla.func @ub_scalar_feeds_gm_with_cube(%gm_arg: !gm) {
    %raw = tla.alloc_ptr {size_bytes = 256} -> !tla.ptr<i8, ub, 256>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %shape = tla.make_shape -> !tla.shape<1>
    %stride = tla.make_stride -> !tla.stride<1>
    %layout = tla.make_layout %shape, %stride
      : !tla.shape<1>, !tla.stride<1>
        -> !tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>
    %coord = tla.make_coord -> !tla.coord<0>
    %tensor = tla.make_tensor %ptr, %layout, %coord
      : !tla.ptr<f32, ub, 256>,
        !tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, RowMajor>,
        !tla.coord<0> -> !ub
    %c0 = arith.constant 0 : index
    %value = tla.scalar_load %tensor[%c0] : !ub -> f32
    tla.scalar_store %gm_arg[%c0], %value : !gm, f32
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK: error: 'tla.scalar_load' op
// CHECK-SAME: UB scalar access must be nested inside a tla.vector region
// CHECK: Failed to parse input MLIR.
