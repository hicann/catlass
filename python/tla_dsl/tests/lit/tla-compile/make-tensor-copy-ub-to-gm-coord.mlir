// RUN: %tla_compile %s -o - | %filecheck %s


module {
  tla.func @make_tensor_copy_ub_to_gm_coord(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    %ub_alloc = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<i8, ub, 256>
    %ub_ptr = tla.recast_ptr %ub_alloc : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>

    // UB source tile (16x32, coord 0,0).
    %ub_shape = tla.make_shape -> !tla.shape<16,32>
    %ub_stride = tla.make_stride -> !tla.stride<32,1>
    %ub_layout = tla.make_layout %ub_shape, %ub_stride : !tla.shape<16,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>
    %ub_coord = tla.make_coord -> !tla.coord<0,0>
    %ub_tensor = tla.make_tensor %ub_ptr, %ub_layout, %ub_coord : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

    // GM destination tile (16x32) at a dynamic row coord = 16 (simulates
    // vec_idx * VECTOR_TILE_M). The store must land at GM base + 16*32 elems.
    %gm_ptr = tla.tensor_ptr %arg0 : !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>> -> !tla.ptr<f32, gm, 4>
    %gm_shape = tla.make_shape -> !tla.shape<16,32>
    %gm_stride = tla.make_stride -> !tla.stride<32,1>
    %gm_layout = tla.make_layout %gm_shape, %gm_stride : !tla.shape<16,32>, !tla.stride<32,1> -> !tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>
    %c16 = arith.constant 16 : index
    %gm_coord = tla.make_coord %c16 -> !tla.coord<?,0>
    %gm_tensor = tla.make_tensor %gm_ptr, %gm_layout, %gm_coord : !tla.ptr<f32, gm, 4>, !tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>, !tla.coord<?,0> -> !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>>

    "tla.vector"() ({
      // dst = GM make_tensor (coord 16), src = UB make_tensor -> UB->GM store.
      tla.copy %gm_tensor, %ub_tensor : !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>>, !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<16,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func private @copy_ub_row_major_to_gm_row_major_float
// The GM dst coord (16) and stride0 (32) travel as i64 constant payload args
// (absCoord0=16, stride0=32), so the stub computes the store offset 16*32 = 512
// elements. Before the cifax lowering the coord was dropped to a 0 offset.
// CHECK-SAME: constant_value = 16 : i64
// CHECK-SAME: constant_value = 32 : i64
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @make_tensor_copy_ub_to_gm_coord
// CHECK: call @copy_ub_row_major_to_gm_row_major_float
// CHECK-NOT: tla.make_tensor
// CHECK-NOT: memref.reinterpret_cast
