// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s
//
// A tile_view keeps its parent's row pitch. Flattening coord<1,0> for a
// shape<1,64> tile over stride<128,1> must therefore use offset 128, not 64.

!parent = !tla.tensor<!tla.layout<!tla.shape<2,128>, !tla.stride<128,1>, !tla.shape<2,128>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
!chunk = !tla.tensor<!tla.layout<!tla.shape<1,64>, !tla.stride<128,1>, !tla.shape<1,64>, row_major>, !tla.coord<1,0>, !tla.ptr<f32, ub, 256>>

module {
  "tla.func"() ({
    %raw = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<i8, ub, 256>
    %ptr = tla.recast_ptr %raw : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %parent_shape = tla.make_shape -> !tla.shape<2,128>
    %parent_stride = tla.make_stride -> !tla.stride<128,1>
    %layout = tla.make_layout %parent_shape, %parent_stride : !tla.shape<2,128>, !tla.stride<128,1> -> !tla.layout<!tla.shape<2,128>, !tla.stride<128,1>, !tla.shape<2,128>, row_major>
    %parent_coord = tla.make_coord -> !tla.coord<0,0>
    %parent = tla.make_tensor %ptr, %layout, %parent_coord : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<2,128>, !tla.stride<128,1>, !tla.shape<2,128>, row_major>, !tla.coord<0,0> -> !parent
    "tla.vector"() ({
      "tla.vec.func"() ({
        %tile_shape = tla.make_shape -> !tla.shape<1,64>
        %tile_coord = tla.make_coord -> !tla.coord<1,0>
        %chunk = tla.tile_view %parent, %tile_shape, %tile_coord : !parent, !tla.shape<1,64>, !tla.coord<1,0> -> !chunk
        %value = tla.load %chunk : !chunk -> !tla.vector<64xf32>
        tla.store %chunk, %value : !chunk, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "vector", function_type = () -> (), sym_name = "vector_tile_view_parent_pitch"} : () -> ()
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[ROW:.*]] = arith.constant 1 : index
// CHECK: %[[COL:.*]] = arith.constant 0 : index
// CHECK: %[[PITCH:.*]] = arith.constant 128 : index
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[ROW]], %[[PITCH]] : index
// CHECK: %[[OFFSET:.*]] = arith.addi %[[ROW_OFFSET]], %[[COL]] : index
// CHECK: memref.reinterpret_cast {{.*}} to offset: [%[[OFFSET]]]
// CHECK-NOT: tla.tile_view
