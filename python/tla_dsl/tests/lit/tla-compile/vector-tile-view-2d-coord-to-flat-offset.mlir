// RUN: %tla_compile %s -o - | %filecheck %s
//
// Test that tile_view with rank-2 coord inside vec.func correctly computes
// flat 1D offset = row * rowStride + col.
// UB tensors have dynamic shape <?,32> (matching real make_tensor_like usage).
// tile_view uses two make_coord: tile_idx(row,0) and offset_idx(row*tile_size,0).
//   chunk 0: coord<0,0> → offset = 0*32 + 0 = 0
//   chunk 1: coord<2,0> → offset = 2*32 + 0 = 64
//   chunk 2: coord<4,0> → offset = 4*32 + 0 = 128

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>):
    %0 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<i8, ub, 256>
    %1 = tla.recast_ptr %0 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %2 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<i8, ub, 256>
    %3 = tla.recast_ptr %2 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>
    %4 = tla.alloc_ptr{size_bytes = 2048} -> !tla.ptr<i8, ub, 256>
    %5 = tla.recast_ptr %4 : !tla.ptr<i8, ub, 256> -> !tla.ptr<f32, ub, 256>

    // Create UB tensors with dynamic <?,32> via make_tensor_like.
    %c16 = arith.constant 16 : index
    %6 = tla.make_shape -> !tla.shape<16,32>
    %7 = tla.make_coord %c16 -> !tla.coord<?,0>
    %8 = tla.make_coord %c16 -> !tla.coord<?,0>
    %9 = tla.tile_view %arg0, %6, %8 : !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, !tla.shape<16,32>, !tla.coord<?,0> -> !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>>
    %10 = tla.make_tensor_like %1 like %9 layoutTag("row_major") : !tla.ptr<f32, ub, 256>, !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
    %11 = tla.make_tensor_like %3 like %9 layoutTag("row_major") : !tla.ptr<f32, ub, 256>, !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
    %12 = tla.make_tensor_like %5 like %9 layoutTag("row_major") : !tla.ptr<f32, ub, 256>, !tla.tensor<!tla.layout<!tla.shape<16,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<?,0>, !tla.ptr<f32, gm, 4>> -> !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

    "tla.vector"() ({
      // Chunk at coord<0,0> → flat offset 0
      "tla.vec.func"() ({
        %13 = tla.make_shape -> !tla.shape<2,32>
        %14 = tla.make_coord -> !tla.coord<0,0>
        %15 = tla.make_coord -> !tla.coord<0,0>
        %16 = tla.tile_view %10, %13, %15 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
        %17 = tla.make_shape -> !tla.shape<2,32>
        %18 = tla.make_coord -> !tla.coord<0,0>
        %19 = tla.make_coord -> !tla.coord<0,0>
        %20 = tla.tile_view %11, %17, %19 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
        %21 = tla.make_shape -> !tla.shape<2,32>
        %22 = tla.make_coord -> !tla.coord<0,0>
        %23 = tla.make_coord -> !tla.coord<0,0>
        %24 = tla.tile_view %12, %21, %23 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
        %25 = tla.load %16 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
        %26 = tla.load %20 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
        %27 = tla.add %25, %26 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
        tla.store %24, %27 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
      }) {mode = "simd"} : () -> ()
      // Chunk at coord<1,0> → flat offset 1*32=32
      "tla.vec.func"() ({
        %13 = tla.make_shape -> !tla.shape<2,32>
        %14 = tla.make_coord -> !tla.coord<1,0>
        %15 = tla.make_coord -> !tla.coord<2,0>
        %16 = tla.tile_view %10, %13, %15 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<2,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
        %17 = tla.make_shape -> !tla.shape<2,32>
        %18 = tla.make_coord -> !tla.coord<1,0>
        %19 = tla.make_coord -> !tla.coord<2,0>
        %20 = tla.tile_view %11, %17, %19 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<2,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
        %21 = tla.make_shape -> !tla.shape<2,32>
        %22 = tla.make_coord -> !tla.coord<1,0>
        %23 = tla.make_coord -> !tla.coord<2,0>
        %24 = tla.tile_view %12, %21, %23 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<2,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
        %25 = tla.load %16 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
        %26 = tla.load %20 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
        %27 = tla.add %25, %26 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>, <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
        tla.store %24, %27 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>, <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<2,0>, !tla.ptr<f32, ub, 256>>
      }) {mode = "simd"} : () -> ()
      // Chunk at coord<2,0> → flat offset 2*32=64
      "tla.vec.func"() ({
        %13 = tla.make_shape -> !tla.shape<2,32>
        %14 = tla.make_coord -> !tla.coord<2,0>
        %15 = tla.make_coord -> !tla.coord<4,0>
        %16 = tla.tile_view %10, %13, %15 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<4,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
        %17 = tla.make_shape -> !tla.shape<2,32>
        %18 = tla.make_coord -> !tla.coord<2,0>
        %19 = tla.make_coord -> !tla.coord<4,0>
        %20 = tla.tile_view %11, %17, %19 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<4,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
        %21 = tla.make_shape -> !tla.shape<2,32>
        %22 = tla.make_coord -> !tla.coord<2,0>
        %23 = tla.make_coord -> !tla.coord<4,0>
        %24 = tla.tile_view %12, %21, %23 : !tla.tensor<!tla.layout<!tla.shape<?,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>, !tla.shape<2,32>, !tla.coord<4,0> -> !tla.tensor<!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
        %25 = tla.load %16 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
        %26 = tla.load %20 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
        %27 = tla.add %25, %26 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>, <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>> -> <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
        tla.store %24, %27 : <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>, <!tla.layout<!tla.shape<2,32>, !tla.stride<32,1>, !tla.shape<?,32>, row_major>, !tla.coord<4,0>, !tla.ptr<f32, ub, 256>>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {tla.exec_units = "vector", function_type = (!tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "tile_view_2d_offset_test"} : () -> ()
}
// chunk 0: coord<0,0> → offset 0
// CHECK: func.func private @vector_region_2
// CHECK: llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK: memref.reinterpret_cast {{%.*}} to offset: [{{%.*}}], sizes: [{{%.*}}], strides: [{{%.*}}]
// chunk 1: coord<2,0> → offset 2*32 = 64
// CHECK: func.func private @vector_region_1
// CHECK: llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK: memref.reinterpret_cast {{%.*}} to offset: [{{%.*}}], sizes: [{{%.*}}], strides: [{{%.*}}]
// chunk 2: coord<4,0> → offset 4*32 = 128
// CHECK: func.func private @vector_region_0
// CHECK: llvm.mlir.constant(128 : index) : i64
// CHECK-NEXT: builtin.unrealized_conversion_cast
// CHECK: memref.reinterpret_cast {{%.*}} to offset: [{{%.*}}], sizes: [{{%.*}}], strides: [{{%.*}}]
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx1.v64f32
// CHECK: hivm_regbaseintrins.intr.hivm.vadd.s.x
// CHECK: hivm_regbaseintrins.intr.hivm.vstsx1.v64f32
// CHECK-NOT: tla.tile_view
// CHECK-NOT: tla.load
// CHECK-NOT: tla.store

