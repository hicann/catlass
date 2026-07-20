// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-lower-func 2>&1 | %filecheck %s
// RUN: %tla_compile %s -o %t --mlir-print-ir-after=tla-lower-scalar-access 2>&1 | %filecheck %s --check-prefix=SCALAR

!root = !tla.tensor<!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<16,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!tile = !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, row_major>, !tla.coord<2,3>, !tla.ptr<f32, gm, 4>>
!child = !tla.tensor<!tla.layout<!tla.shape<2,3>, !tla.stride<16,1>, !tla.shape<2,3>, row_major>, !tla.coord<1,2>, !tla.ptr<f32, gm, 4>>
!padded_row = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<20,1>, !tla.shape<8,16>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!dynamic_row = !tla.tensor<!tla.layout<!tla.shape<?,64>, !tla.stride<?,1>, !tla.shape<128,64>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!dynamic_col = !tla.tensor<!tla.layout<!tla.shape<?,64>, !tla.stride<1,?>, !tla.shape<128,64>, column_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!dynamic_1d = !tla.tensor<!tla.layout<!tla.shape<?>, !tla.stride<1>, !tla.shape<128>, row_major>, !tla.coord<0>, !tla.ptr<i32, gm, 4>>
!rank1_tile = !tla.tensor<!tla.layout<!tla.shape<16>, !tla.stride<1>, !tla.shape<16>, row_major>, !tla.coord<4>, !tla.ptr<i32, gm, 4>>

module {
  tla.func @root_static(%arg0: !root) {
    %shape = tla.make_shape -> !tla.shape<4,8>
    %coord = tla.make_coord -> !tla.coord<2,3>
    %tile = tla.tile_view %arg0, %shape, %coord : !root, !tla.shape<4,8>, !tla.coord<2,3> -> !tile
    %ptr = tla.tensor_ptr %tile : !tile -> !tla.ptr<f32, gm, 4>
    %child_shape = tla.make_shape -> !tla.shape<2,3>
    %child_coord = tla.make_coord -> !tla.coord<1,2>
    %child = tla.tile_view %tile, %child_shape, %child_coord : !tile, !tla.shape<2,3>, !tla.coord<1,2> -> !child
    %c1 = arith.constant 1 : index
    %value = tla.scalar_load %child[%c1, %c1] : !child -> f32
    tla.return
  }

  tla.func @root_padded_row(%arg0: !padded_row) {
    tla.return
  }

  tla.func @root_dynamic_row(%arg0: !dynamic_row) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %value = tla.scalar_load %arg0[%c0, %c3] : !dynamic_row -> f32
    tla.return
  }

  tla.func @root_dynamic_column(%arg0: !dynamic_col) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %value = tla.scalar_load %arg0[%c1, %c2] : !dynamic_col -> f32
    tla.return
  }

  tla.func @root_dynamic_rank1(%arg0: !dynamic_1d) {
    %shape = tla.make_shape -> !tla.shape<16>
    %coord = tla.make_coord -> !tla.coord<4>
    %tile = tla.tile_view %arg0, %shape, %coord : !dynamic_1d, !tla.shape<16>, !tla.coord<4> -> !rank1_tile
    %c1 = arith.constant 1 : index
    %value = tla.scalar_load %tile[%c1] : !rank1_tile -> i32
    tla.return
  }
}

// CHECK-LABEL: func.func @root_static(
// CHECK-SAME: %[[STATIC_ARG:.*]]: memref<16x16xf32, #hivm.address_space<gm>>)
// CHECK: %[[STATIC_DESC:.*]] = tla.tensor_desc %[[STATIC_ARG]][
// CHECK: %[[STATIC_TILE:.*]] = tla.tile_view %[[STATIC_DESC]],
// CHECK: tla.tensor_ptr %[[STATIC_TILE]]
// CHECK-NOT: tla.tile_view %[[STATIC_ARG]],

// CHECK-LABEL: func.func @root_padded_row(
// CHECK-SAME: %[[PADDED_ARG:.*]]: memref<8x16xf32, strided<[20, 1], offset: ?>, #hivm.address_space<gm>>)
// CHECK: tla.tensor_desc %[[PADDED_ARG]][

// CHECK-LABEL: func.func @root_dynamic_row(
// CHECK-SAME: %[[ROW_ARG:.*]]: memref<?x64xf32, strided<{{.*}}>, #hivm.address_space<gm>>)
// CHECK: memref.dim %[[ROW_ARG]],
// CHECK: memref.extract_strided_metadata %[[ROW_ARG]]
// CHECK: %[[ROW_DESC:.*]] = tla.tensor_desc %[[ROW_ARG]][
// CHECK: tla.scalar_load %[[ROW_DESC]]

// CHECK-LABEL: func.func @root_dynamic_column(
// CHECK-SAME: %[[COL_ARG:.*]]: memref<?x64xf32, #hivm.address_space<gm>>)
// CHECK: %[[COL_DIM:.*]] = memref.dim %[[COL_ARG]],
// CHECK: %[[COL_DESC:.*]] = tla.tensor_desc %[[COL_ARG]][
// CHECK: tla.scalar_load %[[COL_DESC]]

// SCALAR-LABEL: func.func @root_static
// SCALAR: memref.reinterpret_cast
// SCALAR: memref.load
// SCALAR-NOT: tla.scalar_load

// SCALAR-LABEL: func.func @root_dynamic_row
// SCALAR: memref.reinterpret_cast
// SCALAR: memref.load
// SCALAR-NOT: tla.scalar_load

// SCALAR-LABEL: func.func @root_dynamic_column
// SCALAR: memref.reinterpret_cast
// SCALAR: memref.load
// SCALAR-NOT: tla.scalar_load

// SCALAR-LABEL: func.func @root_dynamic_rank1
// SCALAR: memref.reinterpret_cast
// SCALAR: memref.load
// SCALAR-NOT: tla.scalar_load

// CHECK-LABEL: func.func @root_dynamic_rank1(
// CHECK-SAME: %[[RANK1_ARG:.*]]: memref<?xi32, strided<{{.*}}>, #hivm.address_space<gm>>)
// CHECK: memref.dim %[[RANK1_ARG]],
// CHECK: %[[RANK1_DESC:.*]] = tla.tensor_desc %[[RANK1_ARG]][
// CHECK: %[[RANK1_TILE:.*]] = tla.tile_view %[[RANK1_DESC]],
// CHECK: tla.scalar_load %[[RANK1_TILE]]
