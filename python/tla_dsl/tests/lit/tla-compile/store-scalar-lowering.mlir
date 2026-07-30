// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-lower-scalar-access 2>&1 | %filecheck %s

// CHECK-LABEL: func.func @scalar_store_lowering
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast
// CHECK: memref.store %{{.*}}, %[[VIEW]][%arg1]
// CHECK-NOT: tla.scalar_store

// The scalar view must include the tensor descriptor's non-zero tile offset.
// CHECK-LABEL: func.func @scalar_store_nonzero_tile_coord
// CHECK: %[[META_BASE:[^,]+]], %[[META_OFFSET:[^,]+]], %[[META_SIZES:[^,]+]], %[[META_STRIDES:[^ ]+]] = memref.extract_strided_metadata %arg0
// CHECK: %[[OFFSET_C16:.*]] = arith.constant 16 : index
// CHECK: %[[STORAGE_OFFSET:.*]] = arith.addi %[[META_OFFSET]], %[[OFFSET_C16]] : index
// CHECK: %[[OFFSET_VIEW:.*]] = memref.reinterpret_cast %[[META_BASE]] to offset: [%[[STORAGE_OFFSET]]]
// CHECK: memref.store %{{.*}}, %[[OFFSET_VIEW]][%arg1]
// CHECK-NOT: tla.scalar_store

// CHECK-LABEL: func.func @scalar_load_nonzero_tile_coord
// CHECK: %[[LOAD_BASE:[^,]+]], %[[LOAD_META_OFFSET:[^,]+]], %[[LOAD_META_SIZES:[^,]+]], %[[LOAD_META_STRIDES:[^ ]+]] = memref.extract_strided_metadata %arg0
// CHECK: %[[LOAD_OFFSET_C16:.*]] = arith.constant 16 : index
// CHECK: %[[LOAD_OFFSET:.*]] = arith.addi %[[LOAD_META_OFFSET]], %[[LOAD_OFFSET_C16]] : index
// CHECK: %[[LOAD_VIEW:.*]] = memref.reinterpret_cast %[[LOAD_BASE]] to offset: [%[[LOAD_OFFSET]]]
// CHECK: memref.load %[[LOAD_VIEW]][%arg2]
// CHECK-NOT: tla.scalar_load

// CHECK-LABEL: func.func @scalar_rank2_load_store
// CHECK: %[[RANK2_LOAD_VIEW:.*]] = memref.reinterpret_cast
// CHECK: %[[RANK2_VALUE:.*]] = memref.load %[[RANK2_LOAD_VIEW]][%{{.*}}, %{{.*}}]
// CHECK: %[[RANK2_STORE_VIEW:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[RANK2_VALUE]], %[[RANK2_STORE_VIEW]][%{{.*}}, %{{.*}}]
// CHECK-NOT: tla.scalar_load
// CHECK-NOT: tla.scalar_store

!vec64 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!tile8 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<16>, !tla.ptr<f32, ub, 4>>
!matrix = !tla.tensor<!tla.layout<!tla.shape<2,4>, !tla.stride<4,1>, !tla.shape<2,4>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @scalar_store_lowering(
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>,
      %index: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64]
      : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index,
         index, index, index, index) -> !vec64
    %value = arith.constant 4.500000e+00 : f32
    "tla.vector"() ({
      tla.scalar_store %dst[%index], %value : !vec64, f32
    }) : () -> ()
    return
  }

  func.func @scalar_store_nonzero_tile_coord(
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>,
      %index: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %tile = tla.tensor_desc %dst_memref[%c0, %c16, %c64, %c1, %c1, %c8, %c1, %c64]
      : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index,
         index, index, index, index) -> !tile8
    %value = arith.constant 4.500000e+00 : f32
    "tla.vector"() ({
      tla.scalar_store %tile[%index], %value : !tile8, f32
    }) : () -> ()
    return
  }

  func.func @scalar_load_nonzero_tile_coord(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %out_memref: memref<1xf32, #hivm.address_space<ub>>,
      %index: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %tile = tla.tensor_desc %src_memref[%c0, %c16, %c64, %c1, %c1, %c8, %c1, %c64]
      : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index,
         index, index, index, index) -> !tile8
    "tla.vector"() ({
      %value = tla.scalar_load %tile[%index] : !tile8 -> f32
      memref.store %value, %out_memref[%c0]
        : memref<1xf32, #hivm.address_space<ub>>
    }) : () -> ()
    return
  }

  func.func @scalar_rank2_load_store(
      %tensor_memref: memref<2x4xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %tensor = tla.tensor_desc %tensor_memref[%c0, %c0, %c4, %c1, %c2, %c4, %c2, %c4]
      : (memref<2x4xf32, #hivm.address_space<ub>>, index, index, index, index,
         index, index, index, index) -> !matrix
    "tla.vector"() ({
      %value = tla.scalar_load %tensor[%c1, %c2] : !matrix -> f32
      tla.scalar_store %tensor[%c1, %c2], %value : !matrix, f32
    }) : () -> ()
    return
  }
}
