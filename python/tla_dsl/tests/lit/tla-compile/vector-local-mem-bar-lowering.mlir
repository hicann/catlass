// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_local_mem_bar(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst = tla.tensor_desc %dst_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %loaded_before = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      tla.store %dst_tile, %loaded_before : !fvec, !tla.vector<64xf32>
      tla.local_mem_bar 3
      %loaded_after = tla.load %dst_tile : !fvec -> !tla.vector<64xf32>
      tla.store %src_tile, %loaded_after : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload
// CHECK: ave.hir.masked_store
// CHECK: %[[KIND:.*]] = arith.constant 3 : i32
// CHECK: ave.hir.membar %[[KIND]]
// CHECK: ave.hir.vload
// CHECK: ave.hir.masked_store
// CHECK-NOT: tla.local_mem_bar
