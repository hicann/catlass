// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region > %t 2>&1
// RUN: %filecheck %s --check-prefix=MAIN < %t
// RUN: %filecheck %s --check-prefix=HELPER < %t

// MAIN-LABEL: func.func @scalar_load_lowering
// MAIN: call @vector_region_
// MAIN: memref.load {{.*}}[%{{.*}}]
// MAIN-NOT: tla.scalar_load
// HELPER-LABEL: func.func private @vector_region_
// HELPER: %[[REDUCED:.*]] = ave.hir.reduction <add>
// HELPER: %[[STORE_MASK:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<64xi1>, index
// HELPER-NOT: ave.hir.broadcast_vector
// HELPER-NOT: ONEPT
// HELPER: ave.hir.masked_store <NORM_B32> {{.*}}, %[[STORE_MASK]], %[[REDUCED]]
// HELPER: %[[BARRIER_KIND:.*]] = arith.constant 5 : i32
// HELPER-NEXT: ave.hir.membar %[[BARRIER_KIND]]
// HELPER-NOT: tla.local_mem_bar

!vec64 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!vec1 = !tla.tensor<!tla.layout<!tla.shape<1>, !tla.stride<1>, !tla.shape<1>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @scalar_load_lowering(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %tmp_memref: memref<1xf32, #hivm.address_space<ub>>,
      %index: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0]
      : memref<64xf32, #hivm.address_space<ub>> -> !vec64
    %tmp = tla.tensor_desc %tmp_memref shape [%c1, %c1, %c1, %c1] stride [%c1, %c1, %c1, %c1] origin_shape [%c1, %c1] coord [%c0, %c0]
      : memref<1xf32, #hivm.address_space<ub>> -> !vec1
    "tla.vector"() ({
      "tla.vec.func"() ({
        %vec_shape64 = "tla.make_shape"() : () -> !tla.shape<64>
        %vec_shape1 = "tla.make_shape"() : () -> !tla.shape<1>
        %vec_coord0 = "tla.make_coord"() : () -> !tla.coord<0>
        %src_vec_tile = "tla.tile_view"(%src, %vec_shape64, %vec_coord0)
          : (!vec64, !tla.shape<64>, !tla.coord<0>) -> !vec64
        %tmp_vec_tile = "tla.tile_view"(%tmp, %vec_shape1, %vec_coord0)
          : (!vec1, !tla.shape<1>, !tla.coord<0>) -> !vec1
        %v = tla.load %src_vec_tile : !vec64 -> !tla.vector<64xf32>
        %reduce_mask = "tla.create_mask"() {pattern = "ALL", dtype = f32}
          : () -> !tla.mask<64>
        %reduced = tla.reduce %v mask %reduce_mask {kind = "add"}
          : !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<1xf32>
        tla.store %tmp_vec_tile, %reduced : !vec1, !tla.vector<1xf32>
        tla.local_mem_bar 5
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    "tla.vector"() ({
      %scalar_shape1 = "tla.make_shape"() : () -> !tla.shape<1>
      %scalar_coord0 = "tla.make_coord"() : () -> !tla.coord<0>
      %tmp_scalar_tile = "tla.tile_view"(%tmp, %scalar_shape1, %scalar_coord0)
        : (!vec1, !tla.shape<1>, !tla.coord<0>) -> !vec1
      %s = tla.scalar_load %tmp_scalar_tile[%index] : !vec1 -> f32
      // Keep the scalar load live until tla-lower-scalar-access lowers it. A bare,
      // unused scalar_load may be removed before TlaVectorRegionPass runs.
      memref.store %s, %tmp_memref[%index]
        : memref<1xf32, #hivm.address_space<ub>>
    }) : () -> ()
    return
  }
}
