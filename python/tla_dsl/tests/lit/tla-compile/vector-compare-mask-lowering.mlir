// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.cmp

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @compare_mask_lowering(
      %lhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %lhs_c0 = arith.constant 0 : index
    %lhs_c1 = arith.constant 1 : index
    %lhs_c64 = arith.constant 64 : index
    %lhs = tla.tensor_desc %lhs_memref[%lhs_c0, %lhs_c0, %lhs_c64, %lhs_c1, %lhs_c1, %lhs_c64, %lhs_c1, %lhs_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    %rhs_c0 = arith.constant 0 : index
    %rhs_c1 = arith.constant 1 : index
    %rhs_c64 = arith.constant 64 : index
    %rhs = tla.tensor_desc %rhs_memref[%rhs_c0, %rhs_c0, %rhs_c64, %rhs_c1, %rhs_c1, %rhs_c64, %rhs_c1, %rhs_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c64, %dst_c1, %dst_c1, %dst_c64, %dst_c1, %dst_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %lhs_tile = "tla.tile_view"(%lhs, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %rhs_tile = "tla.tile_view"(%rhs, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %lhs_vec = tla.load %lhs_tile : !vec -> !tla.vector<64xf32>
      %rhs_vec = tla.load %rhs_tile : !vec -> !tla.vector<64xf32>
      %active = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      %cst = arith.constant 0.000000e+00 : f32
      %lt = tla.cmp "lt" %lhs_vec, %rhs_vec mask %active : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.mask<64>
      %ge = tla.cmp "ge" %lhs_vec, %cst : !tla.vector<64xf32>, f32 -> !tla.mask<64>
      %sum = tla.add %lhs_vec, %rhs_vec mask %lt : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sum mask %ge : !vec, !tla.vector<64xf32> mask !tla.mask<64>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @compare_mask_lowering
// CHECK-LABEL: func.func private @vector_region_{{[0-9]+}}
// CHECK: ave.hir.pge <H>
// CHECK: ave.hir.vcmp <LT> {{.*}} : vector<64xf32>
// CHECK: ave.hir.vcmps <GE> {{.*}} : vector<64xf32>, f32, vector<64xi1> -> vector<64xi1>
// CHECK: ave.hir.vadd
// CHECK: ave.hir.masked_store {{.*}} : memref<64xf32
