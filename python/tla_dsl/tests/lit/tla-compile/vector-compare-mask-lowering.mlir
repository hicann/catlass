// RUN: %tla_compile %s --mlir-print-ir-after=tla-to-vector -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.cmp

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @compare_mask_lowering(
      %lhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %lhs = builtin.unrealized_conversion_cast %lhs_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %rhs = builtin.unrealized_conversion_cast %rhs_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %lhs_tile = "tla.tile_view"(%lhs, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %rhs_tile = "tla.tile_view"(%rhs, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %lhs_vec = tla.load %lhs_tile : !vec -> !vec
      %rhs_vec = tla.load %rhs_tile : !vec -> !vec
      %active = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask
      %cst = arith.constant 0.000000e+00 : f32
      %lt = tla.cmp "lt" %lhs_vec, %rhs_vec mask %active : !vec, !vec mask !tla.mask -> !tla.mask
      %ge = tla.cmp "ge" %lhs_vec, %cst : !vec, f32 -> !tla.mask
      %sum = tla.add %lhs_vec, %rhs_vec mask %lt : !vec, !vec mask !tla.mask -> !vec
      tla.store %dst_tile, %sum mask %ge : !vec, !vec mask !tla.mask
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
