// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region 2>&1 | %filecheck %s

// CHECK: ave.hir.preg.not
// CHECK: ave.hir.preg.and
// CHECK: ave.hir.preg.or
// CHECK: ave.hir.preg.xor
// CHECK: ave.hir.vnot
// CHECK: ave.hir.vand
// CHECK: ave.hir.vor
// CHECK: ave.hir.vxor
// CHECK-NOT: tla.mask_not
// CHECK-NOT: tla.mask_and
// CHECK-NOT: tla.mask_or
// CHECK-NOT: tla.mask_xor
// CHECK-NOT: tla.reg_not
// CHECK-NOT: tla.reg_and
// CHECK-NOT: tla.reg_or
// CHECK-NOT: tla.reg_xor

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @logic_lowering(
      %src0_memref: memref<64xi32, #hivm.address_space<ub>>,
      %src1_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_not_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_and_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_or_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_mask_xor_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_not_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_and_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_or_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_reg_xor_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %src0 = builtin.unrealized_conversion_cast %src0_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %src1 = builtin.unrealized_conversion_cast %src1_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_mask_not = builtin.unrealized_conversion_cast %dst_mask_not_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_mask_and = builtin.unrealized_conversion_cast %dst_mask_and_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_mask_or = builtin.unrealized_conversion_cast %dst_mask_or_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_mask_xor = builtin.unrealized_conversion_cast %dst_mask_xor_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_reg_not = builtin.unrealized_conversion_cast %dst_reg_not_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_reg_and = builtin.unrealized_conversion_cast %dst_reg_and_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_reg_or = builtin.unrealized_conversion_cast %dst_reg_or_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    %dst_reg_xor = builtin.unrealized_conversion_cast %dst_reg_xor_memref : memref<64xi32, #hivm.address_space<ub>> to !vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src0_tile = "tla.tile_view"(%src0, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %src1_tile = "tla.tile_view"(%src1, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_not_tile = "tla.tile_view"(%dst_mask_not, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_and_tile = "tla.tile_view"(%dst_mask_and, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_or_tile = "tla.tile_view"(%dst_mask_or, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_mask_xor_tile = "tla.tile_view"(%dst_mask_xor, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_not_tile = "tla.tile_view"(%dst_reg_not, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_and_tile = "tla.tile_view"(%dst_reg_and, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_or_tile = "tla.tile_view"(%dst_reg_or, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_reg_xor_tile = "tla.tile_view"(%dst_reg_xor, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %all = "tla.create_mask"() {pattern = "ALL", dtype = i32} : () -> !tla.mask
        %h = "tla.create_mask"() {pattern = "H", dtype = i32} : () -> !tla.mask
        %q = "tla.create_mask"() {pattern = "Q", dtype = i32} : () -> !tla.mask
        %m4 = "tla.create_mask"() {pattern = "M4", dtype = i32} : () -> !tla.mask
        %mask_not = "tla.mask_not"(%q, %all) : (!tla.mask, !tla.mask) -> !tla.mask
        %mask_and = "tla.mask_and"(%h, %m4, %all) : (!tla.mask, !tla.mask, !tla.mask) -> !tla.mask
        %mask_or = "tla.mask_or"(%q, %m4, %all) : (!tla.mask, !tla.mask, !tla.mask) -> !tla.mask
        %mask_xor = "tla.mask_xor"(%h, %m4, %all) : (!tla.mask, !tla.mask, !tla.mask) -> !tla.mask
        %reg0 = "tla.load"(%src0_tile) : (!vec) -> !vec
        %reg1 = "tla.load"(%src1_tile) : (!vec) -> !vec
        %zero = arith.constant 0 : i32
        %zero_reg = "tla.full"(%zero) : (i32) -> !vec
        %out_mask_not = "tla.where"(%mask_not, %reg0, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %out_mask_and = "tla.where"(%mask_and, %reg0, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %out_mask_or = "tla.where"(%mask_or, %reg0, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %out_mask_xor = "tla.where"(%mask_xor, %reg0, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %reg_not = "tla.reg_not"(%reg0) : (!vec) -> !vec
        %reg_and = "tla.reg_and"(%reg0, %reg1, %all) : (!vec, !vec, !tla.mask) -> !vec
        %reg_or = "tla.reg_or"(%reg0, %reg1, %all) : (!vec, !vec, !tla.mask) -> !vec
        %reg_xor = "tla.reg_xor"(%reg0, %reg1, %all) : (!vec, !vec, !tla.mask) -> !vec
        "tla.store"(%dst_mask_not_tile, %out_mask_not) : (!vec, !vec) -> ()
        "tla.store"(%dst_mask_and_tile, %out_mask_and) : (!vec, !vec) -> ()
        "tla.store"(%dst_mask_or_tile, %out_mask_or) : (!vec, !vec) -> ()
        "tla.store"(%dst_mask_xor_tile, %out_mask_xor) : (!vec, !vec) -> ()
        "tla.store"(%dst_reg_not_tile, %reg_not) : (!vec, !vec) -> ()
        "tla.store"(%dst_reg_and_tile, %reg_and) : (!vec, !vec) -> ()
        "tla.store"(%dst_reg_or_tile, %reg_or) : (!vec, !vec) -> ()
        "tla.store"(%dst_reg_xor_tile, %reg_xor) : (!vec, !vec) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
