// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-to-vector 2>&1 | %filecheck %s

// CHECK: ave.hir.preg.not
// CHECK: ave.hir.preg.and
// CHECK: ave.hir.preg.or
// CHECK: ave.hir.preg.xor
// CHECK-NOT: tla.mask_not
// CHECK-NOT: tla.mask_and
// CHECK-NOT: tla.mask_or
// CHECK-NOT: tla.mask_xor

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @mask_logic_lowering(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_not_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_and_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_or_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_xor_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %dst_not = builtin.unrealized_conversion_cast %dst_not_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %dst_and = builtin.unrealized_conversion_cast %dst_and_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %dst_or = builtin.unrealized_conversion_cast %dst_or_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %dst_xor = builtin.unrealized_conversion_cast %dst_xor_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_not_tile = "tla.tile_view"(%dst_not, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_and_tile = "tla.tile_view"(%dst_and, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_or_tile = "tla.tile_view"(%dst_or, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %dst_xor_tile = "tla.tile_view"(%dst_xor, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %all = "tla.create_mask"() {pattern = "ALL", dtype = f32} : () -> !tla.mask
        %h = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask
        %q = "tla.create_mask"() {pattern = "Q", dtype = f32} : () -> !tla.mask
        %m4 = "tla.create_mask"() {pattern = "M4", dtype = f32} : () -> !tla.mask
        %not = "tla.mask_not"(%q, %all) : (!tla.mask, !tla.mask) -> !tla.mask
        %and = "tla.mask_and"(%h, %m4, %all) : (!tla.mask, !tla.mask, !tla.mask) -> !tla.mask
        %or = "tla.mask_or"(%q, %m4, %all) : (!tla.mask, !tla.mask, !tla.mask) -> !tla.mask
        %xor = "tla.mask_xor"(%h, %m4, %all) : (!tla.mask, !tla.mask, !tla.mask) -> !tla.mask
        %reg = "tla.load"(%src_tile) : (!vec) -> !vec
        %zero = arith.constant 0.000000e+00 : f32
        %zero_reg = "tla.full"(%zero) : (f32) -> !vec
        %out_not = "tla.where"(%not, %reg, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %out_and = "tla.where"(%and, %reg, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %out_or = "tla.where"(%or, %reg, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        %out_xor = "tla.where"(%xor, %reg, %zero_reg) : (!tla.mask, !vec, !vec) -> !vec
        "tla.store"(%dst_not_tile, %out_not) : (!vec, !vec) -> ()
        "tla.store"(%dst_and_tile, %out_and) : (!vec, !vec) -> ()
        "tla.store"(%dst_or_tile, %out_or) : (!vec, !vec) -> ()
        "tla.store"(%dst_xor_tile, %out_xor) : (!vec, !vec) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
