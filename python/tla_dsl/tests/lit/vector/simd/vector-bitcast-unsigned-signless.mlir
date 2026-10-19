// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// A bitcast to an unsigned type must still produce a SIGNLESS value. Signedness
// lives in the !tla.vector type alone; every other producer in the vector region
// upholds that (a load of a u8 tile yields a signless i8 value), and the AVE ops
// enforce it -- vector<256xui8> fails ave.hir.vextui's operand constraint. A
// bitcast that leaked `ui` into the value type therefore broke any later cast of
// its result, while a bitcast round trip on its own kept passing because
// vector.bitcast accepts `ui` happily and the second bitcast consumed it.

!f32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 4>>

module {
  func.func @vector_bitcast_u32_is_signless(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !f32vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!f32vec, !tla.shape<64>, !tla.coord<0>) -> !f32vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!f32vec, !tla.shape<64>, !tla.coord<0>) -> !f32vec
      %v = tla.load %src_tile : !f32vec -> !tla.vector<64xf32>
      %bits = tla.bitcast %v : !tla.vector<64xf32> -> !tla.vector<64xui32>
      %back = tla.bitcast %bits : !tla.vector<64xui32> -> !tla.vector<64xf32>
      tla.store %dst_tile, %back : !f32vec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  func.func @vector_bitcast_u8_is_signless(
      %src_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %v = tla.load %src_tile : !i8vec -> !tla.vector<256xi8>
      %bits = tla.bitcast %v : !tla.vector<256xi8> -> !tla.vector<256xui8>
      %back = tla.bitcast %bits : !tla.vector<256xui8> -> !tla.vector<256xi8>
      tla.store %dst_tile, %back : !i8vec, !tla.vector<256xi8>
    }) : () -> ()
    return
  }
}

// The u32 leg: the emitted value types are signless on both sides.
// CHECK-LABEL: func.func private @vector_region_
// CHECK: vector.bitcast {{.*}} : vector<64xf32> to vector<64xi32>
// CHECK: vector.bitcast {{.*}} : vector<64xi32> to vector<64xf32>

// The u8 leg: a bitcast to u8 is a no-op on the value type, not a move to ui8.
// CHECK-LABEL: func.func private @vector_region_
// CHECK: vector.bitcast {{.*}} : vector<256xi8> to vector<256xi8>

// No `ui` element type may reach a value type anywhere in the region, and no
// tla.bitcast survives the pass.
// CHECK-NOT: vector<{{[0-9]+}}xui
// CHECK-NOT: tla.bitcast
