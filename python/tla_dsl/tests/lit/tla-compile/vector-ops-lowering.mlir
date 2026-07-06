// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --check-prefix=GATHER < %t

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!i32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @gather_lowering(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %idx_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src = builtin.unrealized_conversion_cast %src_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    %idx_mem = builtin.unrealized_conversion_cast %idx_memref : memref<64xi32, #hivm.address_space<ub>> to !i32vec
    %dst = builtin.unrealized_conversion_cast %dst_memref : memref<64xf32, #hivm.address_space<ub>> to !vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %idx_tile = "tla.tile_view"(%idx_mem, %shape, %coord) : (!i32vec, !tla.shape<64>, !tla.coord<0>) -> !i32vec
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %mask = "tla.create_mask"() {pattern = "M4", dtype = f32} : () -> !tla.mask
        %indices = tla.load %idx_tile : !i32vec -> !i32vec
        %gathered = tla.gather %src_tile, %indices mask %mask : !vec, !i32vec mask !tla.mask -> !vec
        tla.store %dst_tile, %gathered : !vec, !vec
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// GATHER-LABEL: gather_lowering
// GATHER: hivm{{.+}}
// GATHER-NOT: tla.gather
// GATHER: return
