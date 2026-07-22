// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --check-prefix=GATHER < %t

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!i32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @gather_lowering(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %idx_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%src_c0, %src_c0, %src_c64, %src_c1, %src_c1, %src_c64, %src_c1, %src_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    %idx_mem_c0 = arith.constant 0 : index
    %idx_mem_c1 = arith.constant 1 : index
    %idx_mem_c64 = arith.constant 64 : index
    %idx_mem = tla.tensor_desc %idx_memref[%idx_mem_c0, %idx_mem_c0, %idx_mem_c64, %idx_mem_c1, %idx_mem_c1, %idx_mem_c64, %idx_mem_c1, %idx_mem_c64] : (memref<64xi32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !i32vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref[%dst_c0, %dst_c0, %dst_c64, %dst_c1, %dst_c1, %dst_c64, %dst_c1, %dst_c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %idx_tile = "tla.tile_view"(%idx_mem, %shape, %coord) : (!i32vec, !tla.shape<64>, !tla.coord<0>) -> !i32vec
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
        %mask = "tla.create_mask"() {pattern = "M4", dtype = f32} : () -> !tla.mask
        %indices = tla.load %idx_tile : !i32vec -> !tla.vector<64xi32>
        %gathered = tla.gather %src_tile, %indices mask %mask : !vec, !tla.vector<64xi32> mask !tla.mask -> !tla.vector<64xf32>
        tla.store %dst_tile, %gathered : !vec, !tla.vector<64xf32>
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// GATHER-LABEL: gather_lowering
// GATHER: hivm{{.+}}
// GATHER-NOT: tla.gather
// GATHER: return
