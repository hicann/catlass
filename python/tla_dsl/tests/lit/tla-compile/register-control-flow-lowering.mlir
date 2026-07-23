// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region -o %t 2>&1 | %filecheck %s --implicit-check-not="!tla.vector" --implicit-check-not="!tla.mask<"

!vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @register_control_flow(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>,
      %cond: i1, %limit: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    %dst = tla.tensor_desc %dst_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!vec, !tla.shape<64>, !tla.coord<0>) -> !vec
      %value = tla.load %src_tile : !vec -> !tla.vector<64xf32>
      %mask = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      %if_value, %if_mask = scf.if %cond -> (!tla.vector<64xf32>, !tla.mask<64>) {
        %next_value = tla.abs %value : !tla.vector<64xf32> -> !tla.vector<64xf32>
        %next_mask = tla.bitwise_not %mask : !tla.mask<64> -> !tla.mask<64>
        scf.yield %next_value, %next_mask : !tla.vector<64xf32>, !tla.mask<64>
      } else {
        scf.yield %value, %mask : !tla.vector<64xf32>, !tla.mask<64>
      }
      %loop_value, %loop_mask = scf.for %i = %c0 to %limit step %c1
          iter_args(%carried_value = %if_value, %carried_mask = %if_mask)
          -> (!tla.vector<64xf32>, !tla.mask<64>) {
        %next_value, %next_mask = scf.if %cond -> (!tla.vector<64xf32>, !tla.mask<64>) {
          %updated_value = tla.abs %carried_value : !tla.vector<64xf32> -> !tla.vector<64xf32>
          %updated_mask = tla.bitwise_not %carried_mask : !tla.mask<64> -> !tla.mask<64>
          scf.yield %updated_value, %updated_mask : !tla.vector<64xf32>, !tla.mask<64>
        } else {
          scf.yield %carried_value, %carried_mask : !tla.vector<64xf32>, !tla.mask<64>
        }
        scf.yield %next_value, %next_mask : !tla.vector<64xf32>, !tla.mask<64>
      }
      tla.store %dst_tile, %loop_value mask %loop_mask : !vec, !tla.vector<64xf32> mask !tla.mask<64>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_{{[0-9]+}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: ave.hir.pge <H> {element_alignment_bit_width = 32 : i32} : vector<256xi1>
// CHECK: scf.if {{.*}} -> (vector<64xf32>, vector<256xi1>)
// CHECK: ave.hir.vabs
// CHECK: ave.hir.preg.not <b32>
// CHECK: scf.yield {{.*}} : vector<64xf32>, vector<256xi1>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<64xf32>, vector<256xi1>)
// CHECK: scf.if {{.*}} -> (vector<64xf32>, vector<256xi1>)
// CHECK: ave.hir.vabs
// CHECK: ave.hir.preg.not <b32>
// CHECK: scf.yield {{.*}} : vector<64xf32>, vector<256xi1>
// CHECK: ave.hir.masked_store
