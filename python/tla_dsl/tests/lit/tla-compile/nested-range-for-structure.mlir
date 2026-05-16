// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<8192,8192>, !tla.stride<8192,1>, !tla.shape<8192,8192>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, %arg1: !tla.tensor<!tla.layout<!tla.shape<8192,8192>, !tla.stride<8192,1>, !tla.shape<8192,8192>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, %arg2: !tla.tensor<!tla.layout<!tla.shape<8192,8192>, !tla.stride<8192,1>, !tla.shape<8192,8192>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, %out: memref<1xindex>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %grid_n = arith.constant 64 : index
    %total_blocks = arith.constant 4096 : index
    %k_l1_count = arith.constant 64 : index
    %k_l0_count = arith.constant 2 : index
    scf.for %block_linear = %c0 to %total_blocks step %c1 {
      %block_row = arith.divui %block_linear, %grid_n : index
      %block_col = arith.remui %block_linear, %grid_n : index
      scf.for %k_l1 = %c0 to %k_l1_count step %c1 {
        scf.for %k_l0 = %c0 to %k_l0_count step %c1 {
          %buf_idx = arith.remui %k_l0, %c2 : index
          %block_sum = arith.addi %block_row, %block_col : index
          %stored = arith.addi %block_sum, %buf_idx : index
          memref.store %stored, %out[%c0] : memref<1xindex>
        }
      }
    }
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<8192,8192>, !tla.stride<8192,1>, !tla.shape<8192,8192>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, !tla.tensor<!tla.layout<!tla.shape<8192,8192>, !tla.stride<8192,1>, !tla.shape<8192,8192>, row_major>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, !tla.tensor<!tla.layout<!tla.shape<8192,8192>, !tla.stride<8192,1>, !tla.shape<8192,8192>, row_major>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, memref<1xindex>) -> (), sym_name = "nested_range_for_structure"} : () -> ()
}

// CHECK-LABEL: func.func @nested_range_for_structure
// CHECK: scf.for
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.remui
// CHECK: arith.addi
// CHECK: memref.store
// CHECK-NOT: tla.range
// CHECK-NOT: tla.for
// CHECK: return
