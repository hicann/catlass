// RUN: %tla_compile %s --mlir-print-ir-after=combine-ave-ops -o %t 2>&1 | %filecheck %s
// RUN: %tla_compile %s --mlir-print-ir-after=convert-hivmave-to-ave-intrin -o %t 2>&1 | %filecheck %s --check-prefix=INTRIN

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, row_major>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_exp_sub_fusion(
      %lhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %rhs_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %lhs = tla.tensor_desc %lhs_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %rhs = tla.tensor_desc %rhs_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    %dst = tla.tensor_desc %dst_memref[%c0, %c0, %c64, %c1, %c1, %c64, %c1, %c64] : (memref<64xf32, #hivm.address_space<ub>>, index, index, index, index, index, index, index, index) -> !fvec
    "tla.vec.func"() ({
      %lhs_vec = tla.load %lhs : !fvec -> !tla.vector<64xf32>
      %rhs_vec = tla.load %rhs : !fvec -> !tla.vector<64xf32>
      %diff = tla.sub %lhs_vec, %rhs_vec : !tla.vector<64xf32>, !tla.vector<64xf32> -> !tla.vector<64xf32>
      %result = tla.exp %diff : !tla.vector<64xf32> -> !tla.vector<64xf32>
      tla.store %dst, %result : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }
}

// CHECK:        IR Dump After CombineAVEOPs (combine-ave-ops)
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT:    ave.hir.vsub
// CHECK-NOT:    ave.hir.vexp {{.*}} :
// CHECK:        ave.hir.vexpdif

// INTRIN:      IR Dump After
// INTRIN-SAME: convert-hivmave-to-ave-intrin
// INTRIN:      "hivm_regbaseintrins.intr.hivm.vexpdif"
