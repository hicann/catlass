// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-lower-scalar-access 2>&1 | %filecheck %s

module {
  tla.func @read_scalar(%arg0: !tla.tensor<!tla.layout<!tla.shape<1,8>, !tla.stride<8,1>, !tla.shape<1,8>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>>) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = tla.scalar_load %arg0[%c0, %c3] : !tla.tensor<!tla.layout<!tla.shape<1,8>, !tla.stride<8,1>, !tla.shape<1,8>, row_major>, !tla.coord<0,0>, !tla.ptr<i32, gm, 4>> -> i32
    tla.return
  }
}

// ScalarSSA lowers fully in tla-lower-scalar-access (after tla.func→memref bridge).
// Unused memref.load may be folded later by vector-region greedy patterns.
// CHECK-LABEL: func.func @read_scalar
// CHECK: memref.load
// CHECK-NOT: tla.scalar_load
