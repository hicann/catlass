// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s < %t

module {
  func.func @range_for_structure(%out: memref<8xindex>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %i, %out[%i] : memref<8xindex>
    }
    func.return
  }
}

// CHECK-LABEL: func.func @range_for_structure(
// CHECK: cf.br
// CHECK: memref.store
// CHECK-NOT: tla.range
// CHECK-NOT: tla.for
// CHECK: return
