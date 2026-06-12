// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @block_dim_lit(%out: memref<1xindex>) {
    %c0 = arith.constant 0 : index
    %0 = tla.arch.block_dim -> index
    memref.store %0, %out[%c0] : memref<1xindex>
    func.return
  }
}

// CHECK-LABEL: func.func @block_dim_lit
// CHECK: hivm.hir.get_block_num
// CHECK: builtin.unrealized_conversion_cast
// CHECK-NOT: tla.arch.block_dim
