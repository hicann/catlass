// RUN: %tla_compile %s -o - | %filecheck %s

module {
  func.func @block_dim_lit(%out: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %0 = tla.arch.block_dim -> i32
    memref.store %0, %out[%c0] : memref<1xi32>
    func.return
  }
}

// CHECK-LABEL: func.func @block_dim_lit
// CHECK: hivm.hir.get_block_num
// CHECK: llvm.trunc {{.*}} : i64 to i32
// CHECK-NOT: tla.arch.block_dim
