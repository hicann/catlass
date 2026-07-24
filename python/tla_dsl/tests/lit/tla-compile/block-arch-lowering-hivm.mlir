// RUN: if %tla_compile %s --print-pipeline=mlir 2>&1 | grep -q tla-lower-block-idx; then %tla_compile %s -o - | %filecheck %s --check-prefix=HIVM; else echo "tla-lower-block-idx unavailable" | %filecheck %s --check-prefix=NOHIVM; fi

module {
  func.func @kernel_block_arch_ops(%out: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %idx = tla.arch.block_idx -> i32
    %dim = tla.arch.block_dim -> i32
    %sum = arith.addi %idx, %dim : i32
    memref.store %sum, %out[%c0] : memref<1xi32>
    return
  }
}

// HIVM-LABEL: func.func @kernel_block_arch_ops
// HIVM-NOT: tla.arch.block_idx
// HIVM-NOT: tla.arch.block_dim
// HIVM: hivm.hir.get_block_idx
// HIVM: llvm.trunc {{.*}} : i64 to i32
// HIVM: hivm.hir.get_block_num
// HIVM: llvm.trunc {{.*}} : i64 to i32
// HIVM: llvm.add
// HIVM: memref.store
// HIVM-NOT: tla.arch.block_idx
// HIVM-NOT: tla.arch.block_dim
// HIVM: return

// NOHIVM: tla-lower-block-idx unavailable
