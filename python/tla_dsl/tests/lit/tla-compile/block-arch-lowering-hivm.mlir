// RUN: if %tla_compile %s --print-pipeline=mlir 2>&1 | grep -q tla-lower-to-hivm; then %tla_compile %s -o - | %filecheck %s --check-prefix=HIVM; else echo "tla-lower-to-hivm unavailable" | %filecheck %s --check-prefix=NOHIVM; fi

module {
  func.func @kernel_block_arch_ops(%out: memref<1xindex>) {
    %c0 = arith.constant 0 : index
    %idx = tla.arch.block_idx -> index
    %dim = tla.arch.block_dim -> index
    %sum = arith.addi %idx, %dim : index
    memref.store %sum, %out[%c0] : memref<1xindex>
    return
  }
}

// HIVM-LABEL: func.func @kernel_block_arch_ops
// HIVM-NOT: tla.arch.block_idx
// HIVM-NOT: tla.arch.block_dim
// HIVM: hivm.hir.get_block_idx
// HIVM: hivm.hir.get_block_num
// HIVM: llvm.add
// HIVM: builtin.unrealized_conversion_cast
// HIVM: memref.store
// HIVM-NOT: tla.arch.block_idx
// HIVM-NOT: tla.arch.block_dim
// HIVM: return

// NOHIVM: tla-lower-to-hivm unavailable
