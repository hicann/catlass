// RUN: if %tla_compile --print-pipeline=mlir 2>&1 | grep -q add-kernel-prologue-epilogue; then %tla_compile %s -o - | FileCheck %s --check-prefix=HIVM; else echo "add-kernel-prologue-epilogue unavailable" | FileCheck %s --check-prefix=NOHIVM; fi

module {
  func.func @kernel() {
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// HIVM-LABEL: func.func @kernel
// HIVM: hivm.hir.set_ctrl false at ctrl[60]
// HIVM-NEXT: hivm.hir.set_ctrl true at ctrl[48]
// HIVM-COUNT-1: hivm.hir.pipe_barrier[<PIPE_ALL>]
// HIVM: return

// NOHIVM: add-kernel-prologue-epilogue unavailable
