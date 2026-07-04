// RUN: if %tla_compile %s --print-pipeline=mlir 2>&1 | grep -q tla-lower-flag-barrier-to-hivm; then %tla_compile %s -o - | %filecheck %s --check-prefix=HIVM; else %tla_compile %s -o - | %filecheck %s --check-prefix=STD; fi

module {
  tla.func @pipe_barrier() {
    "tla.vector"() ({
      tla.pipe_barrier [#tla.pipe<mte2>]
      tla.pipe_barrier [#tla.pipe<cube>]
      tla.pipe_barrier [#tla.pipe<all>]
    }) : () -> ()
    tla.return
  }
}

// STD-DAG: func.func private @llvm.hivm.BARRIER(i64)
// STD-LABEL: func.func @pipe_barrier
// STD-DAG: arith.constant
// STD-DAG: arith.constant
// STD-DAG: arith.constant
// STD: call @llvm.hivm.BARRIER
// STD: return

// HIVM-LABEL: func.func @pipe_barrier
// HIVM-DAG: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// HIVM-DAG: hivm.hir.pipe_barrier[<PIPE_M>]
// HIVM-DAG: hivm.hir.pipe_barrier[<PIPE_ALL>]
// HIVM-NOT: llvm.hivm.BARRIER
// HIVM: return
