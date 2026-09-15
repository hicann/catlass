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

// -----

// RUN: if %tla_compile %s --print-pipeline=mlir 2>&1 | grep -q tla-lower-flag-barrier-to-hivm; then %tla_compile %s -o - | %filecheck %s --check-prefix=LOCAL_MEM_BAR_HIVM; else %tla_compile %s -o - | %filecheck %s --check-prefix=LOCAL_MEM_BAR_STD; fi

module {
  tla.func @local_mem_bar() {
    "tla.vector"() ({
      tla.local_mem_bar 0
      tla.local_mem_bar 1
      tla.local_mem_bar 2
      tla.local_mem_bar 3
      tla.local_mem_bar 4
      tla.local_mem_bar 5
      tla.local_mem_bar 6
      tla.local_mem_bar 7
      tla.local_mem_bar 8
      tla.local_mem_bar 9
      tla.local_mem_bar 10
      tla.local_mem_bar 11
    }) : () -> ()
    tla.return
  }
}

// LOCAL_MEM_BAR_STD-DAG: func.func private @llvm.hivm.BARRIER(i64)
// LOCAL_MEM_BAR_STD-LABEL: func.func @local_mem_bar
// LOCAL_MEM_BAR_STD: arith.constant
// LOCAL_MEM_BAR_STD: arith.constant
// LOCAL_MEM_BAR_STD: arith.constant
// LOCAL_MEM_BAR_STD: call @llvm.hivm.BARRIER
// LOCAL_MEM_BAR_STD: return

// LOCAL_MEM_BAR_HIVM-LABEL: func.func @local_mem_bar
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vv.all
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vst.vld
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vld.vst
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vst.vst
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vs.all
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vst.ld
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vld.st
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.vst.st
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.sv.all
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.st.vld
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.ld.vst
// LOCAL_MEM_BAR_HIVM-DAG: hivm_regbaseintrins.intr.hivm.mem.bar.st.vst
// LOCAL_MEM_BAR_HIVM-NOT: tla.local_mem_bar
// LOCAL_MEM_BAR_HIVM: return
