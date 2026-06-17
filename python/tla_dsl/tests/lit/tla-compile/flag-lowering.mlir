// RUN: if %tla_compile %s --print-pipeline=mlir 2>&1 | grep -q tla-sync-to-hivm; then %tla_compile %s -o - | %filecheck %s --check-prefix=HIVM; else %tla_compile %s -o - | %filecheck %s --check-prefix=STD; fi

module {
  tla.func @flags() {
    %first = tla.flag "first" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
    %second = tla.flag "second" {src_pipe = #tla.pipe<mte1>, dst_pipe = #tla.pipe<mte2>} -> !tla.flag
    %cross = tla.cross_flag "cross" {src_pipe = #tla.pipe<mte3>, dst_pipe = #tla.pipe<scalar>} -> !tla.cross_flag
    tla.set_flag %first : !tla.flag
    tla.set_flag %second : !tla.flag
    tla.wait_flag %first : !tla.flag
    tla.wait_flag %second : !tla.flag
    tla.cross_core_set_flag %cross : !tla.cross_flag
    tla.cross_core_wait_flag %cross : !tla.cross_flag
    tla.return
  }
}

// STD-DAG: func.func private @llvm.hivm.SET.FLAG.IMM(i64, i64, i64)
// STD-DAG: func.func private @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64)
// STD-DAG: func.func private @_mlir_ciface_tla_cross_core_set_flag(i8, i8, i8)
// STD-DAG: func.func private @_mlir_ciface_tla_cross_core_wait_flag(i8)
// STD-LABEL: func.func @flags
// STD-DAG: arith.constant 4 : i64
// STD-DAG: arith.constant 3 : i64
// STD-DAG: arith.constant 0 : i64
// STD-DAG: call @llvm.hivm.SET.FLAG.IMM
// STD-DAG: call @llvm.hivm.WAIT.FLAG.IMM
// STD-DAG: call @_mlir_ciface_tla_cross_core_set_flag
// STD-DAG: call @_mlir_ciface_tla_cross_core_wait_flag
// STD-NOT: tla.flag
// STD-NOT: tla.cross_flag
// STD-NOT: tla.cross_core
// STD: return

// HIVM-LABEL: func.func @flags
// HIVM-DAG: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// HIVM-DAG: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// HIVM-DAG: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// HIVM-DAG: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// HIVM-DAG: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
// HIVM-DAG: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 0
// HIVM-NOT: llvm.hivm.SET.FLAG.IMM
// HIVM-NOT: llvm.hivm.WAIT.FLAG.IMM
// HIVM-NOT: _mlir_ciface_tla_sync_block
// HIVM-NOT: tla.flag
// HIVM-NOT: tla.cross_flag
// HIVM-NOT: tla.cross_core
// HIVM: return
