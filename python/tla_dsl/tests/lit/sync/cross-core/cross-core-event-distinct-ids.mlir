// RUN: %tla_compile %s -o - | %filecheck %s

// Two flags sharing the same pipe pair must receive distinct, incrementing
// event ids (migrated from test_tla_lower_flag_barrier_to_hivm.py
// test_tla_sync_to_hivm_allocates_distinct_event_ids_for_same_pipe_pair).

module {
  tla.func @same_pair_distinct_ids() {
    "tla.vector"() ({
      %ready = tla.flag "ready" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      %done = tla.flag "done" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      tla.set_flag %ready : !tla.flag
      tla.wait_flag %ready : !tla.flag
      tla.set_flag %done : !tla.flag
      tla.wait_flag %done : !tla.flag
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @same_pair_distinct_ids
// CHECK-COUNT-1: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK-COUNT-1: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK-COUNT-1: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
// CHECK-COUNT-1: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
// CHECK-NOT: tla.flag
// CHECK-NOT: _mlir_ciface_tla_sync_block
