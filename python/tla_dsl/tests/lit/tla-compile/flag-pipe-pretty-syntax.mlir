// RUN: %tla_compile %s -o %t

module {
  tla.func @flag_pipe_pretty() {
    %l1_loaded = tla.flag "l1_loaded" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
    %l0_loaded = tla.flag "l0_loaded" {src_pipe = #tla.pipe<mte1>, dst_pipe = #tla.pipe<cube>} -> !tla.flag
    tla.set_flag %l1_loaded : !tla.flag
    tla.wait_flag %l1_loaded : !tla.flag
    tla.set_flag %l0_loaded : !tla.flag
    tla.wait_flag %l0_loaded : !tla.flag
    tla.return
  }
}
