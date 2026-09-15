// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

module {
  tla.func @auto_mutex_extern(%gm: !tla.ptr<f32, gm, 4>) attributes {tla.auto_sync = "v0"} {
    %ub = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<f32, ub, 256>
    %count = arith.constant 256 : i32
    "tla.vector"() ({
      tla.call_extern @tla_user_gm_to_ub_f32(%gm, %ub, %count) :
        (!tla.ptr<f32, gm, 4>, !tla.ptr<f32, ub, 256>, i32) -> ()
    }) : () -> ()
    tla.return
  }
}

// CHECK: error: auto_sync='v0' cannot be combined with tla.call_extern; external calls require explicit synchronization in v1
