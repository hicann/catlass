// RUN: %tla_compile %s -o - | %filecheck %s

// A packed fp4 buffer has an i4 pointee: no whole-byte width, but still a
// fixed-width scalar, and tla.alloc_ptr accepts it for exactly that reason.
// dynamic_ub_base asked for the pointee's size in *bytes*, which is 0 for any
// sub-byte type, so every fp4 region was rejected -- and the message blamed the
// type for not being fixed-width, which sent the reader somewhere else entirely.

module {
  tla.func @dynamic_ub_base_takes_fp4() {
    %p = "tla.dynamic_ub_base"() : () -> !tla.ptr<!tla.f4e2m1, ub, 256>
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK: tla.ub_dynamic_base = 0
