// RUN: %tla_compile %s -o - | %filecheck %s

// The same 250 KB allocation that ub-over-limit-refused.mlir rejects, accepted
// here because the module carries tla.ub_reserve_released.
//
// The host stamps that attribute when the caller passed both
// --cce-disable-asc-reserved-ubuf and --cce-disable-vf-stack-reserved-ubuf,
// which hand the compiler's 8 KB reserve back. Bisheng options never reach a
// pass, so without the attribute the pass holds every kernel to 248 KB and
// refuses one that legitimately fills the buffer -- which is exactly what broke
// the mmad-epilogue battery, since those kernels size their UB from the full
// capacity and pass both switches.

module attributes {tla.ub_reserve_released} {
  tla.func @fills_the_whole_buffer() {
    %a = tla.alloc_ptr {size_bytes = 256000} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }
}

// The ceiling widens to the whole buffer, so the kernel is accepted and the
// limit it was measured against is reported as 262144 rather than 253952.
// CHECK: tla.ub_programmable_bytes = 262144
