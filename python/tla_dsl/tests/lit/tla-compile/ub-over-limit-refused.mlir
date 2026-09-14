// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s

// UB is 256 KB, of which 8 KB is reserved for the compiler and Ascend C. A SIMD
// kernel may use the remaining 248 KB; this one asks for 250 KB. Nothing in the
// hardware refuses it -- the write lands in the compiler's register-spill space
// and corrupts it silently -- so the limit is enforced here instead.

module {
  tla.func @over_the_simd_limit() {
    %a = tla.alloc_ptr {size_bytes = 256000} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK: error: this kernel allocates 256000 bytes of UB, past the 253952-byte limit for a SIMD kernel
// CHECK-SAME: less the 8192-byte compiler reserve
