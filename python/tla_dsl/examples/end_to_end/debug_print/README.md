# `tla.debug_print` examples

`tla.debug_print` supports exactly one signless `i32` or `f32` scalar in an
explicit `tla.cube` or `tla.vector` region. The FIFO decoder accepts only the
matching CANN scalar frames: `x=%d` and `v=%f`. Runtime scalar
expressions that lower through the canonical Numeric path, such as `x + y`,
are accepted when their result is signless `i32` or `f32`.

The backend reachability audit deliberately excludes historical archives and
finds no supported `tla_printf_ptr` producer or `ptr=%p` helper. Pointer
formatting is therefore not part of this backend's public or runtime contract.

For a C310 mixed kernel, the Cube callsite produces one `x` frame per logical
block while the Vector callsite preserves its native two-AIV-sub-block
execution, producing two `v` frames from distinct cores. Record order is not
defined.

Device expression coverage uses `debug_print.py --expression --rhs VALUE`; it
prints the runtime `lhs + rhs` result for both `i32` and `f32` on AIV and AIC.
