//===- TlaSimtUbLimits.h - UB budget bounds ---------------------*- C++ -*-===//
//
// The one place these numbers are written down. The pass that checks a kernel
// against them and the host that reports what is left to a launch must agree
// exactly, so the host reads the resulting limit from here through the type
// bridge rather than recomputing it.
//
// Unified Buffer is 256 KB, partitioned low-to-high:
//
//     static | dynamic | reserved (8 KB) | Data Cache (32..120 KB)
//
// plus, for SIMT only, a 2 KB per-kernel runtime cost that the binary does not
// account for and that only shows up as a refused launch.
//
// The reserve belongs to the compiler and Ascend C -- 2 KB general plus 6 KB of
// VF stack for register spilling -- and the Data Cache exists only for SIMT,
// which reaches global memory through it. A kernel gets what is left below
// them, so SIMD loses the reserve and SIMT loses the reserve and at least the
// minimum cache. Ascend C computes the same 248 KB for this arch
// (sys_constants.h: ASC_UB_SIZE). A caller can take the reserve back with both
// cce-disable-*-reserved-ubuf switches; the host stamps the module when they
// are set, because bisheng options never reach a pass, and the limit below
// widens to the whole buffer.
//
// None of this is enforced by hardware: past these bounds a kernel silently
// writes into the reserve or the cache, which corrupts spilled registers or
// SIMT global-memory accesses rather than faulting. That is why the bound is
// refused at compile time -- a runtime check cannot catch what does not fail.
//
//===----------------------------------------------------------------------===//

#ifndef TLA_PASSES_TLASIMTUBLIMITS_H
#define TLA_PASSES_TLASIMTUBLIMITS_H

#include <cstdint>

namespace tla {

static constexpr uint64_t kUbTotalBytes = 256 * 1024;
// Present unless the caller explicitly releases it: it is only safe to drop
// for a kernel that never spills, which is the caller's judgement to make.
static constexpr uint64_t kUbReservedBytes = 8 * 1024;
static constexpr uint64_t kUbMinDataCacheBytes = 32 * 1024;
// A SIMT kernel carries a per-kernel runtime cost on top of the cache that the
// binary does not expose. Measured on Ascend950PR / CANN 9.1.0: a SIMT kernel
// with 1 KB of static UB is refused (ret=107000) at an extent of 220160 and
// accepted at 219136, i.e. 2 KB below the cache-only bound; the same kernel
// with 4 KB of static UB reaches its bound exactly. Reserving the 2 KB keeps
// the advertised ceiling reachable, which is the property callers rely on.
static constexpr uint64_t kSimtRuntimeOverheadBytes = 2 * 1024;

// The ceiling a kernel is actually held to. ``reserveReleased`` is true when the
// caller passed both cce-disable-*-reserved-ubuf switches, which hand the 8 KB
// reserve back; the host stamps that on the module, since bisheng options never
// reach a pass.
static constexpr uint64_t ubProgrammableBytes(bool simt, bool reserveReleased)
{
    const uint64_t reserve = reserveReleased ? 0 : kUbReservedBytes;
    const uint64_t simd = kUbTotalBytes - reserve;
    return simt ? simd - kUbMinDataCacheBytes - kSimtRuntimeOverheadBytes : simd;
}

} // namespace tla

#endif // TLA_PASSES_TLASIMTUBLIMITS_H
