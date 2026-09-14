#include <cstdint>

#include <ccelib/internal/debug_tunnel/payload_impl.h>
#include <ccelib/internal/debug_tunnel/tunnel_impl.h>

extern "C" {
[aicore] __attribute__((always_inline)) void _mlir_ciface_init_debug(std::uint64_t state)
{
    __DebugTunnel_Initialize(reinterpret_cast<__gm__ cce::internal::DebugTunnelData*>(state));
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_finish_debug(std::uint64_t state)
{
    __DebugTunnel_Finish(reinterpret_cast<__gm__ cce::internal::DebugTunnelData*>(state));
}
}
