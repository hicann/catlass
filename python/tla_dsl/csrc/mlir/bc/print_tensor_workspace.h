#pragma once

#include <cstdint>

namespace tla::print_tensor {

[aicore] __attribute__((always_inline)) inline void
InitializeWorkspace(uint64_t print_workspace) {
  auto *print_address = reinterpret_cast<__gm__ uint8_t *>(print_workspace);
  // C310 DumpTensor emits TLVs through the shared debug FIFO. This also
  // bypasses CANN 9.1.0-beta.3's no-op InitDump overload.
  g_sysPrintFifoSpace = print_address;
}

} // namespace tla::print_tensor
