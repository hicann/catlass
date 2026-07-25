#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

namespace {
constexpr uint32_t kPrintTensorDescriptor = 0x50524E54; // ASCII "PRNT".
}

extern "C" {

__attribute__((used, section(".tla_print_tensor_abi"))) const char
    tla_print_tensor_abi_v1[] = "tla_print_tensor_abi_v1";

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_print_tensor_gm_f32(uint64_t print_workspace,
                                         uint64_t tensor_address,
                                         uint64_t count) {
  auto *print_address = reinterpret_cast<__gm__ uint8_t *>(print_workspace);
  // C310 AscendC::DumpTensor writes TLVs through the shared debug FIFO.
  g_sysPrintFifoSpace = print_address;

  AscendC::GlobalTensor<float> tensor;
  tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_address), count);
  AscendC::DumpTensor(tensor[0], kPrintTensorDescriptor, count);

  pipe_barrier(PIPE_ALL);
  dsb(mem_dsb_t::DSB_ALL);
  dci();
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_print_tensor_ub_f32(uint64_t print_workspace,
                                         uint64_t tensor_address,
                                         uint64_t count) {
  g_sysPrintFifoSpace =
      reinterpret_cast<__gm__ uint8_t *>(print_workspace);

  AscendC::LocalTensor<float> tensor(
      AscendC::TPosition::VECCALC, static_cast<uint32_t>(tensor_address), count);
  AscendC::DumpTensor(tensor, kPrintTensorDescriptor, count);

  pipe_barrier(PIPE_ALL);
  dsb(mem_dsb_t::DSB_ALL);
  dci();
}
#endif

} // extern "C"
