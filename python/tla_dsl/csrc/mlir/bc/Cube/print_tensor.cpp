#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

namespace {
constexpr uint32_t kPrintTensorDescriptor = 0x50524E54; // ASCII "PRNT".
// CANN's 1 MiB debug FIFO reserves 48 bytes for the shape TLV and 72 bytes
// for the tensor TLV. Its 32-byte payload alignment leaves 262112 f32 values:
// floor((1 MiB - 48 - 72) / 32) * (32 / sizeof(float)).
constexpr uint64_t kMaxFloat32Elements = 262112;

__aicore__ inline AscendC::ShapeInfo MakeShapeInfo(uint32_t shape0,
                                                  uint32_t shape1) {
  uint64_t rank = shape1 == 0U ? 1 : 2;
  uint32_t shape[2] = {static_cast<uint32_t>(shape0),
                       static_cast<uint32_t>(shape1)};
  return AscendC::ShapeInfo(static_cast<uint8_t>(rank), shape);
}
}

extern "C" {

__attribute__((used, section(".tla_print_tensor_abi"))) const char
    tla_print_tensor_abi_v3[] = "tla_print_tensor_abi_v3";

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
[aicore] __attribute__((always_inline))
void _mlir_ciface_tla_print_tensor_gm_f32(uint64_t print_workspace,
                                         uint64_t tensor_address,
                                         uint64_t count,
                                         uint64_t packed_shape) {
  int32_t signedShape0 = static_cast<int32_t>(packed_shape);
  int32_t signedShape1 = static_cast<int32_t>(packed_shape >> 32);
  if (signedShape0 <= 0 || signedShape1 < 0 || count == 0 ||
      count > kMaxFloat32Elements ||
      (tensor_address & (sizeof(float) - 1U)) != 0U ||
      count > static_cast<uint64_t>(signedShape0) *
                  (signedShape1 == 0 ? 1U : static_cast<uint32_t>(signedShape1)))
    return;
  uint32_t shape0 = static_cast<uint32_t>(signedShape0);
  uint32_t shape1 = static_cast<uint32_t>(signedShape1);
  AscendC::ShapeInfo shapeInfo = MakeShapeInfo(shape0, shape1);
  auto *print_address = reinterpret_cast<__gm__ uint8_t *>(print_workspace);
  // C310 AscendC::DumpTensor writes TLVs through the shared debug FIFO.
  g_sysPrintFifoSpace = print_address;

  AscendC::GlobalTensor<float> tensor;
  tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor_address), count);
  AscendC::DumpTensor(tensor[0], kPrintTensorDescriptor,
                      static_cast<uint32_t>(count), shapeInfo);

  pipe_barrier(PIPE_ALL);
  dsb(mem_dsb_t::DSB_ALL);
  dci();
}
#endif

} // extern "C"
