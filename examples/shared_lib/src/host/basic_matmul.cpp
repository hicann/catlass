#include "kernel/basic_matmul.hpp"

#include <acl/acl.h>

#include "catlass_kernel.h"
#include "common.hpp"

namespace ActKernel {
using namespace Catlass;
#define DEFINE_TEMPLATE_INSTANTIATION(BLOCK_NUM, STREAM, KERNEL_INFO, TRANS_A, \
                                      TRANS_B, TRANS_C, IN_DTYPE, OUT_DTYPE)   \
  if (kernelInfo.inputDataType == IN_DTYPE &&                                  \
      kernelInfo.outputDataType == OUT_DTYPE &&                                \
      kernelInfo.transA == TRANS_A && kernelInfo.transB == TRANS_B) {          \
    using LayoutA = typename Transpose2Layout<TRANS_A>::layout;                \
    using LayoutB = typename Transpose2Layout<TRANS_B>::layout;                \
    using LayoutC = layout::RowMajor;                                          \
    using InDType = typename AclType2Type<IN_DTYPE>::type;                     \
    using OutDType = typename AclType2Type<OUT_DTYPE>::type;                   \
    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};          \
    LayoutA layoutA{kernelInfo.m, kernelInfo.k};                               \
    LayoutB layoutB{kernelInfo.k, kernelInfo.n};                               \
    LayoutC layoutC{kernelInfo.m, kernelInfo.n};                               \
    basic_matmul<LayoutA, LayoutB, LayoutC, IN_DTYPE, OUT_DTYPE>               \
        <<<blockNum, nullptr, stream>>>(problemShape,                          \
                                        kernelInfo.inputAddr.at(0), layoutA,   \
                                        kernelInfo.inputAddr.at(1), layoutB,   \
                                        kernelInfo.outputAddr.at(0), layoutC); \
  }
void BasicMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo) {
  DEFINE_TEMPLATE_INSTANTIATION(blockNum, stream, kernelInfo, false, false,
                                false, ACL_FLOAT16, ACL_FLOAT16);
  DEFINE_TEMPLATE_INSTANTIATION(blockNum, stream, kernelInfo, false, false,
                                false, ACL_BF16, ACL_BF16);
}
}  // namespace ActKernel