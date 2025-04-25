#include "kernel/basic_matmul.hpp"

#include <acl/acl.h>

#include "act_kernel.h"

namespace ActKernel {
void BasicMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo) {
  Act::GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using LayoutC = layout::RowMajor;
  LayoutA layoutA{kernelInfo.m, kernelInfo.k};
  LayoutB layoutB{kernelInfo.k, kernelInfo.n};
  LayoutC layoutC{kernelInfo.m, kernelInfo.n};
  if (kernelInfo.inputDataType == ACL_FLOAT16 &&
      kernelInfo.outputDataType == ACL_FLOAT16) {
    basic_matmul<LayoutA, LayoutB, LayoutC, ACL_FLOAT16, ACL_FLOAT16>
        <<<blockNum, nullptr, stream>>>(problemShape,
                                        kernelInfo.inputAddr.at(0), layoutA,
                                        kernelInfo.inputAddr.at(1), layoutB,
                                        kernelInfo.outputAddr.at(0), layoutC);
  } else if (kernelInfo.inputDataType == ACL_BF16 &&
             kernelInfo.outputDataType == ACL_BF16) {
    basic_matmul<LayoutA, LayoutB, LayoutC, ACL_BF16, ACL_BF16>
        <<<blockNum, nullptr, stream>>>(problemShape,
                                        kernelInfo.inputAddr.at(0), layoutA,
                                        kernelInfo.inputAddr.at(1), layoutB,
                                        kernelInfo.outputAddr.at(0), layoutC);
  }
}
}  // namespace ActKernel