#include "kernel/conv_bias.hpp"

#include <acl/acl.h>

#include "catlass_kernel.h"
#include "common.hpp"

namespace CatlassKernel {
using namespace Catlass;

void ConvBias(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    using LayoutFmap = layout::NDC1HWC0;
    using LayoutFilter = layout::KDC1KHKWN1N0C0;
    using LayoutOut = layout::NDC1HWC0;
    using InDType = kernelInfo.inputDataType;
    using BiasDType = kernelInfo.biasDataType;
    using OutDType = kernelInfo.outputDataType;
    Conv3dParams<InDType> problemShape(kernelInfo.fmapOriShape, kernelInfo.filterOriShape, kernelInfo.strideList, kernelInfo.padList, kernelInfo.dilationList);
    basic_matmul<LayoutFmap, LayoutFilter, LayoutOut, InDType, BiasDType, OutDType><<<blockNum, nullptr, stream>>>(
        problemShape, kernelInfo.inputAddr.at(0), kernelInfo.inputAddr.at(1), kernelInfo.inputAddr.at(2), kernelInfo.outputAddr.at(0));
}
} // namespace CatlassKernel