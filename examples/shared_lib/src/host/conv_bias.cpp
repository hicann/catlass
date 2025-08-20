#include "kernel/conv_bias.hpp"

#include <acl/acl.h>

#include "catlass_kernel.h"
#include "common.hpp"

namespace CatlassKernel {
using namespace Catlass;

void ConvBias(uint32_t blockNum, aclrtStream stream, ConvKernelInfo kernelInfo)
{
    if (kernelInfo.inputDataType == ACL_FLOAT16) {
        using LayoutFmap = layout::NDC1HWC0;
        using LayoutFilter = layout::KDC1KHKWN1N0C0;
        using LayoutOut = layout::NDC1HWC0;
        using LayoutBias = layout::VectorLayout;
        using InDType = half;
        using BiasDType = half;
        using OutDType = half;
        Conv3dParams problemShape = Conv3dParams::MakeConvCoord(kernelInfo.fmapRelated.data(), kernelInfo.filterRelated.data(), kernelInfo.padList.data(), kernelInfo.strideList.data(), kernelInfo.dilationList.data());
        conv_bias<LayoutFmap, LayoutFilter, LayoutOut, InDType, BiasDType, OutDType><<<blockNum, nullptr, stream>>>(
            problemShape, kernelInfo.inputAddr.at(0), kernelInfo.inputAddr.at(1), kernelInfo.inputAddr.at(2), kernelInfo.outputAddr.at(0));
    } else if (kernelInfo.inputDataType == ACL_BF16) {
        using LayoutFmap = layout::NDC1HWC0;
        using LayoutFilter = layout::KDC1KHKWN1N0C0;
        using LayoutOut = layout::NDC1HWC0;
        using InDType = bfloat16_t;
        using BiasDType = float;
        using OutDType = bfloat16_t;
        Conv3dParams problemShape = Conv3dParams::MakeConvCoord(kernelInfo.fmapRelated.data(), kernelInfo.filterRelated.data(), kernelInfo.padList.data(), kernelInfo.strideList.data(), kernelInfo.dilationList.data());
        conv_bias<LayoutFmap, LayoutFilter, LayoutOut, InDType, BiasDType, OutDType><<<blockNum, nullptr, stream>>>(
            problemShape, kernelInfo.inputAddr.at(0), kernelInfo.inputAddr.at(1), kernelInfo.inputAddr.at(2), kernelInfo.outputAddr.at(0));
    }
}
} // namespace CatlassKernel