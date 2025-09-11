#ifndef SELECT_KERNEL_H
#define SELECT_KERNEL_H

#include "platform_info.h"
#include "select_kernel_half.h"

template <class DType>
void SelectKernel(TilingParams &tilingParams, TilingKey &tilingKey, PlatformInfo& platformInfo)
{
    SelectKernelHalf(tilingParams, tilingKey, platformInfo);
}

#endif  // SELECT_KERNEL_H