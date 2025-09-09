#ifndef SELECT_KERNEL_H
#define SELECT_KERNEL_H

#include "select_kernel_half.h"

template <class Dtype>
void SelectKernel(TilingParams &tilingParams, TilingKey &tilingKey)
{
    SelectKernelHalf(tilingParams, tilingKey);
}

#endif  // SELECT_KERNEL_H