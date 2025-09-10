#ifndef ADJUST_TILING_H
#define ADJUST_TILING_H

#include "adjust_tiling_b16.h"

template <class DType>
void AdjustTiling(TilingParams &tilingParams)
{
    uint32_t layoutTagA = tilingParams.layoutTagA;
    uint32_t layoutTagB = tilingParams.layoutTagB;

    AdjustTilingB16[layoutTagA][layoutTagB](tilingParams);
}

#endif  // ADJUST_TILING_H