#ifndef TILING_H
#define TILING_H

#include <iostream>
#include <cstdint>
#include <string>

#include "base_info.h"

template <class Dtype>
void GetTiling(TilingParams &tilingParams, TilingKey &tilingKey)
{
    AdjustTiling<Dtype>(tilingParams);
    SelectKernel<Dtype>(tilingParams, tilingKey);
}

template <class Dtype>
void PrintTilingParams(const TilingParams &tilingParams)
{
    uint32_t bytePerC0 = 32;
    uint32_t c0NumPerFractal = 16;
    uint32_t elePerC0 = bytePerC0 / sizeof(Dtype);
    uing32_t m1InL0 = tilingParams.m1 * 16, n1InL0 = tilingParams.n1 * 16, k1LnL0 = 0;
    if (m1InL0 && n1InL0) {
        // TODO
    }
    std::cout << std::dec << "m: " << tilingParams.m << " ,n: " << tilingParams.n << " ,k: " << tilingParams.k
              << " ,layoutTagA: " << static_cast<uint32_t>(tilingParams.layoutTagA)
              << " ,layoutTagB: " << static_cast<uint32_t>(tilingParams.layoutTagB) << std::endl
              << "m1: " << static_cast<uint32_t>(tilingParams.m1) * 16
              << " ,n1: " << static_cast<uint32_t>(tilingParams.m1) * 16
              << " ,k1: " << static_cast<uint32_t>(tilingParams.m1) * 16 << std::endl
              << "m1InL0 " << static_cast<uint32_t>(m1InL0)
              << " ,n1InL0 " << static_cast<uint32_t>(n1InL0)
              << " ,k1InL0 " << static_cast<uint32_t>(k1LnL0) << std::endl
              << "paddingTagA: " << static_cast<uint32_t>(tilingParams.paddingTagB)
              << " ,paddingTagB: " << static_cast<uint32_t>(tilingParams.paddingTagB) << std::endl
              << "aivm1: " << static_cast<uint32_t>(tilingParams.aivm1)
              << " ,aivn1: " << static_cast<uint32_t>(tilingParams.aivn1) << std::endl
              << " ,blockDim: " << static_cast<uint32_t>(tilingParams.blockDim) << std::endl;
}

#endif  // TILING_H