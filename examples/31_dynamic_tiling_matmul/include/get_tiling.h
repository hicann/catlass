#ifndef TILING_H
#define TILING_H

#include <iostream>
#include <cstdint>
#include <string>
#include <iomanip>

#include "tiling_params.h"
#include "launch_map.h"
#include "adjust_tiling.h"
#include "select_kernel.h"
#include "platform_info.h"

template <class DType>
void GetTiling(TilingParams &tilingParams, TilingKey &tilingKey, PlatformInfo& platformInfo)
{
    AdjustTiling<DType>(tilingParams, platformInfo);
    SelectKernel<DType>(tilingParams, tilingKey, platformInfo);
}

template <class DType>
void PrintTilingParams(const TilingParams &tilingParams)
{
    uint32_t bytePerC0 = 32;
    uint32_t c0NumPerFractal = 16;
    uint32_t elePerC0 = bytePerC0 / sizeof(DType);
    uint32_t m0 = tilingParams.m1, n0 = tilingParams.n1, k0 = 0;
    if (m0 && n0) {
        // TODO
    }
    std::cout << std::dec
    << "┌───────────────────────────────────┐\n"
    << "│         Tiling Parameters         │\n"
    << "├──────────────┬────────────────────┤\n"
    << "│ m:           " << std::setw(20) << tilingParams.m << " │\n"
    << "│ n:           " << std::setw(20) << tilingParams.n << " │\n"
    << "│ k:           " << std::setw(20) << tilingParams.k << " │\n"
    << "├──────────────┼────────────────────┤\n"
    << "│ layoutTagA:  " << std::setw(20) << static_cast<uint32_t>(tilingParams.layoutTagA) << " │\n"
    << "│ layoutTagB:  " << std::setw(20) << static_cast<uint32_t>(tilingParams.layoutTagB) << " │\n"
    << "├──────────────┼────────────────────┤\n"
    << "│ mTile:       " << std::setw(20) << static_cast<uint32_t>(tilingParams.m1) << " │\n"
    << "│ nTile:       " << std::setw(20) << static_cast<uint32_t>(tilingParams.n1) << " │\n"
    << "│ kTile:       " << std::setw(20) << static_cast<uint32_t>(tilingParams.k1) << " │\n"
    << "├──────────────┼────────────────────┤\n"
    << "│ mTileInL0:   " << std::setw(20) << static_cast<uint32_t>(m0) << " │\n"
    << "│ nTileInL0:   " << std::setw(20) << static_cast<uint32_t>(n0) << " │\n"
    << "│ kTileInL0:   " << std::setw(20) << static_cast<uint32_t>(k0) << " │\n"
    << "├──────────────┼────────────────────┤\n"
    << "│ paddingTagA: " << std::setw(20) << static_cast<uint32_t>(tilingParams.paddingTagA) << " │\n"
    << "│ paddingTagB: " << std::setw(20) << static_cast<uint32_t>(tilingParams.paddingTagB) << " │\n"
    << "├──────────────┼────────────────────┤\n"
    << "│ blockDim:    " << std::setw(20) << static_cast<uint32_t>(tilingParams.blockDim) << " │\n"
    << "└──────────────┴────────────────────┘" << std::endl;
}

#endif  // TILING_H