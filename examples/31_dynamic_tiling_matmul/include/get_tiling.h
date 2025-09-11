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
    uint32_t m1InL0 = tilingParams.m1 * 16, n1InL0 = tilingParams.n1 * 16, k1LnL0 = 0;
    if (m1InL0 && n1InL0) {
        // TODO
    }
    std::cout << std::dec
    << "┌────────────────────────────────────────────────────────────┐\n"
    << "│                    Tiling Parameters                       │\n"
    << "├──────────────────────────────┬─────────────────────────────┤\n"
    << "│ Basic Dimensions             │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ m                            │ " << std::setw(27) << tilingParams.m << " │\n"
    << "│ n                            │ " << std::setw(27) << tilingParams.n << " │\n"
    << "│ k                            │ " << std::setw(27) << tilingParams.k << " │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ Layout Tags                  │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ layoutTagA                   │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.layoutTagA) << " │\n"
    << "│ layoutTagB                   │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.layoutTagB) << " │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ Tile Sizes (x16)             │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ m1                           │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.m1) * 16 << " │\n"
    << "│ n1                           │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.n1) * 16 << " │\n"
    << "│ k1                           │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.k1) * 16 << " │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ L0 Cache Sizes               │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ m1InL0                       │ " << std::setw(27) << static_cast<uint32_t>(m1InL0) << " │\n"
    << "│ n1InL0                       │ " << std::setw(27) << static_cast<uint32_t>(n1InL0) << " │\n"
    << "│ k1InL0                       │ " << std::setw(27) << static_cast<uint32_t>(k1LnL0) << " │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ Padding Tags                 │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ paddingTagA                  │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.paddingTagA) << " │\n"
    << "│ paddingTagB                  │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.paddingTagB) << " │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ AIV Parameters               │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ aivm1                        │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.aivm1) << " │\n"
    << "│ aivn1                        │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.aivn1) << " │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ Block Dimension              │ Values                      │\n"
    << "├──────────────────────────────┼─────────────────────────────┤\n"
    << "│ blockDim                     │ " << std::setw(27) << static_cast<uint32_t>(tilingParams.blockDim) << " │\n"
    << "└──────────────────────────────┴─────────────────────────────┘" << std::endl;
}

#endif  // TILING_H