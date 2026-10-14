/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCHED_MATMUL_SOFTMAX_TLA_TUNING_H
#define CATLASS_GEMM_KERNEL_BATCHED_MATMUL_SOFTMAX_TLA_TUNING_H

#include <cstdint>

#include "catlass/catlass.hpp"

namespace Catlass::Gemm::Kernel {

namespace BatchedMatmulSoftmaxTuning {

constexpr uint32_t MATMUL_TILE_M = 256;
constexpr uint32_t MATMUL_SMALL_TILE_N = 128;
constexpr uint32_t MATMUL_BASE_TILE_N = 256;
// These thresholds were selected from msprof measurements on Ascend 950.
// They are dispatch policy parameters, rather than functional constraints.
constexpr uint32_t TINY_WORKLOAD_TILES = 2;
constexpr uint32_t SMALL_WORKLOAD_TILES = 9;
constexpr uint32_t TRANSITION_WORKLOAD_TILES = 20;
constexpr uint32_t TINY_WORKLOAD_CORES = 16;
constexpr uint32_t TRANSITION_WORKLOAD_CORES = 24;
constexpr uint32_t SOFTMAX_UB_SIZE = 248 * 1024;
constexpr uint32_t SOFTMAX_TMP_RESERVE = 64 * 1024;
constexpr uint32_t SOFTMAX_BUFFER_COUNT = 2;
constexpr uint32_t SOFTMAX_ELEMENT_BYTES = sizeof(uint16_t);
constexpr uint32_t SOFTMAX_DATA_BLOCK_BYTES = BYTE_PER_BLK;
constexpr uint32_t SOFTMAX_MAX_ALIGNED_N = (1u << 15) - SOFTMAX_DATA_BLOCK_BYTES / SOFTMAX_ELEMENT_BYTES;
constexpr uint32_t SOFTMAX_N_THRESHOLD_128 = 128;
constexpr uint32_t SOFTMAX_N_THRESHOLD_512 = 512;
constexpr uint32_t SOFTMAX_N_THRESHOLD_1024 = 1024;
constexpr uint32_t SOFTMAX_MAX_ROWS_N128 = 64;
constexpr uint32_t SOFTMAX_MAX_ROWS_N512 = 32;
constexpr uint32_t SOFTMAX_MAX_ROWS_N1024 = 16;
constexpr uint32_t SOFTMAX_MAX_ROWS_LARGE = 8;

inline uint32_t AlignSoftmaxN(uint32_t n)
{
    constexpr uint32_t elementsPerBlock = SOFTMAX_DATA_BLOCK_BYTES / SOFTMAX_ELEMENT_BYTES;
    return ((n + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
}

inline uint32_t SelectSoftmaxRowsPerTile(uint32_t alignedN)
{
    uint32_t maxRows =
        alignedN <= SOFTMAX_N_THRESHOLD_128 ?
            SOFTMAX_MAX_ROWS_N128 :
            (alignedN <= SOFTMAX_N_THRESHOLD_512 ?
                 SOFTMAX_MAX_ROWS_N512 :
                 (alignedN <= SOFTMAX_N_THRESHOLD_1024 ? SOFTMAX_MAX_ROWS_N1024 : SOFTMAX_MAX_ROWS_LARGE));
    constexpr uint32_t budgetBytes = SOFTMAX_UB_SIZE - SOFTMAX_TMP_RESERVE;
    uint32_t bytesPerRow = alignedN * SOFTMAX_ELEMENT_BYTES * SOFTMAX_BUFFER_COUNT;
    uint32_t budgetRows = bytesPerRow == 0 ? 0 : budgetBytes / bytesPerRow;
    return budgetRows < maxRows ? budgetRows : maxRows;
}

inline uint32_t SelectSoftmaxBasicBlock(uint32_t n, uint32_t alignedN, uint32_t rowsPerTile)
{
    constexpr uint32_t minN = 128;
    constexpr uint32_t maxN = 2048;
    constexpr uint32_t rowsAlignment = 8;
    return n == alignedN && alignedN >= minN && alignedN <= maxN && rowsPerTile % rowsAlignment == 0;
}

constexpr bool IsSoftmaxNSupported(uint64_t n)
{
    constexpr uint64_t elementsPerBlock = SOFTMAX_DATA_BLOCK_BYTES / SOFTMAX_ELEMENT_BYTES;
    uint64_t alignedN = ((n + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    uint64_t buffers = alignedN * SOFTMAX_ELEMENT_BYTES * SOFTMAX_BUFFER_COUNT;
    return buffers <= SOFTMAX_UB_SIZE - SOFTMAX_TMP_RESERVE && alignedN <= SOFTMAX_MAX_ALIGNED_N;
}

} // namespace BatchedMatmulSoftmaxTuning

// Host-side block count selection, usable from launchers that do not compile the device kernel.
template <
    uint32_t MatmulTileM = BatchedMatmulSoftmaxTuning::MATMUL_TILE_M,
    uint32_t SmallTileN = BatchedMatmulSoftmaxTuning::MATMUL_SMALL_TILE_N,
    uint32_t BaseTileN = BatchedMatmulSoftmaxTuning::MATMUL_BASE_TILE_N>
inline uint32_t SelectBatchedMatmulSoftmaxBlockNum(uint32_t batch, uint32_t m, uint32_t n, uint32_t aicCoreNum)
{
    // Small workloads use all AIC cores; the transition range is capped to reduce dispatch overhead.
    uint32_t matmulTileN = n <= SmallTileN ? SmallTileN : BaseTileN;
    uint64_t matmulBlocks = static_cast<uint64_t>(batch) * ((m + MatmulTileM - 1) / MatmulTileM) *
                            ((static_cast<uint64_t>(n) + matmulTileN - 1) / matmulTileN);
    uint32_t targetBlocks = matmulBlocks <= BatchedMatmulSoftmaxTuning::TINY_WORKLOAD_TILES ?
                                BatchedMatmulSoftmaxTuning::TINY_WORKLOAD_CORES :
                                (matmulBlocks <= BatchedMatmulSoftmaxTuning::SMALL_WORKLOAD_TILES ?
                                     aicCoreNum :
                                     (matmulBlocks <= BatchedMatmulSoftmaxTuning::TRANSITION_WORKLOAD_TILES ?
                                          BatchedMatmulSoftmaxTuning::TRANSITION_WORKLOAD_CORES :
                                          aicCoreNum));
    return targetBlocks < aicCoreNum ? targetBlocks : aicCoreNum;
}

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BATCHED_MATMUL_SOFTMAX_TLA_TUNING_H
