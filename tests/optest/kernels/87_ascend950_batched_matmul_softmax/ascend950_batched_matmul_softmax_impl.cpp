/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <stdexcept>
#include <string>

#include <tiling_api.h>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/batched_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/kernel/batched_matmul_softmax_tla.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A half
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B half
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C half
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementLogits = ElementC;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;
using LayoutLogits = layout::RowMajor;

using ArchTag = Arch::Ascend950;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using BaseL1 = tuple<C<256>, C<256>, C<128>>;
using BaseL0 = tuple<C<256>, C<256>, C<32>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0>::type;
using TileCopy = Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementLogits, LayoutLogits>;
using BaseL1N128 = tuple<C<256>, C<128>, C<128>>;
using BaseL0N128 = tuple<C<256>, C<128>, C<32>>;
using L1TileShapeN128 = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1N128>::type;
using L0TileShapeN128 = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0N128>::type;
using BaseL1M128 = tuple<C<128>, C<256>, C<128>>;
using BaseL0M128 = tuple<C<128>, C<256>, C<32>>;
using L1TileShapeM128 = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1M128>::type;
using L0TileShapeM128 = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0M128>::type;
using BaseL1Small = tuple<C<128>, C<128>, C<256>>;
using BaseL0Small = tuple<C<128>, C<128>, C<32>>;
using L1TileShapeSmall = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1Small>::type;
using L0TileShapeSmall = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0Small>::type;

template <class MatmulKernel_>
using BatchedMatmulSoftmaxKernel =
    Gemm::Kernel::BatchedMatmulSoftmaxTla<MatmulKernel_, LayoutTagA, LayoutTagB, LayoutLogits>;

namespace {

enum class BatchedMatmulSoftmaxMatmulVariant : uint32_t {
    SMALL_N = 0,
    SMALL_M = 1,
    N128 = 2,
    M128 = 3,
    BASE_N = 4,
    BASE_M = 5,
};

inline BatchedMatmulSoftmaxMatmulVariant SelectBatchedMatmulSoftmaxMatmulVariant(
    uint32_t batch, uint32_t m, uint32_t n, uint32_t blockNum)
{
    constexpr uint32_t matmulTileM = 256;
    constexpr uint32_t matmulSmallTileN = 128;
    constexpr uint32_t matmulBaseTileN = 256;
    constexpr uint32_t smallTileCoreRatioNum = 1;
    constexpr uint32_t transitionTileCoreRatioNum = 5;
    constexpr uint32_t tileCoreRatioDen = 2;

    uint32_t tileN = n <= matmulSmallTileN ? matmulSmallTileN : matmulBaseTileN;
    uint64_t outputTiles = static_cast<uint64_t>(batch) * ((m + matmulTileM - 1) / matmulTileM) *
                           ((static_cast<uint64_t>(n) + tileN - 1) / tileN);
    uint64_t blockCount = static_cast<uint64_t>(blockNum);
    if (outputTiles * tileCoreRatioDen < blockCount * smallTileCoreRatioNum) {
        return m > n ? BatchedMatmulSoftmaxMatmulVariant::SMALL_N : BatchedMatmulSoftmaxMatmulVariant::SMALL_M;
    }
    if (outputTiles * tileCoreRatioDen <= blockCount * transitionTileCoreRatioNum) {
        return m > n ? BatchedMatmulSoftmaxMatmulVariant::N128 : BatchedMatmulSoftmaxMatmulVariant::BASE_N;
    }
    if (n <= matmulSmallTileN) {
        return m > n ? BatchedMatmulSoftmaxMatmulVariant::N128 : BatchedMatmulSoftmaxMatmulVariant::M128;
    }
    return m > n ? BatchedMatmulSoftmaxMatmulVariant::BASE_N : BatchedMatmulSoftmaxMatmulVariant::BASE_M;
}

} // namespace

template <class L1TileShape_, class L0TileShape_, uint32_t Direction>
static void LaunchBatchedMatmulSoftmax(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    using BlockMmad_ = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape_, L0TileShape_, ElementA, ElementB, ElementLogits, void, TileCopy>;
    using MatmulKernel_ =
        Gemm::Kernel::BatchedMatmulTla<BlockMmad_, void, Gemm::Block::GemmIdentityBlockSwizzle<3, Direction>>;
    using Kernel = BatchedMatmulSoftmaxKernel<MatmulKernel_>;
    typename Kernel::Arguments arguments{
        params->batch,
        GemmCoord{params->m, params->n, params->k},
        params->inputAddr[0],
        params->inputAddr[1],
        params->outputAddr[0]};
    Catlass::RunKernel<Kernel>(arguments, stream, blockNum);
}

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    // Query the runtime directly so the JIT module does not depend on host tiling API state.
    int32_t deviceId = 0;
    aclError status = aclrtGetDevice(&deviceId);
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("BatchedMatmulSoftmax get device failed: " + std::to_string(status));
    }
    int64_t aicCoreNum = 0;
    status = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_CUBE_CORE_NUM, &aicCoreNum);
    if (status != ACL_SUCCESS || aicCoreNum <= 0) {
        throw std::runtime_error("BatchedMatmulSoftmax get Cube core count failed: " + std::to_string(status));
    }
    uint32_t selectedBlockNum = blockNum > aicCoreNum ? static_cast<uint32_t>(aicCoreNum) : blockNum;
    auto matmulVariant = SelectBatchedMatmulSoftmaxMatmulVariant(params->batch, params->m, params->n, selectedBlockNum);
    switch (matmulVariant) {
        case BatchedMatmulSoftmaxMatmulVariant::SMALL_N:
            LaunchBatchedMatmulSoftmax<L1TileShapeSmall, L0TileShapeSmall, 0>(selectedBlockNum, stream, params);
            break;
        case BatchedMatmulSoftmaxMatmulVariant::SMALL_M:
            LaunchBatchedMatmulSoftmax<L1TileShapeSmall, L0TileShapeSmall, 1>(selectedBlockNum, stream, params);
            break;
        case BatchedMatmulSoftmaxMatmulVariant::N128:
            LaunchBatchedMatmulSoftmax<L1TileShapeN128, L0TileShapeN128, 0>(selectedBlockNum, stream, params);
            break;
        case BatchedMatmulSoftmaxMatmulVariant::M128:
            LaunchBatchedMatmulSoftmax<L1TileShapeM128, L0TileShapeM128, 1>(selectedBlockNum, stream, params);
            break;
        case BatchedMatmulSoftmaxMatmulVariant::BASE_N:
            LaunchBatchedMatmulSoftmax<L1TileShape, L0TileShape, 0>(selectedBlockNum, stream, params);
            break;
        case BatchedMatmulSoftmaxMatmulVariant::BASE_M:
        default:
            LaunchBatchedMatmulSoftmax<L1TileShape, L0TileShape, 1>(selectedBlockNum, stream, params);
            break;
    }
}
