/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <tiling/platform/platform_ascendc.h>
#include <tiling_api.h>

#include "acl/acl.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/batched_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "catlass/gemm/kernel/batched_matmul_softmax_tla.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementLogits = half;
using LayoutTagA = layout::RowMajor;
using LayoutTagB = layout::ColumnMajor;
using LayoutLogits = layout::RowMajor;

using DispatchPolicy = Gemm::MmadPingpong<Arch::Ascend950, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;
using TileCopy = Gemm::Tile::PackedTileCopyTla<
    Arch::Ascend950, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementLogits, LayoutLogits>;

using L1TileShapeN128 = Shape<Int<256>, Int<128>, Int<128>>;
using L0TileShapeN128 = Shape<Int<256>, Int<128>, Int<32>>;

using L1TileShapeM128 = tuple<C<128>, C<256>, C<128>>;
using L0TileShapeM128 = tuple<C<128>, C<256>, C<32>>;

using L1TileShapeSmall = Shape<Int<128>, Int<128>, Int<128>>;
using L0TileShapeSmall = Shape<Int<128>, Int<128>, Int<32>>;

template <class MatmulKernel_>
using BatchedMatmulSoftmaxKernel =
    Gemm::Kernel::BatchedMatmulSoftmaxTla<MatmulKernel_, LayoutTagA, LayoutTagB, LayoutLogits>;

namespace {

std::vector<float> ComputeRowSoftmaxGolden(
    const std::vector<ElementA>& hostA, const std::vector<ElementB>& hostB, uint32_t batch, uint32_t m, uint32_t k,
    uint32_t n, std::vector<ElementLogits>& hostLogits)
{
    std::vector<float> hostGolden(static_cast<size_t>(batch) * m * n);
    for (uint32_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
        for (uint32_t row = 0; row < m; ++row) {
            float rowMax = -std::numeric_limits<float>::infinity();
            for (uint32_t col = 0; col < n; ++col) {
                float acc = 0.0f;
                for (uint32_t kk = 0; kk < k; ++kk) {
                    acc += static_cast<float>(hostA[(batchIdx * m + row) * k + kk]) *
                           static_cast<float>(hostB[(batchIdx * n + col) * k + kk]);
                }
                size_t offset = (batchIdx * m + row) * n + col;
                hostLogits[offset] = static_cast<ElementLogits>(acc);
                rowMax =
                    rowMax > static_cast<float>(hostLogits[offset]) ? rowMax : static_cast<float>(hostLogits[offset]);
            }
            float rowSum = 0.0f;
            for (uint32_t col = 0; col < n; ++col) {
                size_t offset = (batchIdx * m + row) * n + col;
                float value = std::exp(static_cast<float>(hostLogits[offset]) - rowMax);
                hostGolden[offset] = value;
                rowSum += value;
            }
            for (uint32_t col = 0; col < n; ++col) {
                hostGolden[(batchIdx * m + row) * n + col] /= rowSum;
            }
        }
    }
    return hostGolden;
}

} // namespace

template <class L1TileShape_, class L0TileShape_, uint32_t Direction>
static int LaunchBatchedMatmulSoftmax(
    uint32_t batch, uint32_t m, uint32_t k, uint32_t n, uint32_t blockNum, bool verify)
{
    using BlockMmad_ = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape_, L0TileShape_, ElementA, ElementB, ElementLogits, void, TileCopy>;
    using MatmulKernel_ =
        Gemm::Kernel::BatchedMatmulTla<BlockMmad_, void, Gemm::Block::GemmIdentityBlockSwizzle<3, Direction>>;

    using Kernel = BatchedMatmulSoftmaxKernel<MatmulKernel_>;

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&stream));

    size_t lenA = static_cast<size_t>(batch) * m * k;
    size_t lenB = static_cast<size_t>(batch) * k * n;
    size_t lenS = static_cast<size_t>(batch) * m * n;
    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeS = lenS * sizeof(ElementC);

    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    golden::FillRandomData<ElementA>(hostA, -2.0f, 2.0f);
    golden::FillRandomData<ElementB>(hostB, -2.0f, 2.0f);

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceS{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceS), sizeS, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    if (blockNum == 0) {
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        blockNum = Gemm::Kernel::SelectBatchedMatmulSoftmaxBlockNum(batch, m, n, aicCoreNum);
    }
    using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
    typename Kernel::Arguments arguments{batch, GemmCoord{m, n, k}, deviceA, deviceB, deviceS};
    MatmulAdapter matmulOp;
    if (matmulOp.CanImplement(arguments) == Status::kInvalid) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceS));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::cerr << "Unsupported BatchedMatmulSoftmax arguments." << std::endl;
        return 1;
    }
    matmulOp.Initialize(arguments, nullptr);
    matmulOp(stream, blockNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    if (!verify) {
        std::cout << "Run success." << std::endl;
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceS));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        return 0;
    }

    std::vector<ElementC> hostOutput(lenS);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeS, deviceS, sizeS, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<ElementLogits> hostLogits(lenS);
    std::vector<float> hostGolden = ComputeRowSoftmaxGolden(hostA, hostB, batch, m, k, n, hostLogits);

    auto errorIndices = golden::CompareData(hostOutput, hostGolden, k);
    constexpr float minMatchedRatio = 0.99f;
    constexpr float maxAbsErrorLimit = 0.1f;
    float maxAbsError = 0.0f;
    for (uint64_t i = 0; i < hostOutput.size(); ++i) {
        float diff = std::abs(static_cast<float>(hostOutput[i]) - hostGolden[i]);
        maxAbsError = maxAbsError > diff ? maxAbsError : diff;
    }
    float matchedRatio = 1.0f - static_cast<float>(errorIndices.size()) / hostOutput.size();
    bool passed = matchedRatio >= minMatchedRatio && maxAbsError <= maxAbsErrorLimit;
    if (passed) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. matched_ratio=" << matchedRatio << ", max_abs_error=" << maxAbsError << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceS));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return passed ? 0 : 1;
}

static int Run(uint32_t batch, uint32_t m, uint32_t k, uint32_t n, uint32_t blockNum, bool verify)
{
    if (!Gemm::Kernel::BatchedMatmulSoftmaxTuning::IsSoftmaxNSupported(n)) {
        std::cerr << "BatchedMatmulSoftmax N exceeds the available UB workspace." << std::endl;
        return 1;
    }
    uint32_t selectedBlockNum = blockNum;
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    if (selectedBlockNum == 0) {
        selectedBlockNum = Gemm::Kernel::SelectBatchedMatmulSoftmaxBlockNum(batch, m, n, aicCoreNum);
    }
    if (selectedBlockNum > aicCoreNum) {
        selectedBlockNum = aicCoreNum;
    }

    constexpr uint32_t matmulTileM = Gemm::Kernel::BatchedMatmulSoftmaxTuning::MATMUL_TILE_M;
    constexpr uint32_t matmulSmallTileN = Gemm::Kernel::BatchedMatmulSoftmaxTuning::MATMUL_SMALL_TILE_N;
    constexpr uint32_t matmulBaseTileN = Gemm::Kernel::BatchedMatmulSoftmaxTuning::MATMUL_BASE_TILE_N;
    constexpr uint32_t smallTileCoreRatioNum = 1;
    constexpr uint32_t transitionTileCoreRatioNum = 5;
    constexpr uint32_t tileCoreRatioDen = 2;

    uint32_t tileN = n <= matmulSmallTileN ? matmulSmallTileN : matmulBaseTileN;
    uint64_t outputTiles = static_cast<uint64_t>(batch) * ((m + matmulTileM - 1) / matmulTileM) *
                           ((static_cast<uint64_t>(n) + tileN - 1) / tileN);
    uint64_t blockCount = static_cast<uint64_t>(selectedBlockNum);
    if (outputTiles * tileCoreRatioDen < blockCount * smallTileCoreRatioNum) {
        if (m > n) {
            return LaunchBatchedMatmulSoftmax<L1TileShapeSmall, L0TileShapeSmall, 0>(
                batch, m, k, n, selectedBlockNum, verify);
        }
        return LaunchBatchedMatmulSoftmax<L1TileShapeSmall, L0TileShapeSmall, 1>(batch, m, k, n, selectedBlockNum, verify);
    }
    if (outputTiles * tileCoreRatioDen <= blockCount * transitionTileCoreRatioNum) {
        if (m > n) {
            return LaunchBatchedMatmulSoftmax<L1TileShapeN128, L0TileShapeN128, 0>(batch, m, k, n, selectedBlockNum, verify);
        }
        return LaunchBatchedMatmulSoftmax<L1TileShape, L0TileShape, 0>(batch, m, k, n, selectedBlockNum, verify);
    }
    constexpr uint32_t smallTileN = Gemm::Kernel::BatchedMatmulSoftmaxTuning::MATMUL_SMALL_TILE_N;
    if (n <= smallTileN) {
        if (m > n) {
            return LaunchBatchedMatmulSoftmax<L1TileShapeN128, L0TileShapeN128, 0>(
                batch, m, k, n, selectedBlockNum, verify);
        }
        return LaunchBatchedMatmulSoftmax<L1TileShapeM128, L0TileShapeM128, 1>(batch, m, k, n, selectedBlockNum, verify);
    }
    if (m > n) {
        return LaunchBatchedMatmulSoftmax<L1TileShape, L0TileShape, 0>(batch, m, k, n, selectedBlockNum, verify);
    }
    return LaunchBatchedMatmulSoftmax<L1TileShape, L0TileShape, 1>(batch, m, k, n, selectedBlockNum, verify);
}

int main(int argc, const char** argv)
{
    if (argc != 1 && argc != 5 && argc != 6 && argc != 7) {
        std::cerr << "Usage: " << argv[0] << " [batch M K N [block_num [verify]]]" << std::endl;
        return 2;
    }
    uint32_t batch = argc >= 5 ? static_cast<uint32_t>(std::stoul(argv[1])) : 2;
    uint32_t m = argc >= 5 ? static_cast<uint32_t>(std::stoul(argv[2])) : 128;
    uint32_t k = argc >= 5 ? static_cast<uint32_t>(std::stoul(argv[3])) : 128;
    uint32_t n = argc >= 5 ? static_cast<uint32_t>(std::stoul(argv[4])) : 128;
    uint32_t blockNum = argc >= 6 ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;
    bool verify = argc < 7 || std::stoul(argv[6]) != 0;
    return Run(batch, m, k, n, blockNum, verify);
}
