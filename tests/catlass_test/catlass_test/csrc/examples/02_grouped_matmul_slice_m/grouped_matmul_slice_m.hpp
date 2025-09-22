/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass_test/common.hpp"

using namespace Catlass;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class ElementGroupList>
inline TEMPLATE_RET_TYPE GroupedMatmulSliceM(aclrtStream stream, GemmCoord problemShape, uint32_t problemCount, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceGroupList, uint8_t *deviceC) {
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using ArchTag = Arch::AtlasA2;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;
    using BlockEpilogue = void;
    if (problemShape.k() > problemShape.n()) {
        constexpr uint32_t l0AStages = 2;
        constexpr uint32_t l0BStages = 4;

        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
        using L1TileShape = GemmShape<256, 128, 256>;
        using L0TileShape = GemmShape<256, 128, 64>;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{problemShape, problemCount, deviceGroupList, deviceA, deviceB, deviceC};

        // call a kernel
        MatmulAdapter matmulOp;
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);

    } else {
        constexpr uint32_t l0AStages = 4;
        constexpr uint32_t l0BStages = 2;

        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
        using L1TileShape = GemmShape<128, 256, 256>;
        using L0TileShape = GemmShape<128, 256, 64>;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        typename MatmulKernel::Arguments arguments{problemShape, problemCount, deviceGroupList, deviceA, deviceB, deviceC};

        // call a kernel
        MatmulAdapter matmulOp;
        // judge arguments can run
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
    }
}