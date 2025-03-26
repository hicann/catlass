/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "bfloat16.h"

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"
#include "AscendCT/gemm/kernel/kernel_quant_gemm.hpp"
#include "AscendCT/gemm/matmul_type.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/matrix_coord.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/epilogue/tile/tile_copy.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_mul.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_add.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "AscendCT/epilogue/tile/tile_cast.hpp"
#include "AscendCT/epilogue/block/block_epilogue.hpp"
#include "AscendCT/epilogue/tile/tile_swizzle.hpp"

using namespace AscendCT;
using bfloat16 = op::bfloat16;

template <
    class LayoutA,
    class LayoutB,
    class LayoutScale,
    class LayoutPerTokenScale,
    class LayoutBias,
    class LayoutC
>
ASCENDCT_GLOBAL
void QuantGemm(
    uint64_t fftsAddr,
    MatmulCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmWA, LayoutA layoutWA,
    GM_ADDR gmWB, LayoutB layoutWB,
    GM_ADDR gmScale, LayoutScale layoutScale,
    GM_ADDR gmPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
    GM_ADDR gmBias, LayoutBias layoutBias,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWorkspace
){
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using GemmBlockDispatchPolicy = AscendCT::gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using EpilogueBlockDispatchPolicy = AscendCT::epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using AType = gemm::MatmulType<int8_t, LayoutA>;
    using BType = gemm::MatmulType<int8_t, LayoutB>;
    using CType = gemm::MatmulType<bfloat16_t, LayoutC>;
    using XType = gemm::MatmulType<int32_t, LayoutC>;
    using ScaleType = gemm::MatmulType<bfloat16_t, LayoutScale>;
    using PerTokenScaleType = gemm::MatmulType<bfloat16_t, LayoutPerTokenScale>;
    using BiasType = gemm::MatmulType<bfloat16_t, LayoutBias>;
    using TempType = gemm::MatmulType<float, LayoutC>;
    using L1TileShape = MatmulShape<256, 128, 256>;
    using L0TileShape = MatmulShape<256, 128, 128>;
    using TileCastShape = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
    using GemmBlock = gemm::block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, XType>;
    using RowBroadcastMulType = gemm::MatmulType<float, LayoutC>;
    using RowBroadcastAddType = gemm::MatmulType<float, LayoutC>;
    using OneBlkColumnBroadcastAddType = gemm::MatmulType<float, LayoutC>;
    using BroadcastOneBlkType = gemm::MatmulType<float, LayoutC>;
    using OneBlkColumnBroadcastMulType = gemm::MatmulType<float, LayoutC>;
    using EpilogueTileShape = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
    using TileRowBroadcastMul = epilogue::tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = epilogue::tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = epilogue::tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileRowBroadcastAdd = epilogue::tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
    using TileOneBlkColumnBroadcastAdd = epilogue::tile::TileOneBlkColumnBroadcastAdd<ArchTag,
        OneBlkColumnBroadcastAddType, EpilogueTileShape>;
    using TileElemWiseCastTemp = epilogue::tile::TileCast<ArchTag, TempType, XType, TileCastShape>;
    using TileElemWiseCastC = epilogue::tile::TileCast<ArchTag, CType, TempType, TileCastShape>;
    using EpilogueTileCopy = epilogue::tile::TileCopyPerTokenDequantGemm<ArchTag, XType, ScaleType, PerTokenScaleType, BiasType, CType>;
    using TileScheduler = epilogue::tile::EpilogueHorizontalTileSwizzle;
    using EpilogueBlock = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, XType, ScaleType, PerTokenScaleType, BiasType, CType, 
        TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileRowBroadcastAdd, TileOneBlkColumnBroadcastAdd, 
        TileElemWiseCastTemp, TileElemWiseCastC, EpilogueTileCopy, TileScheduler>;
    typename EpilogueBlock::Params epilogueParams{gmScale, layoutScale, gmPerTokenScale, layoutPerTokenScale, gmBias, layoutBias, gmC, layoutC};
    using GemmKernel = gemm::kernel::KernelGemm<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmWorkspace, gmWA, layoutWA, gmWB, layoutWB, epilogueParams}; // 这里得修改 gmX保存A * B
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "21_quantgemm m n k [device_id]";

    MatmulCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv){
        enum ArgsIndex{
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if(argc > ARGS_MAX || argc <= K_INDEX){
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if(argc == ARGS_MAX){
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
}Options;

layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
{
    if (align == 0) {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(1), align));
}

layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0) {
        return layout;
    }
    return layout::ColumnMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(0), align));
}

size_t GetWorkspaceLen(layout::RowMajor layout)
{
    return layout.shape(0) * layout.stride(0);
}

size_t GetWorkspaceLen(layout::ColumnMajor layout)
{
    return layout.shape(1) * layout.stride(1);
}

bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
{
    return layout1.stride(0) == layout2.stride(0);
}

bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
{
    return layout1.stride(1) == layout2.stride(1);
}

void Run(Options options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenBias = static_cast<size_t>(n);
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenX = lenC;
    
    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeC = lenC * sizeof(bfloat16_t);
    size_t sizeX = lenX * sizeof(int32_t);
    size_t sizeScale = lenScale * sizeof(bfloat16_t);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(bfloat16_t);
    size_t sizeBias = lenBias * sizeof(bfloat16_t);

    const uint32_t align = 256;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using LayoutScale = layout::VectorLayout;
    using LayoutPerTokenScale = layout::VectorLayout;
    using LayoutBias = layout::VectorLayout;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    LayoutScale layoutScale{n};
    LayoutPerTokenScale layoutPerTokenScale{m};
    LayoutBias layoutBias{n};
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(int8_t);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(int8_t);

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<bfloat16> hostScale(lenScale);
    std::vector<bfloat16> hostPerTokenScale(lenPerTokenScale);
    std::vector<bfloat16> hostBias(lenBias);
    golden::FillRandomData(hostA,  -16, 16);
    golden::FillRandomData(hostB,  -16, 16);
    golden::FillRandomData(hostScale, 0.0, 1.0);
    golden::FillRandomData(hostPerTokenScale, 0.0, 1.0);
    golden::FillRandomData(hostBias, 0.0, 1.0);

    int8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    int8_t *deviceWA{nullptr};
    if (IsSameStride(layoutWA, layoutA)) {
        deviceWA = deviceA;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    int8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    int8_t *deviceWB{nullptr};
    if (IsSameStride(layoutWB, layoutB)) {
        deviceWB = deviceB;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    bfloat16* deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));

    bfloat16* devicePerTokenScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerTokenScale), sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale, ACL_MEMCPY_HOST_TO_DEVICE));

    bfloat16* deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    bfloat16 *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    
    int32_t *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    QuantGemm<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        options.problemShape,
        (uint8_t*)deviceA, layoutA,
        (uint8_t*)deviceB, layoutB,
        (uint8_t*)deviceWA, layoutWA,
        (uint8_t*)deviceWB, layoutWB,
        (uint8_t*)deviceScale, layoutScale,
        (uint8_t*)devicePerTokenScale, layoutPerTokenScale,
        (uint8_t*)deviceBias, layoutBias,
        (uint8_t*)deviceC, layoutC,
        (uint8_t*)gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    
    std::vector<bfloat16> hostRes(lenC);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    // for(uint32_t i = 0; i < lenC; i++){
    //     std::cout << i << " : " << hostRes[i] << std::endl; 
    // }
    std::vector<float> hostGolden(lenC);
    golden::QuantGemm(
        options.problemShape,
        hostA, layoutA,
        hostB, layoutB,
        hostScale, layoutScale,
        hostPerTokenScale, layoutPerTokenScale,
        hostBias, layoutBias,
        hostGolden, layoutC);
    // golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    if (!IsSameStride(layoutWA, layoutA)) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (!IsSameStride(layoutWB, layoutB)) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(gmWorkspace));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv){
    Options options;
    if(options.Parse(argc, argv) != 0){
        return -1;
    }
    Run(options);
    return 0;
}