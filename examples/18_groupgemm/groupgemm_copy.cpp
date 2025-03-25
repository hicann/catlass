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
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "helper.hpp"
#include "golden.hpp"

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"
#include "AscendCT/gemm/kernel/kernel_groupgemm.hpp"
#include "AscendCT/gemm/matmul_type.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/matrix_coord.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/epilogue/tile/tile_copy.hpp"
#include "AscendCT/epilogue/tile/tile_elemwise_add.hpp"
#include "AscendCT/epilogue/tile/tile_elemwise_muls.hpp"
#include "AscendCT/epilogue/tile/tile_cast.hpp"
#include "AscendCT/epilogue/block/block_epilogue.hpp"

using namespace AscendCT;
using ScalarType = float;

template <
    typename LayoutA,
    typename LayoutB,
    typename LayoutC
>
ASCENDCT_GLOBAL
void GroupGemm(
    uint32_t problemCount,
    uint64_t fftsAddr,
    GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR ptrProblemShape,
    GM_ADDR gmA, GM_ADDR ptrLayoutA,
    GM_ADDR gmB, GM_ADDR ptrLayoutB,
    GM_ADDR gmC, GM_ADDR ptrLayoutC,
    GM_ADDR gmWA, GM_ADDR ptrlayoutWA,
    GM_ADDR gmWB, GM_ADDR ptrlayoutWB,
    GM_ADDR gmWorkspace
){
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using GemmBlockDispatchPolicy = gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using AType = gemm::MatmulType<half, LayoutA>;
    using BType = gemm::MatmulType<half, LayoutB>;
    using CType = gemm::MatmulType<half, LayoutC>;
    using XType = gemm::MatmulType<half, LayoutC>;
    
    using L1TileShape = MatmulShape<128, 256, 256>;
    using L0TileShape = MatmulShape<128, 256, 64>;
    using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;

    using GemmBlock = gemm::block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, XType>; 
    
    using DType = CType;
    using ComputeType = XType;
    constexpr uint32_t computeLength = L1TileShape::MN / 2;

    using TileElemWiseAddGemm = epilogue::tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemm = epilogue::tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;
    using TileElemWistCastC = epilogue::tile::TileCast<ArchTag, ComputeType, CType, TileShapeCast>;
    using TileElemWistCastD = epilogue::tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;

    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, CType, XType, DType>;

    using EpilogueBlock = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulsGemm, TileElemWistCastC, TileElemWistCastD, EpilogueTileCopy>;

    using GroupGemmKernel = gemm::kernel::KernelGroupGemm<GemmBlock, EpilogueBlock>;
    typename GroupGemmKernel::Params params{problemCount,ptrProblemShape, alpha, beta, gmA, ptrLayoutA, gmB, ptrLayoutB, gmWorkspace, ptrLayoutC,gmWA, ptrlayoutWA, gmWB, ptrlayoutWB, gmC, gmC}; // 这里得修改 gmX保存A * B

    GroupGemmKernel groupgemm;
    groupgemm(params);
}

typedef struct Options {
    const std::string HELPER = "04_groupgemm groupCnt mlist nlist klist [deviceId]";
    uint32_t groupCnt = 8;
    std::vector<uint32_t> mList;
    std::vector<uint32_t> nList;
    std::vector<uint32_t> kList;
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv) {
        enum ArgsIndex {
            GROUPCNT_INDEX = 1,
            MLIST_INDEX,
            NLIST_INDEX,
            KLIST_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= KLIST_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        groupCnt = std::atoi(argv[GROUPCNT_INDEX]);
        parseList(argv[MLIST_INDEX], mList);
        parseList(argv[NLIST_INDEX], nList);
        parseList(argv[KLIST_INDEX], kList);

        if (mList.size() != groupCnt || nList.size() != groupCnt || kList.size() != groupCnt) {
            std::cerr << "List lengths do not match groupCnt." << std::endl;
            return -1;
        }

        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }

        return 0;
    }

private:
    void parseList(const char* str, std::vector<uint32_t>& list) {
        char* copy = strdup(str);
        char* token = std::strtok(copy, ",");
        while (token != nullptr) {
            list.push_back(std::atoi(token));
            token = std::strtok(nullptr, ",");
        }
        free(copy);
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

void Run(Options& options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t groupCnt = options.groupCnt;

    uint32_t* M_array= new uint32_t[groupCnt];
    uint32_t* N_array= new uint32_t[groupCnt];
    uint32_t* K_array= new uint32_t[groupCnt];
    for (uint32_t i = 0; i < groupCnt; ++i) {
        M_array[i] = options.mList[i];
        N_array[i] = options.nList[i];
        K_array[i] = options.kList[i];
    }
    
    const uint32_t align = 256; 
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    // crate grouped matmul problem shapes and layouts
    std::vector<MatmulCoord> problemShapeList(groupCnt);
    std::vector<LayoutA> layoutAList(groupCnt);
    std::vector<LayoutB> layoutBList(groupCnt);
    std::vector<LayoutC> layoutCList(groupCnt);
    std::vector<LayoutA> layoutWAList(groupCnt);
    std::vector<LayoutB> layoutWBList(groupCnt);

    uint64_t allMKCnt = 0;
    uint64_t allKNCnt = 0;
    uint64_t allMNCnt = 0;
    uint64_t allMKCnt_padding = 0;
    uint64_t allKNCnt_padding = 0;
    for (uint32_t i = 0; i < groupCnt; ++i) {
        problemShapeList[i] = MatmulCoord{M_array[i], N_array[i], K_array[i]};
        layoutAList[i] = LayoutA{M_array[i], K_array[i]};
        layoutBList[i] = LayoutB{K_array[i], N_array[i]};
        layoutCList[i] = LayoutC{M_array[i], N_array[i]};
        layoutWAList[i] = GetWorkspaceLayout(layoutAList[i], align);
        layoutWBList[i] = GetWorkspaceLayout(layoutBList[i], align);
        allMKCnt += M_array[i] * K_array[i];
        allKNCnt += K_array[i] * N_array[i];
        allMNCnt += M_array[i] * N_array[i];
        allMKCnt_padding += GetWorkspaceLen(layoutWAList[i]);
        allKNCnt_padding += GetWorkspaceLen(layoutWBList[i]);
    }
    size_t scalarSize = groupCnt * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(groupCnt);
    std::vector<ScalarType> hostBeta(groupCnt);
    golden::FillRandomData(hostAlpha,  -1.0f, 1.0f);
    golden::FillRandomData(hostBeta,  -1.0f, 1.0f);

    size_t sizeA = allMKCnt * sizeof(half);
    size_t sizeB = allKNCnt * sizeof(half);
    size_t sizeC = allMNCnt * sizeof(half);
    std::vector<half> hostA(allMKCnt);
    std::vector<half> hostB(allKNCnt);
    std::vector<half> hostC(allMNCnt);
    golden::FillRandomData(hostA,  -1.0f, 1.0f);
    golden::FillRandomData(hostB,  -1.0f, 1.0f);
    golden::FillRandomData(hostC,  -1.0f, 1.0f);

    ScalarType* deviceAlpha{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceAlpha), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAlpha, scalarSize, hostAlpha.data(), scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ScalarType* deviceBeta{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBeta), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBeta, scalarSize, hostBeta.data(), scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    half *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    size_t sizeWA = allMKCnt_padding * sizeof(half);
    half *deviceWA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));

    half *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    size_t sizeWB = allKNCnt_padding * sizeof(half);
    half *deviceWB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));


    half* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
    size_t sizeX = allMNCnt * sizeof(half);
    half *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *problemShapeListDevice{nullptr};
    size_t sizeProblemShapeList = problemShapeList.size() * sizeof(MatmulCoord);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&problemShapeListDevice), sizeProblemShapeList,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(problemShapeListDevice, sizeProblemShapeList,
        problemShapeList.data(), sizeProblemShapeList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutAListDevice{nullptr};
    size_t sizeLayoutAList = layoutAList.size() * sizeof(LayoutA);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutAListDevice), sizeLayoutAList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutAListDevice, sizeLayoutAList,
        layoutAList.data(), sizeLayoutAList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutBListDevice{nullptr};
    size_t sizeLayoutBList = layoutBList.size() * sizeof(LayoutB);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutBListDevice), sizeLayoutBList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutBListDevice, sizeLayoutBList,
        layoutBList.data(), sizeLayoutBList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutCListDevice{nullptr};
    size_t sizeLayoutCList = layoutCList.size() * sizeof(LayoutC);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutCListDevice), sizeLayoutCList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutCListDevice, sizeLayoutCList,
        layoutCList.data(), sizeLayoutCList, ACL_MEMCPY_HOST_TO_DEVICE));
    
    uint8_t *layoutWAListDevice{nullptr};
    size_t sizeLayoutWAList = layoutWAList.size() * sizeof(LayoutA);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutWAListDevice), sizeLayoutWAList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutWAListDevice, sizeLayoutWAList,
        layoutWAList.data(), sizeLayoutWAList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutWBListDevice{nullptr};
    size_t sizeLayoutWBList = layoutWBList.size() * sizeof(LayoutB);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutWBListDevice), sizeLayoutWBList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutWBListDevice, sizeLayoutWBList,
        layoutWBList.data(), sizeLayoutWBList, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    GroupGemm<LayoutA, LayoutB, LayoutC><<<aicCoreNum, nullptr, stream>>>(
        groupCnt,
        fftsAddr,
        (uint8_t*)deviceAlpha, (uint8_t*)deviceBeta,
        problemShapeListDevice,
        (uint8_t*)deviceA, layoutAListDevice,
        (uint8_t*)deviceB, layoutBListDevice,
        (uint8_t*)deviceC, layoutCListDevice,
        (uint8_t*)deviceWA, layoutWAListDevice,
        (uint8_t*)deviceWB, layoutWBListDevice,
        (uint8_t*)gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<half> hostRes(allMNCnt);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> hostGolden(allMNCnt);
    golden::ComputeGroupGemm(groupCnt, problemShapeList,hostAlpha,hostBeta, hostA, layoutAList,
        hostB, layoutBList,hostC,layoutCList, hostGolden, layoutCList);
        
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, allMNCnt);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceWA));
    ACL_CHECK(aclrtFree(deviceWB));
    ACL_CHECK(aclrtFree(deviceAlpha));
    ACL_CHECK(aclrtFree(deviceBeta));
    ACL_CHECK(aclrtFree(problemShapeListDevice));
    ACL_CHECK(aclrtFree(layoutAListDevice));
    ACL_CHECK(aclrtFree(layoutBListDevice));
    ACL_CHECK(aclrtFree(layoutCListDevice));
    ACL_CHECK(aclrtFree(layoutWAListDevice));
    ACL_CHECK(aclrtFree(layoutWBListDevice));
    ACL_CHECK(aclrtFree(gmWorkspace));
    delete[] M_array;
    delete[] N_array;
    delete[] K_array;

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