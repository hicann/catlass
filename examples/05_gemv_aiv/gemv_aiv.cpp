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

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/matmul/dispatch_policy.hpp"
#include "acot/gemv/kernel/kernel_gemv_aiv.hpp"
#include "acot/gemv/block/block_gemv.hpp"
#include "acot/matmul/matmul_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemv/tile/tile_copy.hpp"
#include "acot/gemv/tile/tile_vmad.hpp"
#include "acot/gemv/tile/tile_vmuls.hpp"

using namespace acot;
using UBTileShape = GemvShape<32,512>;
using ScalarType = float;

ACOT_GLOBAL
void GemvAiv(
    GemvCoord problemShape,
    GM_ADDR gmA, layout::RowMajor layoutA,
    GM_ADDR gmX, layout::RowMajor layoutX,
    GM_ADDR gmY, layout::RowMajor layoutY,
    GM_ADDR gmY_read,
    ScalarType alpha,ScalarType beta,
    uint32_t SPLIT
){
    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = matmul::MmadAtlasA2Pingpong<true>;
    

    using AType = matmul::MatmulType<float, layout::RowMajor>;
    using XType = matmul::MatmulType<float, layout::RowMajor>;
    using YType = matmul::MatmulType<float, layout::RowMajor>;
    using BiasType = void;
    using TileCopy = gemv::tile::TileCopy<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    using TileVmad = gemv::tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
    using TileVmuls = gemv::tile::TileVmuls<typename DispatchPolicy::ArchTag, typename AType::Element>;

    using GemvBlock = gemv::block::BlockGemv<DispatchPolicy, UBTileShape, AType, XType, YType,BiasType,TileCopy,TileVmad,TileVmuls>;
    using BlockEpilogue = void;

    // kernel level
    using GemvKernel = gemv::kernel::KernelGemv<GemvBlock, BlockEpilogue>;
    typename GemvKernel::Params params{problemShape, gmA, layoutA, gmX, gmY,gmY_read,alpha,beta,SPLIT};
    // call a kernel
    GemvKernel gemv;
    
    gemv(params);
}

typedef struct Options{
    const std::string HELPER = "05_gemv_aiv m n [device_id]";

    uint32_t M = 32;
    uint32_t N = 32;
    uint32_t deviceId{0};

    Options() = default;
    
    GemvCoord problemShape{M, N};

    int Parse(int argc, const char **argv){
        enum ArgsIndex{
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if(argc > ARGS_MAX || argc <= N_INDEX){
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        if(argc == ARGS_MAX){
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
}Options;

uint32_t getSplictNum(bool trans, uint32_t M, uint32_t N, uint32_t M1, uint32_t N1, uint32_t maxSplict)
{
    uint32_t CORENUM = 20;
    uint32_t splitNum = 1;
    uint32_t maxOccupancy = 0; 
    uint32_t blockNum = (M - 1) / M1 + 1;
    if (!trans)
    {
        splitNum = 1;
    }
    else{
        uint32_t splitNum1 = 1, splitNum2 = 1;
        for (uint32_t i = 1; i <= maxSplict; i += 1)
        {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy)
            {
                maxOccupancy = occupancy;
                splitNum1 = i;
            }
        }
        maxOccupancy = 0;
        for (uint32_t i = 1; i <= maxSplict; i <<= 1)
        {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy)
            {
                maxOccupancy = occupancy;
                splitNum2 = i;
            }
        }
        splitNum = (splitNum1 - splitNum2) > 4 ? splitNum1 : splitNum2;
    }
    return splitNum;
}

void Run(Options options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    
    uint32_t maxSplict = 20;
    uint32_t const SPLIT = getSplictNum(false, m, n, UBTileShape::M, UBTileShape::N, maxSplict);

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenY = static_cast<size_t>(m) * 1;
    size_t scalarLen = 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    
    layout::RowMajor layoutA{m, n};
    layout::RowMajor layoutX{n, 1};
    layout::RowMajor layoutY{m, 1};

    size_t scalarSize = scalarLen * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(scalarLen);
    std::vector<ScalarType> hostBeta(scalarLen);
    golden::FillRandomData(hostAlpha, -1.0f, 1.0f);
    golden::FillRandomData(hostBeta, -1.0f, 1.0f);

    std::vector<float> hostA(lenA);
    std::vector<float> hostX(lenX);
    std::vector<float> hostY_read(lenY);
    golden::FillRandomData(hostA,  -1.0f, 1.0f);
    golden::FillRandomData(hostX,  -1.0f, 1.0f);
    golden::FillRandomData(hostY_read,  -1.0f, 1.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceY_read{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceY_read), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceY_read, sizeY, hostY_read.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceY, sizeY, hostY_read.data(), sizeY, ACL_MEMCPY_HOST_TO_DEVICE));
    
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    GemvAiv<<<aicCoreNum, nullptr, stream>>>(
        options.problemShape,
        deviceA, layoutA,
        deviceX, layoutX,
        deviceY, layoutY,
        deviceY_read,
        hostAlpha[0], hostBeta[0],
        SPLIT
    );
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostRes(lenY);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeY, deviceY, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenY);
    golden::ComputeGemv(options.problemShape, hostAlpha[0], hostBeta[0], hostA, layoutA, hostX, layoutX, hostY_read, layoutY, hostGolden, layoutY);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFree(deviceY_read));

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