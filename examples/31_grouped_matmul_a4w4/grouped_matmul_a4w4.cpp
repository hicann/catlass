/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <bitset>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "bfloat16.h"
#include "fp16_t.h"
#include "int4.h"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_a4w4.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/status.hpp"

using int4_t = op::int4;
using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;
struct Options
{
    const std::string HELPER = "31_grouped_matmul_a4w4 groupCnt mlist nlist klist glist [device_id]";
    uint32_t groupCnt = 1;
    std::vector<uint32_t> mList;
    std::vector<uint32_t> nList;
    std::vector<uint32_t> kList;
    std::vector<uint32_t> gList;
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex
        {
            GROUPCNT_INDEX = 1,
            MLIST_INDEX,
            NLIST_INDEX,
            KLIST_INDEX,
            GLIST_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= GLIST_INDEX)
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        groupCnt = std::atoi(argv[GROUPCNT_INDEX]);
        parseList(argv[MLIST_INDEX], mList);
        parseList(argv[NLIST_INDEX], nList);
        parseList(argv[KLIST_INDEX], kList);
        parseList(argv[GLIST_INDEX], gList);

        if (mList.size() != groupCnt || nList.size() != groupCnt || kList.size() != groupCnt || gList.size() != groupCnt)
        {
            std::cerr << "List lengths do not match groupCnt." << std::endl;
            return -1;
        }

        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }

private:
    void parseList(const std::string &str, std::vector<uint32_t> &list)
    {
        std::istringstream stream(str);
        std::string token;

        while (std::getline(stream, token, ','))
        {
            if (!token.empty())
            {
                try
                {
                    size_t pos = 0;
                    uint32_t value = std::stoul(token, &pos);
                    if (pos == token.length())
                    {
                        list.push_back(value);
                    }
                    else
                    {
                        std::cerr << "Warning: Invalid character in number: " << token << std::endl;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error converting '" << token << "': " << e.what() << std::endl;
                }
            }
        }
    }
};

std::vector<int8_t> convertInt4ToInt8(const std::vector<int4_t> &input)
{
    std::vector<int8_t> result;
    for (size_t i = 0; i < input.size(); i += 2)
    {
        // 每两个int4_t拼接成一个int8_t
        int8_t high = input[i + 1].get_value() & 0x0F; // 高4位
        int8_t low = input[i].get_value() & 0x0F;      // 低4位
        int8_t combine = high << 4 | low;
        result.push_back(combine);
    }
    return result;
}

std::vector<uint64_t> convertQuantTensor(const std::vector<bfloat16> &input)
{
    std::vector<uint64_t> result;
    for (size_t i = 0; i < input.size(); i++)
    {
        bfloat16 bf16_val = input[i];
        float fp32_val = static_cast<float>(bf16_val);
        // 将 float 的二进制表示解释为 uint32_t
        uint32_t uint32_val;
        std::memcpy(&uint32_val, &fp32_val, sizeof(uint32_t));
        uint64_t uint64_val = static_cast<uint64_t>(uint32_val);
        result.push_back(uint64_val);
    }
    return result;
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t groupCnt = options.groupCnt;

    using OutCHostType = fp16_t;
    using OutCKernelType = half;

    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::RowMajor;
    // using LayoutB = Catlass::layout::zN;
    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutC = Catlass::layout::RowMajor;

    std::vector<Catlass::GemmCoord> problemShapeList(groupCnt);
    std::vector<LayoutA> layoutAList(groupCnt);
    std::vector<LayoutB> layoutBList(groupCnt);
    std::vector<LayoutQ> layoutQList(groupCnt);
    std::vector<LayoutC> layoutCList(groupCnt);

    uint64_t allMKCnt = 0;
    uint64_t allKNCnt = 0;
    uint64_t allQNCnt = 0;
    uint64_t allMNCnt = 0;
    for (uint32_t i = 0; i < groupCnt; ++i)
    {
        problemShapeList[i] = Catlass::GemmCoord{options.mList[i], options.nList[i], options.kList[i]};
        layoutAList[i] = LayoutA{options.mList[i], options.kList[i]};
        layoutBList[i] = LayoutB::template MakeLayout<AscendC::int4b_t>(options.kList[i], options.nList[i]);
        layoutQList[i] = LayoutQ{options.gList[i], options.nList[i]};
        layoutCList[i] = LayoutC{options.mList[i], options.nList[i]};

        allMKCnt += options.mList[i] * options.kList[i];
        allKNCnt += options.kList[i] * options.nList[i];
        allQNCnt += options.gList[i] * options.nList[i];
        allMNCnt += options.mList[i] * options.nList[i];
    }

    std::vector<int4_t> hostA(allMKCnt);
    std::vector<int4_t> hostB(allKNCnt);
    std::vector<bfloat16> hostQ(allQNCnt);
    std::vector<uint64_t> kernelQ(allQNCnt);

    // 左右矩阵全随机
    Catlass::golden::FillRandomData<int4_t>(hostA, -8, 7);
    Catlass::golden::FillRandomData<int4_t>(hostB, -8, 7);
    Catlass::golden::FillRandomData<bfloat16>(hostQ, 1.0, 1.0);
    kernelQ = convertQuantTensor(hostQ);

    std::vector<float> hostGolden(allMNCnt);

    Catlass::golden::ComputeGroupMatmulPerGroupQuant(groupCnt, problemShapeList, hostA, layoutAList, hostB, layoutBList, hostQ, layoutQList, hostGolden, layoutCList);

    std::vector<int8_t> hostA_int8 = convertInt4ToInt8(hostA);
    std::vector<int8_t> hostB_int8 = convertInt4ToInt8(hostB);

    size_t sizeA = static_cast<size_t>(allMKCnt * int4_t::size_of());
    size_t sizeB = static_cast<size_t>(allKNCnt * int4_t::size_of());
    size_t sizeQ = static_cast<size_t>(allQNCnt * sizeof(uint64_t));
    size_t sizeC = static_cast<size_t>(allMNCnt * sizeof(OutCHostType));

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA_int8.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB_int8.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceQ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceQ), sizeQ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceQ, sizeQ, kernelQ.data(), sizeQ, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *problemShapeListDevice{nullptr};
    size_t sizeProblemShapeList = problemShapeList.size() * sizeof(Catlass::GemmCoord);
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

    uint8_t *layoutQListDevice{nullptr};
    size_t sizeLayoutQList = layoutQList.size() * sizeof(LayoutQ);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutQListDevice), sizeLayoutQList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutQListDevice, sizeLayoutQList,
                          layoutQList.data(), sizeLayoutQList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutCListDevice{nullptr};
    size_t sizeLayoutCList = layoutCList.size() * sizeof(LayoutC);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutCListDevice), sizeLayoutCList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutCListDevice, sizeLayoutCList,
                          layoutCList.data(), sizeLayoutCList, ACL_MEMCPY_HOST_TO_DEVICE));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Catlass::Arch::AtlasA2;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = Catlass::GemmShape<64, 64, 64>;
    using L0TileShape = Catlass::GemmShape<64, 64, 64>;

    using AType = Catlass::Gemm::GemmType<AscendC::int4b_t, LayoutA>;
    using BType = Catlass::Gemm::GemmType<AscendC::int4b_t, LayoutB>;
    using CType = Catlass::Gemm::GemmType<OutCKernelType, LayoutC>;
    using QType = Catlass::Gemm::GemmType<uint64_t, LayoutQ>;

    using BlockMmad = Catlass::Gemm::Block::BlockGmmQuant<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, QType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::MatmulA4W4<BlockMmad, BlockEpilogue, BlockScheduler>;

    using MatmulAdapter = Catlass::Gemm::Device::DeviceGemm<MatmulKernel>;
    typename MatmulKernel::Arguments arguments{
        groupCnt, problemShapeListDevice,
        deviceA, layoutAListDevice,
        deviceB, layoutBListDevice,
        deviceQ, layoutQListDevice,
        deviceC, layoutCListDevice};
    MatmulAdapter matmul_op;
    matmul_op.CanImplement(arguments);
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0)
    {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0)
    {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    std::vector<OutCHostType> hostC(allMNCnt);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<uint64_t> errorIndices = Catlass::golden::CompareData(hostC, hostGolden, options.kList[0]);
    if (errorIndices.empty())
    {
        std::cout << "Compare success." << std::endl;
    }
    else
    {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceQ));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(problemShapeListDevice));
    ACL_CHECK(aclrtFree(layoutAListDevice));
    ACL_CHECK(aclrtFree(layoutBListDevice));
    ACL_CHECK(aclrtFree(layoutQListDevice));
    ACL_CHECK(aclrtFree(layoutCListDevice));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0)
    {
        return -1;
    }
    Run(options);
    return 0;
}