/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv/kernel/conv3d_bias.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/conv/device/device_conv.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

struct Options {
    const std::string HELPER = "00_basic_conv n cin d h w cout kd kh kw sD sH sW dD dH dW pD pH pW [device_id]";

    uint32_t oriFmapShape[5] = {1, 1, 1, 2, 9};
    uint32_t oriFilterShape[5] = {1, 1, 1, 1, 1};
    uint32_t strides[3] = {1, 1, 1};
    uint32_t pads[6] = {0, 0, 0};
    uint32_t dilations[3] = {1, 1, 1};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            N_INDEX = 1,
            CIN_INDEX,
            D_INDEX,
            H_INDEX,
            W_INDEX,
            COUT_INDEX,
            KD_INDEX,
            KH_INDEX,
            KW_INDEX,
            SD_INDEX,
            SH_INDEX,
            SW_INDEX,
            DD_INDEX,
            DH_INDEX,
            DW_INDEX,
            PD_INDEX,
            PH_INDEX,
            PW_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= PW_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        oriFmapShape[0] = std::atoi(argv[N_INDEX]);
        oriFmapShape[1] = std::atoi(argv[CIN_INDEX]);
        oriFmapShape[2] = std::atoi(argv[D_INDEX]);
        oriFmapShape[3] = std::atoi(argv[H_INDEX]);
        oriFmapShape[4] = std::atoi(argv[W_INDEX]);
        oriFilterShape[0] = std::atoi(argv[COUT_INDEX]);
        oriFilterShape[1] = std::atoi(argv[CIN_INDEX]);
        oriFilterShape[2] = std::atoi(argv[KD_INDEX]);
        oriFilterShape[3] = std::atoi(argv[KH_INDEX]);
        oriFilterShape[4] = std::atoi(argv[KW_INDEX]);
        strides[0] = std::atoi(argv[SD_INDEX]);
        strides[1] = std::atoi(argv[SH_INDEX]);
        strides[2] = std::atoi(argv[SW_INDEX]);
        dilations[0] = std::atoi(argv[DD_INDEX]);
        dilations[1] = std::atoi(argv[DH_INDEX]);
        dilations[2] = std::atoi(argv[DW_INDEX]);
        pads[0] = std::atoi(argv[PD_INDEX]);
        pads[1] = std::atoi(argv[PH_INDEX]);
        pads[2] = std::atoi(argv[PW_INDEX]);

        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }

        return 0;
    }
};

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    Conv3dParams problemShape = Conv3dParams::MakeConvCoord<fp16_t>(options.oriFmapShape, options.oriFilterShape, options.pads, options.dilations, options.strides);

    uint32_t N = problemShape.n();
    uint32_t di = problemShape.di();
    uint32_t cin = problemShape.cin();
    uint32_t cin1 = problemShape.cin1();
    uint32_t hi = problemShape.hi();
    uint32_t wi = problemShape.wi();
    uint32_t cin0 = problemShape.cin0();
    uint32_t kd = problemShape.kd();
    uint32_t kh = problemShape.kh();
    uint32_t kw = problemShape.kw();
    uint32_t kdc1khkw = problemShape.kdc1khkw();
    uint32_t n1 = problemShape.n1();
    uint32_t n0 = problemShape.n0();
    uint32_t dout = problemShape.dout();
    uint32_t ho = problemShape.ho();
    uint32_t wo = problemShape.wo();
    uint32_t cout1 = problemShape.cout1();
    uint32_t cout0 = problemShape.cout0();
    uint32_t cout = problemShape.cout();

    size_t lenFmap = static_cast<size_t>(N) * di * cin1 * hi * wi * cin0;
    size_t lenFilter = static_cast<size_t>(kdc1khkw) * n1 * n0 * cin0;
    size_t lenBias = static_cast<size_t>(cout);
    size_t lenOut = static_cast<size_t>(N) * dout * cout1 * ho * wo * cout0;

    size_t sizeFmap = lenFmap * sizeof(fp16_t);
    size_t sizeFilter = lenFilter * sizeof(fp16_t);
    size_t sizeOut = lenOut * sizeof(fp16_t);
    size_t sizeBias = lenBias * sizeof(float);

    using LayoutFmap = layout::NDC1HWC0;
    using LayoutFilter = layout::KDC1KHKWN1N0C0;
    using LayoutOut = layout::NDC1HWC0;
    using LayoutBias = layout::VectorLayout;
    LayoutFmap layoutFmap{N, cin, di, hi, wi};
    LayoutFilter layoutFilter{cout, cin, kd, kh, kw};
    LayoutOut layoutOut{N, cout, dout, ho, wo};
    LayoutBias layoutBias{cout};

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<float> hostBias(lenBias);   //////vectorlayout
    golden::FillRandomData<fp16_t>(hostFmap, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostFilter, -5.0f, 5.0f);
    golden::FillRandomData<float>(hostBias, -5.0f, 5.0f);

    uint8_t *deviceFmap{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFmap), sizeFmap, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFmap, sizeFmap, hostFmap.data(), sizeFmap, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceFilter{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceFilter), sizeFilter, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceFilter, sizeFilter, hostFilter.data(), sizeFmap, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceOut{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceOut), sizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    constexpr uint32_t l1AStages = 1;
    constexpr uint32_t l1BStages = 1;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Conv::ConvAtlasA2Pingpong<
        l1AStages, l1BStages,
        l0AStages, l0BStages,
        l0CStages, enableUnitFlag
    >;

    using FmapType = Gemm::GemmType<half, LayoutFmap>;
    using FilterType = Gemm::GemmType<half, LayoutFilter>;
    using BiasType = Gemm::GemmType<float, LayoutBias>;
    using OutType = Gemm::GemmType<half, LayoutOut>;
    using CoreTileShape = ConvCoreShape<1, 1, 1, 9>;
    using FmapL1TileShape = ConvFmapL1Shape<16, 1, 1>;
    using FilterL1TileShape = ConvFilterL1Shape<1, 1, 16>;
    using L0TileShape = ConvL0Shape<16, 16, 16>;

    using BlockConv = Conv::Block::BlockConv<DispatchPolicy, CoreTileShape, FmapL1TileShape, FilterL1TileShape, L0TileShape, FmapType, FilterType, OutType, BiasType>;
    using BlockEpilogue = void;    //////是否需要尾块处理

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Conv::Block::Conv3dIdentityBlockSwizzle<3, 0>;

    // kernel level
    using ConvKernel = Conv::Kernel::ConvBias<BlockConv, BlockEpilogue, BlockScheduler>;

    using ConvAdapter = Conv::Device::DeviceConv<ConvKernel>;
    ConvKernel::Arguments arguments{problemShape, deviceFmap, deviceFilter, deviceBias, deviceOut};
    ConvAdapter conv_op;   //////TODO
    conv_op.CanImplement(arguments);
    size_t sizeWorkspace = conv_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    conv_op.Initialize(arguments, deviceWorkspace);
    conv_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    // std::vector<fp16_t> hostOut(lenOut);
    // ACL_CHECK(aclrtMemcpy(hostOut.data(), sizeOut, deviceOut, sizeOut, ACL_MEMCPY_DEVICE_TO_HOST));

    // std::vector<float> hostGolden(lenC);    ///////////重新写
    // golden::ComputeConv3d(options.problemShape, hostFmap, layoutFmap, hostFilter, layoutFilter, hostBias, layoutBias, hostGolden, layoutOut);

    // std::vector<uint64_t> errorIndices = golden::CompareData(hostOut, hostGolden, k);
    // if (errorIndices.empty()) {
    //     std::cout << "Compare success." << std::endl;
    // } else {
    //     std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    // }

    ACL_CHECK(aclrtFree(deviceFmap));
    ACL_CHECK(aclrtFree(deviceFilter));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(deviceOut));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}