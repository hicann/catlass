/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

using namespace Catlass;
using namespace tla;

struct GroupedGemmOptionsWithMxType : public GroupedGemmOptions {
    const std::string HELPER = "problem_count m n k trans_b quant_type [device_id]";

    uint32_t transB{0};
    std::string quantDataType = "float8_e4m3fn";
    const std::array<std::string, 3> quantDataTypes = {"float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2"};

    GroupedGemmOptionsWithMxType() = default;

    int Parse(int argc, const char **argv) {
        enum class ArgsIndex {
            GROUP_COUNT = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            TRANS_B_INDEX,
            QUANT_TYPE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX)
            || argc < static_cast<uint32_t>(ArgsIndex::QUANT_TYPE_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }
        problemCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_COUNT)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        transB = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::TRANS_B_INDEX)]);
        quantDataType = std::string(argv[static_cast<uint32_t>(ArgsIndex::QUANT_TYPE_INDEX)]);

        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }

        // valid check
        if (quantDataTypes.end() == std::find(quantDataTypes.begin(), quantDataTypes.end(), quantDataType)) {
            std::cerr << "Invalid quantData type: " << quantDataType << std::endl;
            return -1;
        }

        return 0;
    }
};

using Options = GroupedGemmOptionsWithMxType;

// malloc @npu device and prepare copying data
#define MALLOC_AND_COPY_TO_NPU_DEVICE(descr, hostData, size) \
    uint8_t *device##descr{nullptr}; \
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&device##descr), size, ACL_MEM_MALLOC_HUGE_FIRST)); \
    ACL_CHECK(aclrtMemcpy(device##descr, size, (hostData).data(), size, ACL_MEMCPY_HOST_TO_DEVICE))

template <typename T>
bool SaveResult(const std::string &filename, const std::vector<T> &data, const size_t dataSize) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    std::vector<char> buffer(dataSize);
    std::memcpy(buffer.data(), data.data(), dataSize);

    file.write(buffer.data(), dataSize);
    if (!file) {
        file.close();
        return false;
    }

    file.close();
    return true;
}

template <class ElementA, class ElementB,
    class L1TileShape, class L0TileShape,
    bool transB = false
>
void MxGroupedMatmulSliceM(Options const &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t groupCount = options.problemCount;
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    // compute mxScale len, k must be multiples of 2
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    using ElementC = float;
    using ElementMxScale = float8_e8m0_t;
    using ElementGroupList = int64_t;
    static_assert((std::is_same_v<ElementA, float8_e5m2_t> ||
        std::is_same_v<ElementA, float8_e4m3_t> || std::is_same_v<ElementA, float4_e2m1x2_t>) &&
        (std::is_same_v<ElementB, float8_e5m2_t> || std::is_same_v<ElementB, float8_e4m3_t> ||
            std::is_same_v<ElementB, float4_e2m1x2_t> ) && std::is_same_v<ElementMxScale, float8_e8m0_t>,
        "ElementA and ElementB must be float8_e5m2_t, float8_e4m3_t, or float4_e2m1x2_t, ElementMxScale must be float8_e8m0_t");

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::template MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity() * groupCount;
    size_t lenC = tagC.Capacity();
    size_t lenMxScaleA = m * mxScaleAlignedK;
    size_t lenMxScaleB = mxScaleAlignedK * n * groupCount;

    size_t sizeA = lenA * SizeOfBits<ElementA>::value / 8;
    size_t sizeB = lenB * SizeOfBits<ElementB>::value / 8;
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeGroupList = groupCount * sizeof(ElementGroupList);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeWorkspace = 0;

    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    std::vector<int8_t> hostMxScaleA(lenMxScaleA);
    std::vector<int8_t> hostMxScaleB(lenMxScaleB);

    // Read mxfp4 data from bin file(data_gen)
    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile("./data/a_8.bin", hostA.data(), sizeA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/b_8.bin", hostB.data(), sizeB)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/a_scale.bin", hostMxScaleA.data(), sizeMxScaleA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile("./data/b_scale.bin", hostMxScaleB.data(), sizeMxScaleB)) {
        releaseAclEarly();
        return;
    }

    // Generate average group size
    auto groupList = golden::GenerateAverageGroupList<ElementGroupList, false /* non-cumsum*/>(m, groupCount);

    // Malloc @npu device
    MALLOC_AND_COPY_TO_NPU_DEVICE(GroupList, groupList, sizeGroupList);
    MALLOC_AND_COPY_TO_NPU_DEVICE(A, hostA, sizeA);
    MALLOC_AND_COPY_TO_NPU_DEVICE(B, hostB, sizeB);
    MALLOC_AND_COPY_TO_NPU_DEVICE(MxScaleA, hostMxScaleA, sizeMxScaleA);
    MALLOC_AND_COPY_TO_NPU_DEVICE(MxScaleB, hostMxScaleB, sizeMxScaleB);

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    // kernel level
    using MatmulKernel = Gemm::Kernel::GroupedMxMatmulSliceMTla<BlockMmad,
            BlockEpilogue, BlockScheduler, ElementGroupList>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    typename MatmulKernel::Arguments arguments{options.problemShape, options.problemCount,
        deviceGroupList, deviceA, layoutA, deviceB, layoutB,
        deviceMxScaleA, layoutMxScaleA, deviceMxScaleB, layoutMxScaleB,
        deviceC, layoutC};

    // call a kernel
    MatmulAdapter matmul_op;
    // Judge whether arguments can run
    matmul_op.CanImplement(arguments);
    // get workspace
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // Initialize kernel argument
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Save GMM output
    std::vector<ElementC> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string outputFileName = "./data/result.bin";
    SaveResult<ElementC>(outputFileName, hostC, sizeC);

    // Finalize
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtFree(deviceMxScaleA));
    ACL_CHECK(aclrtFree(deviceMxScaleB));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

template <bool TB>
void Run(Options const &options) {
    if (options.quantDataType == "float8_e4m3fn") {
        MxGroupedMatmulSliceM<float8_e4m3_t, float8_e4m3_t, Shape<Int<256>,Int<256>,Int<256>>, Shape<Int<256>,Int<256>,Int<128>>, TB>(options);
    } else if (options.quantDataType == "float8_e5m2") {
        MxGroupedMatmulSliceM<float8_e5m2_t, float8_e5m2_t, Shape<Int<256>,Int<256>,Int<256>>, Shape<Int<256>,Int<256>,Int<128>>, TB>(options);
    } else if (options.quantDataType == "float4_e2m1fn_x2") {
        MxGroupedMatmulSliceM<float4_e2m1x2_t, float4_e2m1x2_t, Shape<Int<256>,Int<256>,Int<512>>, Shape<Int<256>,Int<256>,Int<256>>, TB>(options);
    } else {
        std::cerr << "Unexpected quant data-type mismatch." << std::endl;
    }
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    // execute the kernel
    options.transB ? Run<true>(options) : Run<false>(options);
    return 0;
}

