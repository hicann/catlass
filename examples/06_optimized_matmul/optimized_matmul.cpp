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

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/layout/layout.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/optimized_matmul.hpp"
#include "act/gemm/gemm_type.hpp"

using fp16_t = op::fp16_t;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC,
    class LayoutWA,
    class LayoutWB,
    class BlockMmad
>
ACT_DEVICE
void LaunchMatmulDynamicSwizzle(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, LayoutWA layoutWA,
    GM_ADDR gmWB, LayoutWB layoutWB
)
{
    //当m>n的时候选择offset=3, direction=0的swizzle策略
    if (problemShape.m() > problemShape.n()) {
        //设置swizzle策略
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using BlockEpilogue = void;
        // kernel level
        //将BlockMmad等组件拼装成kernel
        using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        //构建输入参数
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
            gmWA, layoutWA, gmWB, layoutWB};
        // 定义matmul kernel对象
        MatmulKernel matmul;
        // 调用kernel
        matmul(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
            gmWA, layoutWA, gmWB, layoutWB};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACT_GLOBAL
void OptimizedMatmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, GM_ADDR gmWB
)
{
    //设置该样例的硬件标签
    using ArchTag = Act::Arch::AtlasA2;
    //设置ffts同步地址
    AscendC::SetSyncBaseAddr(fftsAddr);

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    // 设置dispatch policy，包括流水阶段数量，使能unitflag和shuffleK
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    // 如果LayoutA和LayoutB都是ColumnMajor，L1TileShape是使用GemmShape<256, 128, 256>效率最高
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutA, Act::layout::ColumnMajor> &&std::is_same_v<LayoutB, Act::layout::ColumnMajor>,
        GemmShape<256, 128, 256>,
        GemmShape<128, 256, 256>>;
    using L0TileShape = std::conditional_t<std::is_same_v<
        LayoutA, Act::layout::ColumnMajor> &&std::is_same_v<LayoutB, Act::layout::ColumnMajor>,
        GemmShape<256, 128, 64>,
        GemmShape<128, 256, 64>>;;
    // 如果不需要padding
    if (gmA == gmWA && gmB == gmWB) {
        using LayoutWA = LayoutA;
        using LayoutWB = LayoutB;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        //根据当前的参数输入配置BlockMmad组件
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1));
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1));
        //根据输入输出问题的shape动态选择swizzle策略
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else if (gmA == gmWA && gmB != gmWB) {
        // no need to padding A, but B needs padding.
        using LayoutWA = LayoutA;
        using LayoutWB = std::conditional_t<std::is_same_v<LayoutB, Act::layout::RowMajor>,
            Act::layout::PaddingRowMajor, Act::layout::PaddingColumnMajor>;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1));
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else if (gmA != gmWA && gmB == gmWB) {
        // no need to padding B, but A needs padding.
        using LayoutWA = std::conditional_t<std::is_same_v<LayoutA, Act::layout::RowMajor>,
            Act::layout::PaddingRowMajor, Act::layout::PaddingColumnMajor>;
        using LayoutWB = LayoutB;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1));
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else {
        // Both A and B need padding.
        using LayoutWA = std::conditional_t<std::is_same_v<LayoutA, Act::layout::RowMajor>,
            Act::layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using LayoutWB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    }
}

struct Options {
    const std::string HELPER = "06_optimizd_matmul m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

template<class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
        RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}
//这里只有当stride是小于65536并且列方向对齐了才不需要padding
//对于rowmajor判断是否需要padding是要判断行方向的stride，就是列的宽度，就是stride(0)是对齐的
bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

//对于column major是看这个函数，判断列间的步长
bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

void Run(Options const &options)
{
    //定义aclStream流
    aclrtStream stream{nullptr};

    //初始化设备，创建stream
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    //获取problem shape
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    //分别获取A B C矩阵的元素数量
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    //计算A B C矩阵的内存空间大小
    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    //数据搬运在列方向256元素对齐效率最高
    const uint32_t align = 256;

    //在样例中指定A B C矩阵的layout。其他的layout也可以支持
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // 根据layout判断A B矩阵是否需要padding，如果非256元素对齐就要padding，对齐后搬运效率更高
    bool isNeedPaddingA = IsNeedPadding(layoutA, align);
    bool isNeedPaddingB = IsNeedPadding(layoutB, align);

    // 如果LayoutA和LayoutB都是ColumnMajor，L1TileShape是使用GemmShape<256, 128, 256>效率最高
    using L1TileShape = std::conditional_t<
        std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
        GemmShape<256, 128, 256>,     // 上方条件同时成立
        GemmShape<128, 256, 256>>;    // 否则使用这个选项

    // 计算Workspace的占用空间大小，这里输入的layout，行方向的分块个数，列方向的分块个数
    // 用来给每个L1Tile块申请空间
    size_t sizeWA = GetWorkspaceLen(layoutA, L1TileShape::M, L1TileShape::K) * sizeof(fp16_t);
    size_t sizeWB = GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(fp16_t);


    // 定义host侧用于cpu标杆计算的空间，并初始化数据填充
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    // 申请device侧A、B、C矩阵的指针，并将数据拷贝到device侧
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWA{nullptr};
    //根据是否需要padding进行判断，需要padding则单独申请空间，不需要padding则使workspaceA指向device侧A矩阵空间
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }
    // 逻辑同上
    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    // FFTS空间申请
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // 获取当前硬件的cube core数量
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // 内核调用符调用核函数执行
    // 需要传递到内核使其感知的参数，包括ffts的地址，problem shape，设备侧A/B/C矩阵的地址和layout，A/B矩阵的workspace空间
    OptimizedMatmul<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC,
        deviceWA, deviceWB);
    //流同步
    ACL_CHECK(aclrtSynchronizeStream(stream));

    //在数据计算完成之后从device侧将数据拷贝到host侧
    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    //准备CPU计算的标杆数据
    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    //精度判断
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    //释放设备侧资源
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    //如果进行了对齐操作，workspace资源也需要释放
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    //销毁stream
    ACL_CHECK(aclrtDestroyStream(stream));
    //重置device
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}
// 算子样例入口，从命令行读取problemshape参数，并调用Run执行
int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
