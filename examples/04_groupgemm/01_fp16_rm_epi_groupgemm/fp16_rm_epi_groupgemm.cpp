#include <iostream>
#include <vector>
#include <stdio.h>

#include "data_utils.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemm/block/block_gemm.hpp"
#include "acot/gemm/kernel/kernel_groupgemm_epilogue.hpp"
#include "acot/matmul/matmul_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matmul/dispatch_policy.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/tile/tile_elemwise_gemm.hpp"
#include "acot/epilogue/block/block_epilogue.hpp"

using namespace acot;
using ScalarType = float;
// 已经进入核函数了
template <
    typename LayoutA,
    typename LayoutB,
    typename LayoutC
>
ACOT_GLOBAL
void FP16CMGroupGemm(
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
    // 开启pingpong机制s
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using GemmBlockDispatchPolicy = matmul::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using AType = matmul::MatmulType<half, LayoutA>;
    using BType = matmul::MatmulType<half, LayoutB>;
    using CType = matmul::MatmulType<half, LayoutC>;
    using XType = matmul::MatmulType<half, LayoutC>;
    // 使用Coord来传递值
    using L1TileShape = MatmulShape<128, 256, 256>;
    using L0TileShape = MatmulShape<128, 256, 64>;

    // 调用block层函数
    using GemmBlock = gemm::block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, XType>; // 这个还是乘法
    // using TileElemWiseEpilogue = void;
    using DType = CType;
    using ComputeType = XType;
    constexpr uint32_t computeLength = 16384; // 128 * 128 / 2 开启双缓冲机制
    // 后处理部分
    using TileElemWiseAddGemm = epilogue::tile::TileElemWiseAddGemm<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulGemm = epilogue::tile::TileElemWiseMulGemm<ArchTag, ComputeType, computeLength>;
    // 拷贝函数实例化
    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, CType, XType, DType>;
    // 实例化Epilogue部分
    using EpilogueBlock = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulGemm, EpilogueTileCopy>;
    // typename EpilogueBlock::Params epilogueParams{gmC, gmC}; // x只是传了一个地址
    // 实例化Gemm部分
    using GemmKernel = gemm::kernel::KernelGroupGemmEpilogue<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemCount,ptrProblemShape, alpha, beta, gmA, ptrLayoutA, gmB, ptrLayoutB, gmWorkspace, ptrLayoutC,gmWA, ptrlayoutWA, gmWB, ptrlayoutWB, gmC, gmC}; // 这里得修改 gmX保存A * B
    // 调用核函数
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "04_groupgemm/01_fp16_rm_epi_groupgemm groupCnt [deviceId]";
    uint32_t groupCnt = 8;

    int32_t deviceId{0}; // 成员变量
    uint32_t mode{0};

    Options() = default;

    int Parse(int argc, const char **argv){
        enum ArgsIndex{
            GROUPCNT_INDEX = 1,
            DEVICE_ID_INDEX,
            MODE_INDEX,
            ARGS_MAX
        };
        if(argc > ARGS_MAX || argc <= GROUPCNT_INDEX){
            std::cerr << HELPER << std::endl;
            return -1;
        }
        // 设置矩阵形状 + 矩阵步长
        groupCnt = std::atoi(argv[GROUPCNT_INDEX]);
        if(argc >= ARGS_MAX - 1){
            mode = std::atoi(argv[MODE_INDEX]);
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

void Run(Options& options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t groupCnt = options.groupCnt;
    
    size_t arraySize = groupCnt * sizeof(uint32_t);
    uint32_t* M_array;
    uint32_t* N_array;
    uint32_t* K_array;
    ACL_CHECK(aclrtMallocHost((void**)(&M_array), arraySize));
    ACL_CHECK(aclrtMallocHost((void**)(&N_array), arraySize));
    ACL_CHECK(aclrtMallocHost((void**)(&K_array), arraySize));
    ReadFile("./data/input/M_array.bin", arraySize, M_array, arraySize);
    ReadFile("./data/input/N_array.bin", arraySize, N_array, arraySize);
    ReadFile("./data/input/K_array.bin", arraySize, K_array, arraySize);
    
    const uint32_t align = 256; //M 和 K的L1切分粒度
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
    size_t scalarSize = groupCnt * sizeof(float);
    float* hostAlpha;
    ACL_CHECK(aclrtMallocHost((void**)(&hostAlpha), scalarSize));
    float* hostBeta;
    ACL_CHECK(aclrtMallocHost((void**)(&hostBeta), scalarSize));
    
    size_t sizeA = allMKCnt * sizeof(half);
    half* hostA;
    ACL_CHECK(aclrtMallocHost((void**)(&hostA), sizeA));
    size_t sizeB = allKNCnt * sizeof(half);
    half* hostB;
    ACL_CHECK(aclrtMallocHost((void**)(&hostB), sizeB));
    size_t sizeC = allMNCnt * sizeof(half);
    half* hostC;
    ACL_CHECK(aclrtMallocHost((void**)(&hostC), sizeC));
    if(options.mode == 0){
        ReadFile("./data/input/alpha.bin", scalarSize, hostAlpha, scalarSize);
        ReadFile("./data/input/beta.bin", scalarSize, hostBeta, scalarSize);
        ReadFile("./data/input/A.bin", sizeA, hostA, sizeA);
        ReadFile("./data/input/B.bin", sizeB, hostB, sizeB);
        ReadFile("./data/input/C.bin", sizeC, hostC, sizeC);
    }
    float* deviceAlpha{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceAlpha), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAlpha, scalarSize, hostAlpha, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    float* deviceBeta{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBeta), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBeta, scalarSize, hostBeta, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    half *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    size_t sizeWA = allMKCnt_padding * sizeof(half);
    half *deviceWA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));

    half *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    size_t sizeWB = allKNCnt_padding * sizeof(half);
    half *deviceWB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));


    half* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC, sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
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
    
    // 获得当前核心数
    auto aicCoreNum = arch::AscendC910B3::MaxBlock;
    FP16CMGroupGemm<LayoutA, LayoutB, LayoutC><<<aicCoreNum, nullptr, stream>>>(
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

    ACL_CHECK(aclrtMemcpy(hostC, sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    if(options.mode == 0){
        WriteFile("./data/output/our_res.bin",hostC,sizeC);
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
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostB));
    ACL_CHECK(aclrtFreeHost(hostC));
    ACL_CHECK(aclrtFreeHost(hostAlpha));
    ACL_CHECK(aclrtFreeHost(hostBeta));
    ACL_CHECK(aclrtFreeHost(M_array));
    ACL_CHECK(aclrtFreeHost(N_array));
    ACL_CHECK(aclrtFreeHost(K_array));

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