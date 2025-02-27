#include <iostream>
#include <vector>

#include "data_utils.hpp"
#include "golden.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemm/block/block_gemm.hpp"
#include "acot/gemm/kernel/kernel_groupgemm_epilogue.hpp"
#include "acot/gemm/gemm_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/gemm/dispatch_policy.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/tile/tile_elemwise_gemm.hpp"
#include "acot/epilogue/block/block_epilogue.hpp"

using namespace acot;

// 已经进入核函数了
template <
    typename LayoutA,
    typename LayoutB,
    typename LayoutC
>
ACOT_GLOBAL
void BF16CMGroupGemm(
    uint32_t problemCount,
    uint64_t fftsAddr,
    GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR ptrProblemShape,
    GM_ADDR gmA, GM_ADDR ptrLayoutA,
    GM_ADDR gmB, GM_ADDR ptrLayoutB,
    GM_ADDR gmC, GM_ADDR ptrLayoutC,
    GM_ADDR gmWorkspace
){
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = arch::AscendC910B3;
    // 开启pingpong机制
    using GemmBlockDispatchPolicy = gemm::GemmAscendC910B3Pingpong<true>;
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAscendC910B3Gemm;
    using AType = gemm::GemmType<bfloat16_t, LayoutA>;
    using BType = gemm::GemmType<bfloat16_t, LayoutB>;
    using CType = gemm::GemmType<bfloat16_t, LayoutC>;
    using XType = gemm::GemmType<float, LayoutC>;
    // 使用Coord来传递值
    using L1TileShape = MatmulShape<128, 128, 128>;
    using L0TileShape = MatmulShape<128, 128, 64>;

    // 调用block层函数
    using GemmBlock = gemm::block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, XType>; // 这个还是乘法
    // using TileElemWiseEpilogue = void;
    using DType = CType;
    using ComputeType = XType;
    constexpr uint32_t computeLength = 4096; // 128 * 128 / 2 开启双缓冲机制
    // 后处理部分
    using TileElemWiseAddGemm = epilogue::tile::TileElemWiseAddGemm<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulGemm = epilogue::tile::TileElemWiseMulGemm<ArchTag, ComputeType, computeLength>;
    // 拷贝函数实例化
    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, CType, XType, DType>;
    // 实例化Epilogue部分
    using EpilogueBlock = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulGemm, EpilogueTileCopy>;
    layout::ColumnMajor layoutC{1024, 100};  //暂用
    typename EpilogueBlock::Params epilogueParams{1.0, 1.0, gmC, layoutC, gmC, layoutC}; // x只是传了一个地址
    // 实例化Gemm部分
    using GemmKernel = gemm::kernel::KernelGroupGemmEpilogue<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemCount, ptrProblemShape, alpha, beta, gmA, ptrLayoutA, gmB, ptrLayoutB, gmWorkspace,ptrLayoutC, epilogueParams}; // 这里得修改 gmX保存A * B
    // 调用核函数
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "03_gemm/01_fp16_rm_epi_gemm groupCnt [deviceId]";
    uint32_t groupCnt = 8;

    int32_t deviceId{0}; // 成员变量

    Options() = default;

    int Parse(int argc, const char **argv){
        enum ArgsIndex{
            GROUPCNT_INDEX = 1,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if(argc > ARGS_MAX || argc <= GROUPCNT_INDEX){
            std::cerr << HELPER << std::endl;
            return -1;
        }
        // 设置矩阵形状 + 矩阵步长
        groupCnt = std::atoi(argv[GROUPCNT_INDEX]);
        if(argc == ARGS_MAX){
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        
        return 0;
    }
}Options;

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
    
    using LayoutA = layout::ColumnMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::ColumnMajor;

    // crate grouped matmul problem shapes and layouts
    std::vector<MatmulCoord> problemShapeList(groupCnt);
    std::vector<LayoutA> layoutAList(groupCnt);
    std::vector<LayoutB> layoutBList(groupCnt);
    std::vector<LayoutC> layoutCList(groupCnt);

    uint32_t allMKCnt = 0;
    uint32_t allKNCnt = 0;
    uint32_t allMNCnt = 0;
    for (uint32_t i = 0; i < groupCnt; ++i) {
        problemShapeList[i] = MatmulCoord{M_array[i], N_array[i], K_array[i]};
        layoutAList[i] = LayoutA{M_array[i], K_array[i]};
        layoutBList[i] = LayoutB{K_array[i], N_array[i]};
        layoutCList[i] = LayoutC{M_array[i], N_array[i]};
        allMKCnt += M_array[i] * K_array[i];
        allKNCnt += K_array[i] * N_array[i];
        allMNCnt += M_array[i] * N_array[i];
    }
    // std::cout << allMKCnt << " " << allKNCnt << " " << allMNCnt << std::endl;
    size_t scalarSize = groupCnt * sizeof(float);
    float* hostAlpha;
    ACL_CHECK(aclrtMallocHost((void**)(&hostAlpha), scalarSize));
    ReadFile("./data/input/alpha.bin", scalarSize, hostAlpha, scalarSize);
    float* deviceAlpha{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceAlpha), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAlpha, scalarSize, hostAlpha, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    float* hostBeta;
    ACL_CHECK(aclrtMallocHost((void**)(&hostBeta), scalarSize));
    ReadFile("./data/input/beta.bin", scalarSize, hostBeta, scalarSize);
    float* deviceBeta{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBeta), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBeta, scalarSize, hostBeta, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));
    
    size_t sizeA = allMKCnt * sizeof(bfloat16_t);
    bfloat16_t* hostA;
    ACL_CHECK(aclrtMallocHost((void**)(&hostA), sizeA));
    ReadFile("./data/input/A.bin", sizeA, hostA, sizeA);
    bfloat16_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t sizeB = allKNCnt * sizeof(bfloat16_t);
    bfloat16_t* hostB;
    ACL_CHECK(aclrtMallocHost((void**)(&hostB), sizeB));
    ReadFile("./data/input/B.bin", sizeB, hostB, sizeB);
    bfloat16_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    
    size_t sizeC = allMNCnt * sizeof(bfloat16_t);
    bfloat16_t* hostC;
    ACL_CHECK(aclrtMallocHost((void**)(&hostC), sizeC));
    ReadFile("./data/input/C.bin", sizeC, hostC, sizeC);
    bfloat16_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC, sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
    size_t sizeX = allMNCnt * sizeof(float);
    float *gmWorkspace{nullptr};
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

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    
    // 获得当前核心数
    auto aicCoreNum = arch::AscendC910B3::MaxBlock;
    BF16CMGroupGemm<LayoutA, LayoutB, LayoutC><<<aicCoreNum, nullptr, stream>>>(
        groupCnt,
        fftsAddr,
        (uint8_t*)deviceAlpha, (uint8_t*)deviceBeta,
        problemShapeListDevice,
        (uint8_t*)deviceA, layoutAListDevice,
        (uint8_t*)deviceB, layoutBListDevice,
        (uint8_t*)deviceC, layoutCListDevice,
        (uint8_t*)gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC, sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./data/output/our_res.bin",hostC,sizeC);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceAlpha));
    ACL_CHECK(aclrtFree(deviceBeta));
    ACL_CHECK(aclrtFree(problemShapeListDevice));
    ACL_CHECK(aclrtFree(layoutAListDevice));
    ACL_CHECK(aclrtFree(layoutBListDevice));
    ACL_CHECK(aclrtFree(layoutCListDevice));
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