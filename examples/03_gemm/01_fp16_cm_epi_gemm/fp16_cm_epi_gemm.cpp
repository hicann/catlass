#include <iostream>
#include <vector>

#include "data_utils.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemm/block/block_gemm.hpp"
#include "acot/gemm/kernel/kernel_gemm_epilogue.hpp"
#include "acot/gemm/gemm_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/gemm/dispatch_policy.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/tile/tile_elemwise_gemm.hpp"
#include "acot/epilogue/block/block_epilogue.hpp"

using namespace acot;

using ScalarType = float;

// 已经进入核函数了
template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACOT_GLOBAL
void FP16EpiGemm(
    uint64_t fftsAddr,
    ScalarType alpha, ScalarType beta,
    MatmulCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWorkspace
){
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = arch::AscendC910B3;
    // 开启pingpong机制
    using GemmBlockDispatchPolicy = gemm::GemmAscendC910B3Pingpong<true>;
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAscendC910B3Gemm;
    using AType = gemm::GemmType<half, LayoutA>;
    using BType = gemm::GemmType<half, LayoutB>;
    using CType = gemm::GemmType<half, LayoutC>;
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
    typename EpilogueBlock::Params epilogueParams{alpha, beta, gmC, layoutC, gmC, layoutC}; // x只是传了一个地址
    // 实例化Gemm部分
    using GemmKernel = gemm::kernel::KernelGemmEpilogue<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmWorkspace, epilogueParams}; // 这里得修改 gmX保存A * B
    // 调用核函数
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "03_gemm/01_fp16_cm_epi_gemm m n k [device_id]";

    MatmulCoord problemShape{128, 128, 128};
    int32_t deviceId{0}; // 成员变量

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
        // 设置矩阵形状 + 矩阵步长
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if(argc == ARGS_MAX){
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
}Options;

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
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenX = lenC; // A * B  
    // size_t lenD = lenX; // 最后的大小
    
    size_t sizeA = lenA * sizeof(half);
    size_t sizeB = lenB * sizeof(half);
    size_t sizeC = lenC * sizeof(half);
    size_t sizeX = lenX * sizeof(float);
    // size_t sizeD = sizeX;

    layout::ColumnMajor layoutA{m, k};
    layout::ColumnMajor layoutB{k, n};
    layout::ColumnMajor layoutC{m, n};
    // layout::RowMajor layoutX{m, n};
    // layout::RowMajor layoutD{m, n}; // 最后的答案矩阵

    size_t scalarSize = 1 * sizeof(float);
    float* alpha;
    ACL_CHECK(aclrtMallocHost((void**)(&alpha), scalarSize));
    ReadFile("./data/input/alpha.bin", scalarSize, alpha, scalarSize);
    
    float* beta;
    ACL_CHECK(aclrtMallocHost((void**)(&beta), scalarSize));
    ReadFile("./data/input/beta.bin", scalarSize, beta, scalarSize);

    half* hostA;
    ACL_CHECK(aclrtMallocHost((void**)(&hostA), sizeA));
    ReadFile("./data/input/A.bin", sizeA, hostA, sizeA);
    half *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    half* hostB;
    ACL_CHECK(aclrtMallocHost((void**)(&hostB), sizeB));
    ReadFile("./data/input/B.bin", sizeB, hostB, sizeB);
    half *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    half* hostC;
    ACL_CHECK(aclrtMallocHost((void**)(&hostC), sizeC));
    ReadFile("./data/input/C.bin", sizeC, hostC, sizeC);
    half *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC, sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
    float *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // half *deviceD{nullptr};
    // ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // 获得当前核心数
    auto aicCoreNum = arch::AscendC910B3::MaxBlock;
    FP16EpiGemm<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        alpha[0], beta[0],
        options.problemShape,
        (uint8_t*)deviceA, layoutA,
        (uint8_t*)deviceB, layoutB,
        (uint8_t*)deviceC, layoutC,
        (uint8_t*)gmWorkspace);
        // (uint8_t*)deviceX, layoutX,
        // (uint8_t*)deviceD, layoutD);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // half* hostD;
    // ACL_CHECK(aclrtMallocHost((void**)(&hostD), sizeD));
    ACL_CHECK(aclrtMemcpy(hostC, sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./data/output/our_res.bin",hostC,sizeC);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    // ACL_CHECK(aclrtFree(deviceD));
    ACL_CHECK(aclrtFree(gmWorkspace));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostB));
    ACL_CHECK(aclrtFreeHost(hostC));
    // ACL_CHECK(aclrtFreeHost(hostD));

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