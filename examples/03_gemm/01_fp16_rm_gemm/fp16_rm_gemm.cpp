#include <iostream>
#include <vector>

#include "data_utils.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemm/block/block_gemm.hpp"
#include "acot/gemm/kernel/kernel_gemm_PL_PA_epilogue.hpp"
#include "acot/gemm/gemm_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/gemm/dispatch_policy.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/tile/tile_elemwise_gemm.hpp"
#include "acot/epilogue/block/block_epilogue.hpp"

using namespace acot;

using ScalarType = half;

// 已经进入核函数了
template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACOT_GLOBAL
void FP16RMGemm(
    uint64_t fftsAddr,
    ScalarType alpha, ScalarType beta,
    MatmulCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, LayoutA layoutWA,
    GM_ADDR gmWB, LayoutB layoutWB,
    GM_ADDR gmWorkspace
){
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = arch::AscendC910B3;
    // 开启pingpong机制
    // 开启shuffleK机制
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using GemmBlockDispatchPolicy = gemm::GemmAscendC910B3Preload<enableUnitFlag, enableShuffleK>;
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAscendC910B3Gemm;
    using AType = gemm::GemmType<half, LayoutA>;
    using BType = gemm::GemmType<half, LayoutB>;
    using CType = gemm::GemmType<half, LayoutC>;
    using XType = gemm::GemmType<half, LayoutC>;
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
    typename EpilogueBlock::Params epilogueParams{alpha, beta, gmC, layoutC, gmC, layoutC}; // x只是传了一个地址
    // 实例化Gemm部分
    using GemmKernel = gemm::kernel::KernelGemmEpilogue<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmWorkspace, gmWA, layoutWA, gmWB, layoutWB, epilogueParams}; // 这里得修改 gmX保存A * B
    // 调用核函数
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "03_gemm/01_fp16_rm_gemm m n k [device_id]";

    MatmulCoord problemShape{128, 128, 128};
    int32_t deviceId{0}; // 成员变量
    uint32_t mode{0}; // 默认测数据
    Options() = default;

    int Parse(int argc, const char **argv){
        enum ArgsIndex{
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            MODE_INDEX,
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
        if(argc >=  ARGS_MAX - 1){
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
    size_t sizeX = lenX * sizeof(half);

    const uint32_t align = 256;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align); // 就是stride方向进行padding操作
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(half);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(half);

    size_t scalarSize = 1 * sizeof(half);
    half* alpha;
    ACL_CHECK(aclrtMallocHost((void**)(&alpha), scalarSize));
    half* beta;
    ACL_CHECK(aclrtMallocHost((void**)(&beta), scalarSize));
    half* hostA;
    ACL_CHECK(aclrtMallocHost((void**)(&hostA), sizeA));
    half* hostB;
    ACL_CHECK(aclrtMallocHost((void**)(&hostB), sizeB));
    half* hostC;
    ACL_CHECK(aclrtMallocHost((void**)(&hostC), sizeC));
    if(options.mode == 0){ // 测数据
        ReadFile("./data/input/alpha.bin", scalarSize, alpha, scalarSize);
        ReadFile("./data/input/beta.bin", scalarSize, beta, scalarSize);
        ReadFile("./data/input/A.bin", sizeA, hostA, sizeA);
        ReadFile("./data/input/B.bin", sizeB, hostB, sizeB);
        ReadFile("./data/input/C.bin", sizeC, hostC, sizeC);
    }
    half *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    half *deviceWA{nullptr};
    if (IsSameStride(layoutWA, layoutA)) {
        deviceWA = deviceA;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    half *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    half *deviceWB{nullptr};
    if (IsSameStride(layoutWB, layoutB)) {
        deviceWB = deviceB;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    half *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC, sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
    half *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // half *deviceD{nullptr};
    // ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // 获得当前核心数
    auto aicCoreNum = arch::AscendC910B3::MaxBlock;
    FP16RMGemm<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        alpha[0], beta[0],
        options.problemShape,
        (uint8_t*)deviceA, layoutA,
        (uint8_t*)deviceB, layoutB,
        (uint8_t*)deviceC, layoutC,
        (uint8_t*)deviceWA, layoutWA,
        (uint8_t*)deviceWB, layoutWB,
        (uint8_t*)gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // half* hostD;
    // ACL_CHECK(aclrtMallocHost((void**)(&hostD), sizeD));
    ACL_CHECK(aclrtMemcpy(hostC, sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    if(options.mode == 0){
        WriteFile("./data/output/our_res.bin",hostC,sizeC);
    }

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