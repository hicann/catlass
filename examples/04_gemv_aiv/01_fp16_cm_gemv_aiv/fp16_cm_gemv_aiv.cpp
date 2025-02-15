#include <iostream>
#include <vector>
#include <iostream>

#include "data_utils.hpp"
#include "helper.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemv_aiv/block/block_gemv.hpp"
#include "acot/gemv_aiv/block/block_swizzle.hpp"
#include "acot/gemv_aiv/dispatch_policy.hpp"

#include "acot/gemv_aiv/kernel/kernel_gemv_aiv.hpp"
#include "acot/gemv_aiv/gemv_type.hpp"
#include "acot/layout/layout.hpp"


using namespace acot;

// 已经进入核函数了
// 单纯行优先
ACOT_GLOBAL
void FP16CMGemvAiv(
    MatmulCoord problemShape,
    __gm__ half* gmA, layout::ColumnMajor layoutA,
    __gm__ half* gmX, layout::ColumnMajor layoutX,
    __gm__ half* gmY, layout::ColumnMajor layoutY
){
    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = gemv::MmadAtlasA2Pingpong<true>;
    using UBTileShape = MatmulShape<512, 32, 1>;

    using AType = gemv::GemvType<half, layout::ColumnMajor>;
    using XType = gemv::GemvType<half, layout::ColumnMajor>;
    using YType = gemv::GemvType<half, layout::ColumnMajor>;

    // 调用block层函数
    using GemvBlock = gemv::block::BlockGemv<DispatchPolicy, UBTileShape, AType, XType, YType>;
    using BlockEpilogue = void;
    using TileScheduler = typename gemv::block::MatmulIdentityBlockSwizzle<3, 0>;

    // kernel level
    using GemvKernel = gemv::kernel::KernelGemv<GemvBlock, BlockEpilogue, TileScheduler>;
    typename GemvKernel::Params params{problemShape, gmA, layoutA, gmX, gmY};
    

    // call a kernel
    GemvKernel gemv;

    gemv(params);

}

typedef struct Options{
    const std::string HELPER = "04_gemv_aiv/01_fp16_cm_gemv_aiv m n [device_id]";

    uint32_t M = 32;
    uint32_t N = 32;

    Options() = default;
    
    MatmulCoord problemShape{M, N, 1};
    int32_t deviceId{0}; // 成员变量

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
        // 设置矩阵形状 + 矩阵步长
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = 1;
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
    uint32_t k = 1;

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenY = static_cast<size_t>(m) * 1;

    size_t sizeA = lenA * sizeof(half);
    size_t sizeX = lenX * sizeof(half);
    size_t sizeY = lenY * sizeof(half);
    
    layout::ColumnMajor layoutA{m, n};
    layout::ColumnMajor layoutX{n, 1};
    layout::ColumnMajor layoutY{m, 1};

    half* hostA;
    ACL_CHECK(aclrtMallocHost((void**)(&hostA),sizeA));
    ReadFile("./data/input/matrix_gm.bin", sizeA, hostA, sizeA);
    half *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    half* hostX;
    ACL_CHECK(aclrtMallocHost((void**)(&hostX), sizeX));
    ReadFile("./data/input/vector_gm.bin", sizeX, hostX, sizeX);
    half *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX, sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    half *deviceY{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 获得当前核心数
    auto aicCoreNum = arch::AscendC910B3::MaxBlock;
    FP16CMGemvAiv<<<aicCoreNum, nullptr, stream>>>(
        options.problemShape,
        deviceA, layoutA,
        deviceX, layoutX,
        deviceY, layoutY);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    half* hostY;
    ACL_CHECK(aclrtMallocHost((void**)(&hostY), sizeY));
    ACL_CHECK(aclrtMemcpy(hostY, sizeY, deviceY, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./data/output/our_res.bin",hostY,sizeY);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostX));
    ACL_CHECK(aclrtFreeHost(hostY));

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