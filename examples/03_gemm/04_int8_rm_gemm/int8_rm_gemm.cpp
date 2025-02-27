#include <iostream>
#include <vector>
#include <iostream>

#include "data_utils.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemm/block/block_gemm.hpp"
#include "acot/gemm/kernel/kernel_gemm.hpp"
#include "acot/gemm/gemm_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/gemm/dispatch_policy.hpp"

using namespace acot;

// 已经进入核函数了
template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACOT_GLOBAL
void INT8Gemm(
    MatmulCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
){
    using ArchTag = arch::AscendC910B3;
    // 开启pingpong机制
    using DispatchPolicy = gemm::GemmAscendC910B3Pingpong<true>;
    using AType = gemm::GemmType<int8_t, LayoutA>;
    using BType = gemm::GemmType<int8_t, LayoutB>;
    using CType = gemm::GemmType<int32_t, LayoutC>;
    // 使用Coord来传递值
    using L1TileShape = MatmulShape<256, 128, 256>;
    using L0TileShape = MatmulShape<256, 128, 64>;

    // 调用block层函数
    using GemmBlock = gemm::block::BlockGemm<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using GemmKernel = gemm::kernel::KernelGemm<GemmBlock>;
    typename GemmKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
    // 调用核函数
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "03_gemm/04_int8_rm_gemm m n k [device_id]";

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

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeC = lenC * sizeof(int32_t);

    layout::RowMajor layoutA{m, k};
    layout::RowMajor layoutB{k, n};
    layout::RowMajor layoutC{m, n};
    
    int8_t* hostA;
    ACL_CHECK(aclrtMallocHost((void**)(&hostA), sizeA));
    ReadFile("./data/input/A.bin", sizeA, hostA, sizeA);
    int8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    int8_t* hostB;
    ACL_CHECK(aclrtMallocHost((void**)(&hostB), sizeB));
    ReadFile("./data/input/B.bin", sizeB, hostB, sizeB);
    int8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    int32_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 获得当前核心数
    auto aicCoreNum = arch::AscendC910B3::MaxBlock;
    INT8Gemm<<<aicCoreNum, nullptr, stream>>>(
        options.problemShape,
        (uint8_t*)deviceA, layoutA,
        (uint8_t*)deviceB, layoutB,
        (uint8_t*)deviceC, layoutC);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    int32_t* hostC;
    ACL_CHECK(aclrtMallocHost((void**)(&hostC), sizeC));
    ACL_CHECK(aclrtMemcpy(hostC, sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./data/output/our_res.bin",hostC,sizeC);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostB));
    ACL_CHECK(aclrtFreeHost(hostC));

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