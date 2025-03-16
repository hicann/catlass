#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
#include "acot/gemm/block/block_gemm.hpp"
#include "acot/gemm/kernel/kernel_gemm.hpp"
#include "acot/matmul/matmul_type.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/matmul/dispatch_policy.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/tile/tile_elemwise_add.hpp"
#include "acot/epilogue/tile/tile_elemwise_muls.hpp"
#include "acot/epilogue/tile/tile_cast.hpp"
#include "acot/epilogue/block/block_epilogue.hpp"

using namespace acot;

using ScalarType = float;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACOT_GLOBAL
void Gemm(
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
    using ArchTag = arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using GemmBlockDispatchPolicy = matmul::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using AType = matmul::MatmulType<half, LayoutA>;
    using BType = matmul::MatmulType<half, LayoutB>;
    using CType = matmul::MatmulType<half, LayoutC>;
    using XType = matmul::MatmulType<half, LayoutC>;
    using L1TileShape = MatmulShape<128, 256, 256>;
    using L0TileShape = MatmulShape<128, 256, 64>;
    using TileShapeCast = MatrixShape<64, 256>;
    using GemmBlock = gemm::block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, XType>;
    using DType = CType;
    using ComputeType = XType;
    constexpr uint32_t computeLength = L1TileShape::MN / 2;
    using TileElemWiseAddGemm = epilogue::tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulGemm = epilogue::tile::TileElemWiseMul<ArchTag, ComputeType, computeLength>;
    using TileElemWistCastC = epilogue::tile::TileCast<ArchTag, ComputeType, CType, TileShapeCast>;
    using TileElemWistCastD = epilogue::tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, CType, XType, DType>;
    using EpilogueBlock = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulGemm, TileElemWistCastC, TileElemWistCastD, EpilogueTileCopy>;
    using GemmKernel = gemm::kernel::KernelGemmEpilogue<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmWorkspace, gmWA, layoutWA, gmWB, layoutWB, alpha, beta, gmC, gmC}; // 这里得修改 gmX保存A * B
    GemmKernel gemm;
    gemm(params);
}

typedef struct Options{
    const std::string HELPER = "03_gemm m n k [device_id]";

    MatmulCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

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
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if(argc == ARGS_MAX){
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
    size_t lenX = lenC; 
    size_t scalarLen = 1;
    
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
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(half);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(half);

    size_t scalarSize = scalarLen * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(scalarLen);
    std::vector<ScalarType> hostBeta(scalarLen);
    golden::FillRandomData(hostAlpha, -5.0f, 5.0f);
    golden::FillRandomData(hostBeta, -5.0f, 5.0f);
    std::vector<half> hostA(lenA);
    std::vector<half> hostB(lenB);
    std::vector<half> hostC(lenC);
    golden::FillRandomData(hostA, -5.0f, 5.0f);
    golden::FillRandomData(hostB, -5.0f, 5.0f);
    golden::FillRandomData(hostC, -5.0f, 5.0f);
    half *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    half *deviceWA{nullptr};
    if (IsSameStride(layoutWA, layoutA)) {
        deviceWA = deviceA;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    half *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    half *deviceWB{nullptr};
    if (IsSameStride(layoutWB, layoutB)) {
        deviceWB = deviceB;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    half *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
    half *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    Gemm<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        hostAlpha[0], hostBeta[0],
        options.problemShape,
        (uint8_t*)deviceA, layoutA,
        (uint8_t*)deviceB, layoutB,
        (uint8_t*)deviceC, layoutC,
        (uint8_t*)deviceWA, layoutWA,
        (uint8_t*)deviceWB, layoutWB,
        (uint8_t*)gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    
    std::vector<half> hostRes(lenC);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> hostGolden(lenC);
    golden::ComputeGemm(options.problemShape, hostAlpha[0], hostBeta[0], hostA, layoutA, hostB, layoutB, hostC, layoutC, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m * n * k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (!IsSameStride(layoutWA, layoutA)) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (!IsSameStride(layoutWB, layoutB)) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    ACL_CHECK(aclrtFree(gmWorkspace));

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