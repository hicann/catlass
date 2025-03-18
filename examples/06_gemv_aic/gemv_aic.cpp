#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"

#include "acot/acot.hpp"
#include "acot/arch/arch.hpp"
// #include "acot/gemv/block/block_gemv_aic.hpp"
#include "acot/gemv/block/block_gemv.hpp"
#include "acot/gemv/block/block_swizzle.hpp"

#include "acot/gemv/kernel/kernel_gemv_aic.hpp"

#include "acot/matmul/dispatch_policy.hpp"
#include "acot/matmul/matmul_type.hpp"

#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/epilogue/tile/tile_elemwise_add.hpp"
#include "acot/epilogue/tile/tile_elemwise_muls.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/block/block_epilogue.hpp"

#include "acot/layout/layout.hpp"

using namespace acot;

using ScalarType = float;

template <
    class LayoutA,
    class LayoutX,
    class LayoutTemp>
ACOT_GLOBAL void GemvAic(
    uint64_t fftsAddr,
    ScalarType alpha, ScalarType beta,
    GemvCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmX, LayoutX layoutX,
    GM_ADDR gmZ, LayoutTemp layoutZ,
    GM_ADDR gmWorkspace)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = arch::AtlasA2;

    // Block level, define BlockGemv
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = matmul::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using L1TileShape = GemvShape<32, 512>;
    using L0TileShape = GemvShape<32, 256>;
    using AType = matmul::MatmulType<float, LayoutA>;
    using XType = matmul::MatmulType<float, LayoutX>;
    using TempType = matmul::MatmulType<float, LayoutTemp>;
    using BiasType = void;
    using TileCopy = gemv::tile::TileCopy<typename DispatchPolicy::ArchTag, AType, XType, TempType, BiasType>;
    using TileMmad = gemv::tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;

    using BlockGemv = gemv::block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, AType, XType, TempType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using YType = matmul::MatmulType<float, LayoutTemp>;
    using ZType = matmul::MatmulType<float, LayoutTemp>;
    using ComputeType = TempType;
    constexpr uint32_t computeLength = 8192;

    using TileElemWiseAddGemv = epilogue::tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulGemv = epilogue::tile::TileElemWiseMul<ArchTag, ComputeType, computeLength>;

    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, YType, TempType, ZType>;

    using BlockEpilogue = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, TempType, YType, ZType, TileElemWiseAddGemv, TileElemWiseMulGemv, EpilogueTileCopy>;

    using TileScheduler = typename gemv::block::GemvIdentityBlockSwizzle<3, 0>; // 暂时未使用

    // kernle levels
    using GemvKernel = gemv::kernel::GemvEpilogue<BlockGemv, BlockEpilogue, TileScheduler>;

    // Prepare params
    typename BlockEpilogue::Params epilogueParams{alpha, beta, gmZ, layoutZ, gmZ, layoutZ};
    typename GemvKernel::Params params{problemShape, gmX, layoutX, gmA, layoutA, gmWorkspace, epilogueParams};

    // call a kernel
    GemvKernel gemv;
    gemv(params);
}

typedef struct Options
{
    const std::string HELPER = "06_gemv_aic m n [device_id]";

    GemvCoord problemShape{128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc < N_INDEX)
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
} Options;

layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
{
    if (align == 0)
    {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
                            RoundUp(layout.shape(1), align));
}

layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0)
    {
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

void Run(Options options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenY = static_cast<size_t>(m) * 1;
    size_t lenZ = static_cast<size_t>(m) * 1;
    size_t scalarLen = 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeZ = lenZ * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    size_t sizeWorkspace = lenZ * sizeof(float);

    using LayoutX = layout::RowMajor;
    using LayoutA = layout::RowMajor;
    using LayoutZ = layout::RowMajor;

    LayoutX layoutX{1, n};
    LayoutA layoutA{m, n};
    LayoutZ layoutZ{1, m};

    LayoutZ layoutY_r{m, 1};
    LayoutX layoutX_r{n, 1};

    size_t scalarSize = scalarLen * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(scalarLen);
    std::vector<ScalarType> hostBeta(scalarLen);
    golden::FillRandomData(hostAlpha, -1.0f, 1.0f);
    golden::FillRandomData(hostBeta, -1.0f, 1.0f);

    std::vector<float> hostA(lenA);
    std::vector<float> hostX(lenX);
    std::vector<float> hostY(lenY); // 输入
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    golden::FillRandomData(hostX, -1.0f, 1.0f);
    golden::FillRandomData(hostY, -1.0f, 1.0f);
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceZ), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeZ, hostY.data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    GemvAic<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        hostAlpha[0], hostBeta[0],
        options.problemShape,
        deviceA, layoutA,
        deviceX, layoutX,
        deviceZ, layoutZ,
        deviceWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostRes(lenZ);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenZ);

    golden::ComputeGemv(options.problemShape, hostAlpha[0], hostBeta[0], hostA, layoutA, hostX, layoutX_r, hostY, layoutY_r, hostGolden, layoutY_r);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);

    if (errorIndices.empty())
    {
        std::cout << "Compare success." << std::endl;
    }
    else
    {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceZ));
    ACL_CHECK(aclrtFree(deviceWorkspace));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0)
    {
        return -1;
    }
    Run(options);
    return 0;
}