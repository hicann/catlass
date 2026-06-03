#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/fp8_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common/workspace_alloc.h"

using namespace Catlass;

using ArchTag = Arch::AtlasA2;

using ElementA = half;
using ElementPrologueA = int8_t;
using ElementB = half;
using ElementPrologueB = int8_t;
using ElementC = float;

using LayoutA = layout::RowMajor;
using LayoutPrologueA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutPrologueB = layout::RowMajor;
using LayoutC = layout::RowMajor;

using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;

constexpr uint32_t mScalar = 2;
constexpr uint32_t nScalar = 2;
constexpr uint32_t splitkLength = 1024;

constexpr bool ENABLE_UNIT_FLAG = true;
using DispatchPolicy = Gemm::MmadAtlasA2PingpongSliceKWithPrologue<ENABLE_UNIT_FLAG>;

using PrologueSrcTypeA = Gemm::GemmType<ElementPrologueA, LayoutPrologueA>;
using PrologueDstTypeA = Gemm::GemmType<ElementA, LayoutA>;
using PrologueSrcTypeB = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
using PrologueDstTypeB = Gemm::GemmType<ElementB, LayoutB>;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

constexpr uint32_t COMPUTE_LENGTH_A = 16 * 1024 / sizeof(int8_t);
constexpr uint32_t COMPUTE_LENGTH_B = 16 * 1024 / sizeof(int8_t);
using PrologueA = Gemm::Tile::TileCastFp8ToFp16Dequant<ArchTag, PrologueSrcTypeA, PrologueDstTypeA, COMPUTE_LENGTH_A>;
using PrologueB = Gemm::Tile::TileCastFp8ToFp16Dequant<ArchTag, PrologueSrcTypeB, PrologueDstTypeB, COMPUTE_LENGTH_B>;
using TileCopy = Gemm::Tile::TileCopyWithPrologue<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

using MatmulKernel = Gemm::Kernel::FP8Matmul<BlockMmadOpt, BlockEpilogue, BlockScheduler, mScalar, nScalar, splitkLength>;

__global__ __aicore__ void fp8_matmul_entry(MatmulKernel::Params params)
{
    MatmulKernel op;
    op(params);
}

extern "C" int rtGetC2cCtrlAddr(uint64_t*, uint32_t*);

extern "C" void A2Fp8E4M3Matmul(
    const uint32_t blockNum, aclrtStream stream, const CatlassKernel::TParams& tParams,
    const CatlassKernel::MatmulParams& params)
{
    uint32_t m = params.m;
    uint32_t n = params.n;
    uint32_t k = params.k;

    uint8_t* deviceA = params.inputAddr[0];
    uint8_t* deviceB = params.inputAddr[1];
    uint8_t* deviceC = params.outputAddr[0];

    half scalar = 1.0;
    half zeroPoint = 0.0;

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k}, blockNum, deviceA, deviceB, deviceC,
        scalar, zeroPoint};

    size_t wsSize = MatmulAdapter::GetWorkspaceSize(arguments);
    uint8_t* ws = nullptr;
    if (wsSize > 0) {
        if (g_catlassWorkspaceAlloc) {
            ws = g_catlassWorkspaceAlloc(wsSize);
        } else {
            aclrtMalloc(reinterpret_cast<void**>(&ws), wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
    }
    MatmulAdapter matmulOp;
    matmulOp.Initialize(arguments, ws);
    matmulOp.Run(stream, blockNum, fftsAddr);
    aclrtSynchronizeStream(stream);
    if (ws && !g_catlassWorkspaceAlloc) {
        aclrtFree(ws);
    }
}
