#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif

using namespace Catlass;

static uint32_t getSplitNum(bool trans, uint32_t M, uint32_t N, uint32_t M1, uint32_t N1, uint32_t maxSplict) {
    uint32_t CORENUM = 20;
    uint32_t splitNum = 1;
    uint32_t maxOccupancy = 0;
    uint32_t blockNum = (M - 1) / M1 + 1;
    if (!trans) {
        splitNum = 1;
    } else {
        uint32_t splitNum1 = 1, splitNum2 = 1;
        for (uint32_t i = 1; i <= maxSplict; i += 1) {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy) {
                maxOccupancy = occupancy;
                splitNum1 = i;
            }
        }
        maxOccupancy = 0;
        for (uint32_t i = 1; i <= maxSplict; i <<= 1) {
            uint32_t occupancy = (i * blockNum) % (CORENUM * 2);
            if (!occupancy)
                occupancy = (CORENUM * 2);
            if (occupancy > maxOccupancy) {
                maxOccupancy = occupancy;
                splitNum2 = i;
            }
        }
        splitNum = (splitNum1 - splitNum2) > 4 ? splitNum1 : splitNum2;
    }
    return splitNum;
}

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutX = layout::VectorLayout;
using LayoutY = layout::VectorLayout;
using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::GemvAtlasA2;
using UBTileShape = GemvShape<32, 512>;
using AType = Gemm::GemmType<float, LayoutA>;
using XType = Gemm::GemmType<float, LayoutX>;
using YType = Gemm::GemmType<float, LayoutY>;
using BiasType = void;
using TileCopy = Gemv::Tile::TileCopyGemvAiv<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
using TileVmad = Gemv::Tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
using TileVmuls = Gemv::Tile::TileVmuls<typename DispatchPolicy::ArchTag, XType>;
using GemvBlock = Gemv::Block::BlockGemv<DispatchPolicy, UBTileShape, AType, XType, YType, BiasType, TileCopy, TileVmad, TileVmuls>;
using GemvKernel = Gemv::Kernel::KernelGemvAiv<GemvBlock, void>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GemmParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    // Aligned with example: gemv_aiv uses trans=false, split is always 1
    uint32_t maxSplict = 20;
    uint32_t const split = getSplitNum(false, m, n, UBTileShape::M, UBTileShape::N, maxSplict);

    // Arguments order: {shape, A, X, Y_read, Z_write, alpha, beta, split}
    // inputAddr[2] = Y (read), outputAddr[0] = output (write)
    typename GemvKernel::Arguments arguments{
        GemvCoord{m, n}, params->inputAddr[0], params->inputAddr[1],
        params->inputAddr[2], params->outputAddr[0], params->alpha, params->beta, split};

    Catlass::RunKernel<GemvKernel>(arguments, stream, blockNum);
}
