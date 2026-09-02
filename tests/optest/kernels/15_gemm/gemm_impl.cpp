#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/gemm.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"
#include "common/workspace_alloc.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;
using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutX = layout::CATLASS_JIT_LAYOUT_C;
using ArchTag = Arch::AtlasA2;
using GemmBlockDP = Gemm::GemmAtlasA2<true, true, true>;
using EpiBlockDP = Epilogue::EpilogueAtlasA2Gemm;
using BaseL1 = GemmShape<128, 128, 128>;
using BaseL0 = GemmShape<128, 128, 64>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, float, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, float, BaseL0>::type;
using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutX>;
using XType = Gemm::GemmType<ElementC, LayoutX>;
using DType = XType;
using ComputeType = CType;
using GemmBlock = Gemm::Block::BlockGemm<GemmBlockDP, L1TileShape, L0TileShape, AType, BType, CType>;
constexpr uint32_t cL = L1TileShape::MN / 2;
using TileAdd = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, cL>;
using TileMul = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, cL>;
using TileCast = Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
using TileCp = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
using EpiBlock = Epilogue::Block::BlockEpilogue<EpiBlockDP, CType, XType, DType, TileAdd, TileMul, TileCast, TileCp>;
using Kernel = Gemm::Kernel::KernelGemm<GemmBlock, EpiBlock>;

// ----------------------------------------------------------------------
// Non-aligned shape workspace helpers (mimicking gemm.cpp L69-96)
// ----------------------------------------------------------------------

inline layout::RowMajor GetWorkspaceLayout(const layout::RowMajor& layout, uint32_t align) {
    if (align == 0) return layout;
    return layout::RowMajor(layout.shape(0), layout.shape(1), RoundUp(layout.shape(1), align));
}

inline layout::ColumnMajor GetWorkspaceLayout(const layout::ColumnMajor& layout, uint32_t align) {
    if (align == 0) return layout;
    return layout::ColumnMajor(layout.shape(0), layout.shape(1), RoundUp(layout.shape(0), align));
}

inline size_t GetWorkspaceLen(const layout::RowMajor& layout) {
    return layout.shape(0) * layout.stride(0);
}

inline size_t GetWorkspaceLen(const layout::ColumnMajor& layout) {
    return layout.shape(1) * layout.stride(1);
}

inline bool IsSameStride(const layout::RowMajor& l1, const layout::RowMajor& l2) {
    return l1.stride(0) == l2.stride(0);
}

inline bool IsSameStride(const layout::ColumnMajor& l1, const layout::ColumnMajor& l2) {
    return l1.stride(1) == l2.stride(1);
}

// ----------------------------------------------------------------------

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GemmParams* params)
{
    uint32_t m = params->m, n = params->n, k = params->k;
    float alpha = params->alpha, beta = params->beta;

    const uint32_t align = 128;  // matching gemm.cpp L126

    // Create layouts (matching gemm.cpp L130-132)
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutX layoutX{m, n};

    // Create workspace layouts with aligned strides
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);

    // Compute workspace sizes
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(ElementA);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(ElementB);

    // Evaluate whether need workspace alloc
    const bool needWA = !IsSameStride(layoutWA, layoutA);
    const bool needWB = !IsSameStride(layoutWB, layoutB);

    // Allocate workspace buffers for non-aligned strides
    // (matching gemm.cpp L147-164: deviceWA / deviceWB)
    uint8_t* deviceWA = params->inputAddr[0];
    uint8_t* deviceWB = params->inputAddr[1];
    if (needWA) {
        deviceWA = g_catlassWorkspaceAlloc(sizeWA);
    }
    if (needWB) {
        deviceWB = g_catlassWorkspaceAlloc(sizeWB);
    }

    // Matrix C: inputAddr[2] holds the residual matrix C data.
    // Copy C into outputAddr[0] (deviceX) to initialize the epilogue input.
    // Then use outputAddr[0] as both epilogue X input and output destination.
    uint8_t* deviceX = params->outputAddr[0];
    size_t sizeX = (size_t)m * n * sizeof(ElementC);
    aclrtMemcpy(deviceX, sizeX, params->inputAddr[2], sizeX, ACL_MEMCPY_DEVICE_TO_DEVICE);

    // Epilogue: D = alpha * (A*B) + beta * X
    typename EpiBlock::Params epilogueParams{alpha, beta, deviceX, layoutX, deviceX, layoutX};

    uint8_t* deviceWorkspace = g_catlassWorkspaceAlloc(sizeX);

    typename Kernel::Arguments args{GemmCoord{m,n,k}, align,
        params->inputAddr[0], params->inputAddr[1],
        deviceWorkspace, deviceWA, deviceWB, epilogueParams};

    Catlass::RunKernel<Kernel>(args, stream, blockNum);

    aclError error = aclrtSynchronizeStream(stream);
    if (error == ACL_ERROR_NONE) { // mem free if needed
        if (needWA) {
            g_catlassWorkspaceFree(deviceWA, sizeWA);
        }
        if (needWB) {
            g_catlassWorkspaceFree(deviceWB, sizeWB);
        }

        g_catlassWorkspaceFree(deviceWorkspace, sizeX);
    }
}
