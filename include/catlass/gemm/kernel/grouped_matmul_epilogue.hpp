#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_EPILOGUE_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_EPILOGUE_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

namespace detail {

template <class T>
CATLASS_DEVICE void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len) {
  for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
    reinterpret_cast<uint64_t *>(dst)[i] =
        reinterpret_cast<__gm__ uint64_t *>(src)[i];
  }
}

} // namespace detail

// Template for grouped matmul add kernel. Compute grouped D = A * B + X
// X and D share the same memory pointer
// C (matmul result) is stored in workspace during computation
//
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GroupedMatmulEpilogue {
public:
  using BlockMmad = BlockMmad_;
  using ArchTag = typename BlockMmad::ArchTag;
  using L1TileShape = typename BlockMmad::L1TileShape;
  using ElementA = typename BlockMmad::ElementA;
  using LayoutA = typename BlockMmad::LayoutA;
  using ElementB = typename BlockMmad::ElementB;
  using LayoutB = typename BlockMmad::LayoutB;
  using ElementC = typename BlockMmad::ElementC;
  using LayoutC = typename BlockMmad::LayoutC;
  using ElementAccumulator = typename BlockMmad::ElementAccumulator;

  using BlockEpilogue = BlockEpilogue_;
  using LayoutD = LayoutC;

  using BlockScheduler = BlockScheduler_;
  static constexpr uint32_t MAX_TENSOR_COUNT = 256;

  /// Parameters structure
  struct Params {
    uint32_t problemCount;
    GM_ADDR ptrProblemShape;
    GM_ADDR ptrA;
    GM_ADDR ptrLayoutA;
    GM_ADDR ptrB;
    GM_ADDR ptrLayoutB;
    GM_ADDR ptrD;
    GM_ADDR ptrLayoutD;
    GM_ADDR ptrWorkspace;

    CATLASS_HOST_DEVICE
    Params() {}

    CATLASS_HOST_DEVICE
    Params(uint32_t problemCount_, GM_ADDR ptrProblemShape_, GM_ADDR ptrA_,
           GM_ADDR ptrLayoutA_, GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
           GM_ADDR ptrD_, GM_ADDR ptrLayoutD_, GM_ADDR ptrWorkspace_)
        : problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
          ptrA(ptrA_), ptrLayoutA(ptrLayoutA_), ptrB(ptrB_),
          ptrLayoutB(ptrLayoutB_), ptrD(ptrD_), ptrLayoutD(ptrLayoutD_),
          ptrWorkspace(ptrWorkspace_) {}
  };

  struct Arguments {
    uint32_t problemCount;
    uint8_t *ptrProblemShape;
    uint8_t *ptrA;
    uint8_t *ptrLayoutA;
    uint8_t *ptrB;
    uint8_t *ptrLayoutB;
    uint8_t *ptrD;
    uint8_t *ptrLayoutD;
  };

  static bool CanImplement(const Arguments &args) { return true; }
  static size_t GetWorkspaceSize(const Arguments &args) {
    // Workspace size should be provided by user
    return 0;
  }
  static Params ToUnderlyingArguments(const Arguments &args,
                                      uint8_t *workspace) {
    Params params{
        args.problemCount, args.ptrProblemShape, args.ptrA, args.ptrLayoutA,
        args.ptrB,         args.ptrLayoutB,      args.ptrD, args.ptrLayoutD,
        workspace};
    return params;
  }

  CATLASS_HOST_DEVICE
  GroupedMatmulEpilogue() {}
  CATLASS_HOST_DEVICE
  ~GroupedMatmulEpilogue() {}

  template <int32_t CORE_TYPE = g_coreType>
  CATLASS_DEVICE void operator()(Params const &params);

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params) {
    GemmCoord problemShapeList[MAX_TENSOR_COUNT];
    LayoutA layoutAList[MAX_TENSOR_COUNT];
    LayoutB layoutBList[MAX_TENSOR_COUNT];
    LayoutD layoutDList[MAX_TENSOR_COUNT];

    detail::UnpackListParam(problemShapeList, params.ptrProblemShape,
                            params.problemCount);
    detail::UnpackListParam(layoutAList, params.ptrLayoutA,
                            params.problemCount);
    detail::UnpackListParam(layoutBList, params.ptrLayoutB,
                            params.problemCount);
    detail::UnpackListParam(layoutDList, params.ptrLayoutD,
                            params.problemCount);

    BlockScheduler blockScheduler;
    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad(resource);

    AscendC::GlobalTensor<ElementA> gmA;
    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
    AscendC::GlobalTensor<ElementB> gmB;
    gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
    AscendC::GlobalTensor<ElementC> gmWorkspace;
    gmWorkspace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreNum = AscendC::GetBlockNum();
    int64_t inGroupOffsetA = 0;
    int64_t inGroupOffsetB = 0;
    int64_t inGroupOffsetD = 0;

    uint32_t startCoreIdx = 0;
    for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
      GemmCoord problemShape = problemShapeList[groupIdx];
      LayoutA layoutA = layoutAList[groupIdx];
      LayoutB layoutB = layoutBList[groupIdx];
      LayoutD layoutD = layoutDList[groupIdx];

      typename BlockEpilogue::Params epilogueParams{
          params.ptrD + inGroupOffsetD, layoutD, params.ptrD + inGroupOffsetD,
          layoutD};
      BlockEpilogue blockEpilogue(resource, epilogueParams);

      blockScheduler.Update(problemShape,
                            MakeCoord(L1TileShape::M, L1TileShape::N));
      uint32_t coreLoops = blockScheduler.GetCoreLoops();

      // Determine the starting loopIdx of the current core under the current
      // groupIdx
      uint32_t startLoopIdx;
      if (coreIdx < startCoreIdx) {
        startLoopIdx = coreIdx + coreNum - startCoreIdx;
      } else {
        startLoopIdx = coreIdx - startCoreIdx;
      }
      for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops;
           loopIdx += coreNum) {
        GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape =
            blockScheduler.GetActualBlockShape(blockCoord);

        MatrixCoord offsetA{blockCoord.m() * L1TileShape::M,
                            blockCoord.k() * L1TileShape::K};
        MatrixCoord offsetB{blockCoord.k() * L1TileShape::K,
                            blockCoord.n() * L1TileShape::N};
        MatrixCoord offsetC{blockCoord.m() * L1TileShape::M,
                            blockCoord.n() * L1TileShape::N};
        int64_t gmOffsetA = layoutA.GetOffset(offsetA);
        int64_t gmOffsetB = layoutB.GetOffset(offsetB);
        int64_t gmOffsetC = layoutD.GetOffset(offsetC);

        blockMmad(gmA[inGroupOffsetA + gmOffsetA], layoutA,
                  gmB[inGroupOffsetB + gmOffsetB], layoutB,
                  gmWorkspace[inGroupOffsetD + gmOffsetC], layoutD,
                  actualBlockShape);

        auto gmBlockC = gmWorkspace[inGroupOffsetD + gmOffsetC];
        auto layoutBlockC =
            layoutD.GetTileLayout(actualBlockShape.GetCoordMN());
        blockEpilogue(L1TileShape::ToCoord(), blockCoord, actualBlockShape,
                      gmBlockC, layoutBlockC);
      }

      inGroupOffsetA += problemShape.m() * problemShape.k();
      inGroupOffsetB += problemShape.k() * problemShape.n();
      inGroupOffsetD += problemShape.m() * problemShape.n();

      startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
    }

    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
      blockMmad.SynchronizeBlock();
    }

    AscendC::PipeBarrier<PIPE_ALL>();
  }

  template <>
  CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params) {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_EPILOGUE_HPP
