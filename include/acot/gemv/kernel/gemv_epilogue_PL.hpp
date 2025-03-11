#ifndef ACOT_GEMV_KERNLE_GEMV_EPILOGUE_HPP
#define ACOT_GEMV_KERNLE_GEMV_EPILOGUE_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/arch/cross_core_sync.hpp"
#include "acot/gemv_coord.hpp"
#include "acot/matrix_coord.hpp"

namespace acot::gemv::kernel
{

    // tmeplate for gemv kernle, Compute z = αAx + βy
    template <
        class BlockGemv_,
        class BlockEpilogue_,
        class TileScheduler_>
    class GemvEpilogue
    {
    public:
        using BlockGemv = BlockGemv_;
        using ArchTag = typename BlockGemv::ArchTag;
        using L1TileShape = typename BlockGemv::L1TileShape;
        using L0TileShape = typename BlockGemv::L0TileShape;

        using Elementx = typename BlockGemv::Elementx;
        using Layoutx = typename BlockGemv::Layoutx;

        using ElementA = typename BlockGemv::ElementA;
        using LayoutA = typename BlockGemv::LayoutA;
        using Elementy = typename BlockGemv::Elementy;
        using Layouty = typename BlockGemv::Layouty;

        using BlockEpilogue = BlockEpilogue_;
        using Elementz = typename BlockEpilogue::ElementZ;
        using Layoutz = typename BlockEpilogue::LayoutZ;
        using EpilogueParams = typename BlockEpilogue::Params;

        // 计算时，L0C的数据类型
        using ElementAccumulator = typename gemv::helper::ElementAccumulatorSelector<ElementA, Elementx>::ElementAccumulator;

        using TileScheduler = TileScheduler_;

        // static_assert(std::is_same_v<typename BlockEpilogue::ElementY, Elementy> &&
        //                   std::is_same_v<typename BlockEpilogue::LayoutY, Layouty>,
        //               "The yType of Gemv and Epilogue should be consistent.");

        /// Parameters structure
        struct Params
        {
            // Data members
            GemvCoord problemShape;
            GM_ADDR ptrX;
            Layoutx layoutX;
            GM_ADDR ptrA;
            LayoutA layoutA;
            GM_ADDR ptrWorkspace;
            EpilogueParams epilogueParams;

            // Methods
            ACOT_DEVICE
            Params() {}

            ACOT_DEVICE
            Params(
                GemvCoord const &problemShape_,
                GM_ADDR ptrX_, Layoutx const &layoutX_,
                GM_ADDR ptrA_, LayoutA const &layoutA_,
                GM_ADDR ptrWorkspace_, EpilogueParams const &epilogueParams_) : problemShape(problemShape_), ptrX(ptrX_), layoutX(layoutX_), ptrA(ptrA_), layoutA(layoutA_),
                                                                                ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
        };

        // Methods
        ACOT_DEVICE
        GemvEpilogue() {}

        template <int32_t CORE_TYPE = g_coreType>
        ACOT_DEVICE void operator()(Params const &params);

        template <>
        ACOT_DEVICE void operator()<AscendC::AIC>(Params const &params)
        {
            TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            // uint32_t coreLoops = matmulTileScheduler.GetCoreLoops();

            arch::Resource<ArchTag> resource;
            BlockGemv blockGemv(resource);

            // Represent the full gm
            AscendC::GlobalTensor<Elementx> gmx;
            gmx.SetGlobalBuffer((__gm__ Elementx *)params.ptrX);
            AscendC::GlobalTensor<ElementA> gmA;

            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            AscendC::GlobalTensor<Elementy> gmy;
            gmy.SetGlobalBuffer((__gm__ Elementy *)params.ptrWorkspace);

            // layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
            layout::RowMajor layouty(1, params.problemShape.m());

            uint32_t maxMPerBlock = L1TileShape::M;
            uint32_t maxNPerBlock = L1TileShape::N;
            uint32_t M = params.problemShape.m();
            uint32_t N = params.problemShape.n();

            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t coreLoops = MLoops;
            uint32_t singleIdx = 0;

            static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
            static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
            static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);

// 初始化核间流水
#pragma unroll
            for (uint32_t i = 0; i < L0C_TILE_NUM; i++)
            {
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)i);
            }

            for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum())
            {
                // Compute Block location
                uint32_t MGmBlockIdx = loopIdx;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = N;
                int64_t gmOffsetx;
                int64_t gmOffsetA;
                int64_t gmOffsety;
                int64_t gmOffsetNextx;
                int64_t gmOffsetNextA;
                int64_t gmOffsetNexty;

                // 计算A，x，y的当前块的偏移量
                if constexpr (std::is_same<LayoutA, acot::layout::RowMajor>::value) // 行优先情况
                {
                    gmOffsetx = 0;
                    gmOffsetA = MGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);
                    gmOffsety = MGmBlockIdx * maxMPerBlock;
                }
                else // 列优先情况
                {
                    gmOffsetx = 0;
                    gmOffsetA = MGmBlockIdx * maxMPerBlock;
                    gmOffsety = MGmBlockIdx * maxMPerBlock;
                }

                bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
                bool hasNextBlock = false;
                uint32_t MNextGmBlockIdx;                         // 下一个块的M方向的偏移
                GemvCoord nextActualBlockShape;                   // 下一个块的实际shape
                if (loopIdx + AscendC::GetBlockNum() < coreLoops) // 预加载下一块
                {
                    hasNextBlock = true;
                    uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                    MNextGmBlockIdx = nextLoopIdx;
                    uint32_t MNextGmActual = (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                    uint32_t NNextGmActual = N;
                    nextActualBlockShape = GemvCoord(MNextGmActual, NNextGmActual);
                }
                // 计算A，x，y的下一块的偏移量
                if constexpr (std::is_same<LayoutA, acot::layout::RowMajor>::value) // 行优先情况
                {
                    gmOffsetNextx = 0;
                    gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock * params.layoutA.stride(0);
                    gmOffsetNexty = MNextGmBlockIdx * maxMPerBlock;
                }
                else // 列优先情况
                {
                    gmOffsetNextx = 0;
                    gmOffsetNextA = MNextGmBlockIdx * maxMPerBlock;
                    gmOffsetNexty = MNextGmBlockIdx * maxMPerBlock;
                }

                GemvCoord actualBlockShape = GemvCoord(MGmActual, NGmActual); // 当前块的实际shape

                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

                // Compute block-scoped matrix multiply-add
                blockGemv(gmx[gmOffsetx], params.layoutX,
                          gmA[gmOffsetA], params.layoutA,
                          gmy[gmOffsety], layouty,
                          gmx[gmOffsetNextx],
                          gmA[gmOffsetNextA],
                          actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock,
                          singleIdx);

                // AscendC::PipeBarrier<PIPE_ALL>();

                arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)singleIdx % L0C_TILE_NUM);

                singleIdx++;
            }

#pragma unroll
            for (uint32_t i = 0; i < L0C_TILE_NUM; i++)
            {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)i);
            }
        }

        template <>
        ACOT_DEVICE void operator()<AscendC::AIV>(Params const &params)
        {
            TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            // uint32_t coreLoops = matmulTileScheduler.GetCoreLoops();

            BlockEpilogue blockEpilogue(resource, params.epilogueParams);

            // Represent the full gm
            AscendC::GlobalTensor<Elementy> gmy;
            gmy.SetGlobalBuffer((__gm__ Elementy *)params.ptrWorkspace);
            // layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
            layout::RowMajor layouty(1, params.problemShape.m());

            // Get aicore information 获取核idx，核数，子核idx
            uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); // 0-19
            uint32_t aicoreNum = AscendC::GetBlockNum();                               // 20
            uint32_t subcoreIndex = AscendC::GetSubBlockIdx();
            // 扩大空间的尝试
            // GemvShape Ubshape<1024, 1>;

            // 计算总共需要的循环次数
            uint32_t maxMPerBlock = L1TileShape::M;
            uint32_t maxNPerBlock = L1TileShape::N;
            uint32_t M = params.problemShape.m();
            uint32_t N = params.problemShape.n();
            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t coreLoops = MLoops;

            // Loop through the epilogue calculations of each basic block
            // GemvCoord blockShape = L1TileShape::ToCoord(); // blockshape就是l1中矩阵的大小
            GemvCoord blockShape{L1TileShape::N, L1TileShape::M};

            for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum)
            {
                // Compute block location
                // GemvCoord blockCoord = matmulTileScheduler.GetBlockCoord(loopIdx);
                // MatmulCoord actualBlockShape = matmulTileScheduler.GetActualBlockShape(blockCoord);
                // GemvCoord blockCoord = GemvCoord(loopIdx, 0);
                GemvCoord blockCoord = GemvCoord(0, loopIdx);
                uint32_t MGmActual = (loopIdx == coreLoops) ? M - loopIdx * maxMPerBlock : maxMPerBlock;
                uint32_t NGmActual = 1;

                GemvCoord actualBlockShape = GemvCoord(NGmActual, MGmActual);

                // Get the offset
                MatrixCoord blockOffset = blockCoord.GetCoordMN() * blockShape.GetCoordMN();

                // Get the data and layout of y under the current basic block
                auto gmBlocky = gmy[layouty.GetOffset(blockOffset)];
                auto layoutBlocky = layouty.GetTileLayout(actualBlockShape.GetCoordMN());

                // Synchronize cross core
                arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

                // Actual calculatioin logic for performing block-scoped epilogue
                // blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlocky, layoutBlocky);
                // Actual calculatioin logic for performing block-scoped epilogue
                blockEpilogue(blockOffset, actualBlockShape, gmBlocky, layoutBlocky);
            }
        }

    private:
        // ID used for inter-core synchronization
        static constexpr arch::FlagID FLAG_AIC_FINISH_STORE = 0;
        static constexpr arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
        arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
        arch::Resource<ArchTag> resource;
    };

} // namespace acot::gemv::kernel

#endif // ACOT_GEMV_KERNLE_GEMV_EPILOGUE_HPP
