#ifndef ACOT_GEMV_BLOCK_BLOCK_GEMV_PINGPONG_PL_HPP
#define ACOT_GEMV_BLOCK_BLOCK_GEMV_PINGPONG_PL_HPP

#include "acot/acot.hpp"
#include "acot/gemv/dispatch_policy.hpp"
#include "acot/arch/resource.hpp"
#include "acot/coord.hpp"
// #include "acot/matmul_coord.hpp"

#include "acot/gemv_coord.hpp"

#include "acot/gemv/dispatch_policy.hpp"
#include "acot/gemv/helper.hpp"

namespace acot::gemv::block
{

    template <
        bool ENABLE_UNIT_FLAG_,
        bool ENABLE_SHUFFLE_K_,
        class L1TileShape_,
        class L0TileShape_,
        class xType_, // 向量x
        class AType_, // 里面有成员elemenet, layout
        class yType_, // 向量y
        class BiasType_,
        class TileCopy_,
        class TileMmad_> // BiasType, TileCopy, TileMmad这三个参数采用默认值，在block_gemv.hpp文件中
    struct BlockGemv<
        // GemvAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
        GemvAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
        L1TileShape_,
        L0TileShape_,
        xType_,
        AType_,
        yType_,
        BiasType_,
        TileCopy_,
        TileMmad_>
    {
    public:
        // Type Aliases
        // using DispatchPolicy = GemvAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
        using DispatchPolicy = GemvAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
        using ArchTag = typename DispatchPolicy::ArchTag;

        using L1TileShape = L1TileShape_;
        using L0TileShape = L0TileShape_;

        // 矩阵、向量数据数据类型、排布方式
        using ElementA = typename AType_::Element;
        using LayoutA = typename AType_::Layout;
        using Elementx = typename xType_::Element;
        using Layoutx = typename xType_::Layout;
        using Elementy = typename yType_::Element;
        using Layouty = typename yType_::Layout;

        // tile层搬运函数
        using TileMmad = TileMmad_;
        using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
        using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
        using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
        using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;

        // 计算时，L0C的数据类型
        using ElementAccumulator = typename gemv::helper::ElementAccumulatorSelector<ElementA, Elementx>::ElementAccumulator;

        using LayoutxInL1 = typename CopyL1ToL0A::LayoutSrc;
        using LayoutAInL1 = typename CopyL1ToL0B::LayoutSrc;
        using LayoutxInL0 = typename CopyL1ToL0A::LayoutDst;
        using LayoutAInL0 = typename CopyL1ToL0B::LayoutDst;
        using LayoutyInL0 = layout::zN; // 这里应该是默认

        // x和A的对齐方式不同
        using L1AAlignHelper = gemv::helper::L1AlignHelper<ElementA, LayoutA>;
        using L1XAlignHelper = gemv::helper::L1AlignHelper<Elementx, Layoutx>;

        static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG; // 使能单元标志
        static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K; // ShuffleK开启标志
        static constexpr uint32_t STAGES = DispatchPolicy::STAGES;                 // 双流水

        static constexpr uint32_t L1A_SIZE = ArchTag::L1_SIZE / 2 / STAGES;
        static constexpr uint32_t L1B_SIZE = ArchTag::L1_SIZE / 2 / STAGES;
        static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
        static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

        static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementAccumulator);
        static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE;

        // 默认L0A,L0B划分出两片空间，单位：字节
        static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
        static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

        // Check L1TileShape
        static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

        static constexpr uint32_t L0A_TILE_SIZE = 16 * L0TileShape::N * sizeof(Elementx);
        static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementA);
        static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
        static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");

        /// Construct
        ACOT_DEVICE
        BlockGemv(arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
        {
            uint32_t l1AOffset = l1BufAddrStart;
            uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++)
            {
                // Assign L1/L0A/L0B space for each stages 为双缓冲分配空间
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<Elementx>(l1AOffset + L1A_SIZE * i);
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BOffset + L1B_SIZE * i);
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<Elementx>(L0A_PINGPONG_BUF_SIZE * i);
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(L0B_PINGPONG_BUF_SIZE * i);

                // todo:分配流水。先用pipeall
                l1AEventList[i] = i;          // 0, 1
                l1BEventList[i] = i + STAGES; // 2, 3
                l0AEventList[i] = i;          // 0, 1
                l0BEventList[i] = i + STAGES; // 2, 3

                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

            // todo:分配流水。先用pipeall
            // AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        /// Destructor
        ACOT_DEVICE
        ~BlockGemv()
        {
            // todo:释放流水。先用pipeall
            for (uint32_t i = 0; i < STAGES; i++)
            {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            // AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        /// Perform a block-scoped vector-matrix multiply-accumulate
        ACOT_DEVICE
        void operator()(
            AscendC::GlobalTensor<Elementx> const &gmBlockx, Layoutx const &layoutx,
            AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
            AscendC::GlobalTensor<Elementy> const &gmBlocky, Layouty const &layouty,
            AscendC::GlobalTensor<Elementx> const &gmNextBlockx,
            AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
            GemvCoord const &actualShape, GemvCoord const &actualShapeNext,
            bool isFirstBlock, bool hasNextBlock,
            uint32_t singleIdx)
        {
            auto layoutxInL1 = LayoutxInL1::template MakeLayout<Elementx>(L1XAlignHelper::M_ALIGNED, L1TileShape::N); // 16, N
            auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::N);            // M,N 行优先，列优先也是这个
            auto layoutInL0C = LayoutyInL0::MakeLayoutInL0C(MatrixCoord(L1XAlignHelper::M_ALIGNED, actualShape.m())); // 16, M

            // uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m()); // m方向要和16或者32对齐，判断条件具体和数据类型有关

            uint32_t nTileCount = CeilDiv<L1TileShape::N>(actualShape.n()); // 实际上L1TileShape::N是maxNPerBlock, actualShape.n()是传入的n，在单核外没做切分

            uint32_t startTileIdx = 0;
            if constexpr (ENABLE_SHUFFLE_K)
            {
                startTileIdx = AscendC::GetBlockIdx(); // 对于不同的AIC，读取的起始矩阵块不同
            }
            uint32_t firstTileIdx = startTileIdx % nTileCount;
            uint32_t lastTileIdx = (startTileIdx + nTileCount - 1) % nTileCount;
            uint32_t nActual = min(actualShape.n(), L1TileShape::N);
            uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nActual);

            // main loop, GM->L1, N方向做切分
            for (uint32_t nLoopIdx = 0; nLoopIdx < nTileCount; nLoopIdx++)
            {
                uint32_t shuffleKIdx = (startTileIdx + nLoopIdx) % nTileCount;
                if (shuffleKIdx == firstTileIdx && isFirstBlock)
                {
                    MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::N};
                    MatrixCoord gmTilexOffset{0, shuffleKIdx * L1TileShape::N};
                    auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                    auto gmTilex = gmBlockx[layoutx.GetOffset(gmTilexOffset)];

                    // load first vector x tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    auto layoutTilex = layoutx.GetTileLayout(MakeCoord(uint32_t(1), nRound));
                    copyGmToL1A(l1ATensorList[l1ListId], gmTilex, layoutxInL1, layoutTilex);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                    // load first matrix A tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nActual));
                    copyGmToL1B(l1BTensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                }

                uint32_t l1ListIdNext = (l1ListId + 1) % STAGES;
                uint32_t nActualNext{0};
                uint32_t nRoundNext{0};
                // uint32_t nActual = (nLoopIdx < nTileCount - 1) ? L1TileShape::N : (actualShape.n() - nLoopIdx * L1TileShape::N);

                uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nActual);

                // preload next tile from GM to L1
                if (shuffleKIdx != lastTileIdx)
                {
                    uint32_t shuffleKIdxNext = (shuffleKIdx + 1) % nTileCount;
                    // uint32_t shuffleKIdxNext = (startTileIdx + nLoopIdx + 1) % nTileCount;
                    nActualNext = (shuffleKIdxNext < nTileCount - 1) ? L1TileShape::N : (actualShape.n() - shuffleKIdxNext * L1TileShape::N);
                    // 需要对齐，因为在列优先中，矩阵A的列方向默认对齐16，向量x的列方向默认对齐32B/sizeof。因此不同数据之间会存在对不齐的情况
                    nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(nActualNext);

                    // Get L1 tensor
                    auto l1ATensor = l1ATensorList[l1ListIdNext];
                    auto l1BTensor = l1BTensorList[l1ListIdNext];

                    // Get GM tile 算偏移量
                    MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::N};
                    MatrixCoord gmTilexOffset{0, shuffleKIdxNext * L1TileShape::N};

                    auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                    auto gmTilex = gmBlockx[layoutx.GetOffset(gmTilexOffset)];

                    // load next vector x tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    auto layoutTilex = layoutx.GetTileLayout(MakeCoord(uint32_t(1), nRoundNext));
                    copyGmToL1A(l1ATensor, gmTilex, layoutxInL1, layoutTilex);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                    // load next Matrix A tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nActualNext));
                    copyGmToL1B(l1BTensor, gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
                if (shuffleKIdx == lastTileIdx && hasNextBlock)
                {
                    // Get L1 tensor for next stage
                    auto l1ATensor = l1ATensorList[l1ListIdNext];
                    auto l1BTensor = l1BTensorList[l1ListIdNext];
                    // Get GM tensor for next stage
                    nActualNext = min(actualShapeNext.n(), L1TileShape::N);
                    nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(nActualNext);

                    MatrixCoord gmTileAOffset{0, firstTileIdx * L1TileShape::N};
                    MatrixCoord gmTilexOffset{0, firstTileIdx * L1TileShape::N};
                    auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                    auto gmTilex = gmBlockx[layoutx.GetOffset(gmTilexOffset)];

                    // load next vector x tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    auto layoutTilex = layoutx.GetTileLayout(MakeCoord(uint32_t(1), nRoundNext));
                    copyGmToL1A(l1ATensor, gmTilex, layoutxInL1, layoutTilex);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                    // load next Matrix A tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nActualNext));
                    copyGmToL1B(l1BTensor, gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }

                // get L1 Tensor for current stage
                auto l1ATensor = l1ATensorList[l1ListId];
                auto l1BTensor = l1BTensorList[l1ListId];

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

                uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);
                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++)
                {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutxInL0 layoutxInL0 = LayoutxInL0::template MakeLayout<Elementx>(L1XAlignHelper::M_ALIGNED, nPartActual);

                    MatrixCoord l1xOffset{0, nPartIdx * L0TileShape::N};
                    auto l1ATile = l1ATensor[layoutxInL1.GetOffset(l1xOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutxInL0, layoutxInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

                    // Locate the current tile on L0B
                    auto l0BTile = l0BTensorList[l0BListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(L0TileShape::M, nPartActual);

                    MatrixCoord l1AOffset{0, nPartIdx * L0TileShape::N};
                    auto l1BTile = l1BTensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    // Load current tile from L1 to L0B
                    copyL1ToL0B(l0BTile, l1BTile, layoutAInL0, layoutAInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

                    // auto l0CTile = l0CTensor[(singleIdx * L0C_TILE_SIZE) % L0C_TILE_NUM];
                    // auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];
                    auto l0CTile = l0CTensor;

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = ((nLoopIdx == 0) && (nPartIdx == 0));

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                    tileMmad(l0CTile, l0ATile, l0BTile, L1XAlignHelper::M_ALIGNED, L0TileShape::M, nPartActual, initC);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    // 更新l0BListId 和 l0AListId
                    l0AListId = (l0AListId + 1) % STAGES;
                    l0BListId = (l0BListId + 1) % STAGES;

                    // AscendC::PipeBarrier<PIPE_ALL>(); // 不加这个会报错，但是还不知道如何优化
                }
                // AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);

                // 更新l1ListId
                l1ListId = l1ListIdNext;
                nActual = nActualNext;
            }

            // auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];
            auto l0CTile = l0CTensor;

            // copy block out
            Layouty layoutBlock = layouty.GetTileLayout(MakeCoord(uint32_t(1), actualShape.m()));

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

            copyL0CToGm(gmBlocky, l0CTile, layoutBlock, layoutInL0C);

            // AscendC::PipeBarrier<PIPE_ALL>();
        }

    protected:
        // Multi-stage tensors list
        AscendC::LocalTensor<Elementx> l1ATensorList[STAGES];
        AscendC::LocalTensor<ElementA> l1BTensorList[STAGES];
        AscendC::LocalTensor<Elementx> l0ATensorList[STAGES];
        AscendC::LocalTensor<ElementA> l0BTensorList[STAGES];
        AscendC::LocalTensor<ElementAccumulator> l0CTensor;

        // Multi-stage event id list
        int32_t l1AEventList[STAGES];
        int32_t l1BEventList[STAGES];
        int32_t l0AEventList[STAGES];
        int32_t l0BEventList[STAGES];

        // The id of current stage
        uint32_t l1ListId{0};
        uint32_t l0AListId{0};
        uint32_t l0BListId{0};

        TileMmad tileMmad;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;
    };

} // namespace acot::gemv::block

#endif // ACOT_GEMV_BLOCK_BLOCK_GEMV_PINGPONG_PL_HPP