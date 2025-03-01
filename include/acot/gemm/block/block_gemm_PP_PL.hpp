#ifndef ACOT_GEMM_BLOCK_BLOCK_GEMM_PINGPONG_HPP
#define ACOT_GEMM_BLOCK_BLOCK_GEMM_PINGPONG_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"
#include "acot/gemm/tile/tile_copy.hpp"
#include "acot/gemm/tile/tile_mmad.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/gemm/dispatch_policy.hpp"
#include "acot/arch/resource.hpp"

namespace acot::gemm::block{
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockGemm<
    GemmAscendC910B3Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>{
public:
    using DispatchPolicy = GemmAscendC910B3Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    using LayoutBInL1 = typename CopyGmToL1B::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES; // 开启双缓冲机制的
    const uint32_t L1Size = ArchTag::L1_SIZE;
    const uint32_t L1ASize = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    const uint32_t L1BSize = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    const uint32_t cSize = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);
    const uint32_t BlockCnt = L1TileShape::M * L1TileShape::N;
    const uint32_t L0ASize = ArchTag::L0A_SIZE;
    const uint32_t L0BSize = ArchTag::L0B_SIZE;
    const uint32_t L0CSize = ArchTag::L0C_SIZE;
    const uint32_t L0A_PINGPONG_BUF_LEN = (L0ASize / STAGES);
    const uint32_t L0B_PINGPONG_BUF_LEN = (L0BSize / STAGES);
    static constexpr bool RowOrColumn = std::is_same<LayoutA, layout::RowMajor>::value && std::is_same<LayoutB, layout::RowMajor>::value;
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;
    static constexpr uint32_t MAX_TENSOR_COUNT = 256;
    
    ACOT_DEVICE
    BlockGemm(arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0){
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1ASize * STAGES;
        for(uint32_t i = 0; i < STAGES; i++){
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1ASize * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1BSize * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_LEN * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_LEN * i);
            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }
    // destroy function
    ACOT_DEVICE
    ~BlockGemm(){
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    ACOT_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        MatmulCoord const &actualShape, MatmulCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx
    ){
        // 这一部分完成矩阵乘法
        Matmul(gmA, layoutA, gmB, layoutB, gmC, layoutC, gmNextBlockA, gmNextBlockB, actualShape, actualShapeNext, isFirstBlock, hasNextBlock, singleIdx);
    }
private:
    // 双缓冲区
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
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
    
    ACOT_DEVICE
    void MatmulRowMajor(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        MatmulCoord const &actualShape, MatmulCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx
    ){
        // 先实现行优先
        uint32_t MAlignment = C0_NUM_PER_FRACTAL;
        uint32_t NAlignment = BYTE_PER_C0 / sizeof(ElementB);
        if constexpr (std::is_same<ElementB, float>::value){
            NAlignment = C0_NUM_PER_FRACTAL; // N方向向16对齐
        }
        uint32_t MActual = actualShape.m();
        uint32_t NActual = actualShape.n();
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t KLoops = CeilDiv(K, maxKPerBlock);
        // 进行切分操作
        for(uint32_t KIdx = 0; KIdx < KLoops; KIdx++){
            uint32_t KGmActual = (KIdx == KLoops - 1) ? (K - KIdx * maxKPerBlock) : maxKPerBlock;
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), KGmActual));
            auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(actualShape.m(), KGmActual);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(KGmActual, actualShape.n()));
            auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(KGmActual, actualShape.n());
            if(KIdx % 1 == 0){ // ABBA
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);  // 涉及太多计算 减少计算
                copyGmToL1A(l1ATensor[KIdx % STAGES],gmA[KIdx * maxKPerBlock],layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[KIdx % STAGES],gmB[KIdx * maxKPerBlock * layoutB.stride(0)],layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }else{
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[KIdx % STAGES],gmB[KIdx * maxKPerBlock * layoutB.stride(0)],layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                copyGmToL1A(l1ATensor[KIdx % STAGES],gmA[KIdx * maxKPerBlock],layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            }

            // 在K方向再进行一次切分处理
            uint32_t KL0TileSize = L0TileShape::K;
            uint32_t KL0Loops = CeilDiv(KGmActual, KL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            for(uint32_t KL0Idx = 0; KL0Idx < KL0Loops; KL0Idx++){
                uint32_t KL0Actual = (KL0Idx == KL0Loops - 1) ? (KGmActual - KL0Idx * KL0TileSize) : KL0TileSize;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(actualShape.m(), KL0Actual);
                copyL1ToL0A(l0ATensor[KL0Idx % STAGES],l1ATensor[KIdx % STAGES][KL0Idx * KL0TileSize * MRound],layoutAInL0, layoutAInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(KL0Actual, actualShape.n());
                // 偏移量不一样
                if constexpr (std::is_same<ElementB, int8_t>::value){
                    copyL1ToL0B(l0BTensor[KL0Idx % STAGES],l1BTensor[KIdx % STAGES][KL0Idx * KL0TileSize * NAlignment],layoutBInL0, layoutBInL1);
                }else{
                    copyL1ToL0B(l0BTensor[KL0Idx % STAGES],l1BTensor[KIdx % STAGES][KL0Idx * KL0TileSize * NRound],layoutBInL0, layoutBInL1);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                
                // 进行计算
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                tileMmad(l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt],l0ATensor[KL0Idx % STAGES],l0BTensor[KL0Idx % STAGES],MRound,NRound,KL0Actual,(KIdx == 0) && (KL0Idx == 0));
                // if(MActual != MRound || NActual != NRound){
                //     AscendC::PipeBarrier<PIPE_M>();  // 优化点
                // }
                AscendC::PipeBarrier<PIPE_ALL>();  // 优化点 这里必须是all 不知道为啥
                // AscendC::PipeBarrier<PIPE_M>();  // 优化点
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                //更新l0BListId 和 l0AListId
                l0AListId = 1 - l0AListId;
                l0BListId = 1 - l0BListId;
            }
            // 方便下次循环的使用
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            //更新l1ListId
            l1ListId = 1 - l1ListId;
        }
    }

    ACOT_DEVICE
    void MatmulColumnMajor(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        MatmulCoord const &actualShape, MatmulCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx
    ){
        uint32_t MAlignment = C0_NUM_PER_FRACTAL; // 对齐32byte
        if constexpr (std::is_same<ElementA, int8_t>::value){
            MAlignment = BYTE_PER_C0 / sizeof(ElementA);
        }
        uint32_t NAlignment = C0_NUM_PER_FRACTAL; // 对齐16行
        uint32_t MActual = actualShape.m();
        uint32_t NActual = actualShape.n();
        uint32_t MRound = RoundUp(MActual, MAlignment);
        uint32_t NRound = RoundUp(NActual, NAlignment);
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t KLoops = CeilDiv(K, maxKPerBlock);
        for(uint32_t KIdx = 0; KIdx < KLoops; KIdx++){
            uint32_t KGmActual = (KIdx == KLoops - 1) ? (K - KIdx * maxKPerBlock) : maxKPerBlock;
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(KGmActual, actualShape.m()));
            auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(KGmActual, actualShape.m());
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.n(), KGmActual));
            auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(actualShape.n(), KGmActual);
            if(KIdx % 1 == 0){ // ABBA
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                copyGmToL1A(l1ATensor[KIdx % STAGES],gmA[KIdx * maxKPerBlock * layoutA.stride(1)],layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[KIdx % STAGES],gmB[KIdx * maxKPerBlock],layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }else{
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[KIdx % STAGES],gmB[KIdx * maxKPerBlock],layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                copyGmToL1A(l1ATensor[KIdx % STAGES],gmA[KIdx * maxKPerBlock * layoutA.stride(1)],layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            }
            
            uint32_t KL0TileSize = L0TileShape::K;
            uint32_t KL0Loops = CeilDiv(KGmActual, KL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            for(uint32_t KL0Idx = 0; KL0Idx < KL0Loops; KL0Idx++){
                uint32_t KL0Actual = (KL0Idx == KL0Loops - 1) ? (KGmActual - KL0Idx * KL0TileSize) : KL0TileSize;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(KL0Actual, actualShape.m());
                if constexpr (std::is_same<ElementA, int8_t>::value){
                    copyL1ToL0A(l0BTensor[KL0Idx % STAGES],l1ATensor[KIdx % STAGES][KL0Idx * KL0TileSize * MAlignment],layoutAInL0, layoutAInL1);
                }else{
                    copyL1ToL0A(l0BTensor[KL0Idx % STAGES],l1ATensor[KIdx % STAGES][KL0Idx * KL0TileSize * MRound],layoutAInL0, layoutAInL1);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(actualShape.n(), KL0Actual);
                copyL1ToL0B(l0ATensor[KL0Idx % STAGES],l1BTensor[KIdx % STAGES][KL0Idx * KL0TileSize * NRound],layoutBInL0, layoutBInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                // 进行计算
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                tileMmad(l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt],l0ATensor[KL0Idx % STAGES],l0BTensor[KL0Idx % STAGES],NRound,MRound,KL0Actual,(KIdx == 0) && (KL0Idx == 0));
                // if(MActual != MRound || NActual != NRound){
                //     AscendC::PipeBarrier<PIPE_M>();  // 优化点
                // }
                AscendC::PipeBarrier<PIPE_ALL>();  // 优化点
                // AscendC::PipeBarrier<PIPE_M>();  // 优化点
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                //更新l0BListId 和 l0AListId
                l0AListId = 1 - l0AListId;
                l0BListId = 1 - l0BListId;
            }
            // 方便下次循环的使用
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            //更新l1ListId
            l1ListId = 1 - l1ListId;
        }
    }

    ACOT_DEVICE
    void Matmul(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        MatmulCoord const &actualShape, MatmulCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx
    ){
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(L1TileShape::M, L1TileShape::N)); // 获得MNCoord
        if constexpr (RowOrColumn){
            MatmulRowMajor(gmA, layoutA, gmB, layoutB, gmC, layoutC, gmNextBlockA, gmNextBlockB, actualShape, actualShapeNext, isFirstBlock, hasNextBlock, singleIdx);
        }else{
            MatmulColumnMajor(gmA, layoutA, gmB, layoutB, gmC, layoutC, gmNextBlockA, gmNextBlockB, actualShape, actualShapeNext, isFirstBlock, hasNextBlock, singleIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0CBlockNum));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0CBlockNum));
        LayoutC layoutBlock = layoutC.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        copyL0CToGm(gmC,l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt],layoutBlock, layoutInL0C);
    }
};
}

#endif // ACOT_GEMM_BLOCK_BLOCK_GEMM_PINGPONG_HPP

/*
优化点：
    1. ABBA 现在已经加上了
    2. PreLoad 
    3. Double Buffer 已经加上了
    4. Swizzle
*/