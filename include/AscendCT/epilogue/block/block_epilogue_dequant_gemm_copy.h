#ifndef ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMM_DEQUANT_HPP
#define ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMM_DEQUANT_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/resource.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/matrix_coord.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/detail/callback.hpp"

namespace AscendCT::epilogue::block{
template<
    class XType_,
    class ScaleType_,
    class PerTokenScaleType_,
    class BiasType_,
    class CType_,
    class TileRowBroadcastMul_,
    class TileBroadcastOneBlk_,
    class TileOneBlkColumnBroadcastMul_,
    class TileRowBroadcastAdd_,
    class TileElemWiseCastTemp_,
    class TileElemWiseCastC_,
    class TileCopy_,
    class EpilogueTileSwizzle_
>
class BlockEpilogue<
    epilogue::EpilogueAtlasA2ElemWiseOneSource,
    XType_, // A * B 矩阵
    ScaleType_,
    PerTokenScaleType_,
    BiasType_,
    CType_,
    TileRowBroadcastMul_,
    TileBroadcastOneBlk_,
    TileOneBlkColumnBroadcastMul_,
    TileRowBroadcastAdd_,
    TileElemWiseCastTemp_,
    TileElemWiseCastC_,
    TileCopy_,
    EpilogueTileSwizzle_
>{
public:
    // Type aliases
    using DispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element; // bf16
    using LayoutC = typename CType_::Layout;
    using ElementTemp = float; // fp32
    using ElementX = typename XType_::Element; // int32
    using LayoutX = typename XType_::Layout;
    using ElementScale = typename ScaleType_::Element; // fp32
    using LayoutScale = typename ScaleType_::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType_::Element; // bf16
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;
    using ElementBias = typename BiasType_::Element; // bf16
    using LayoutBias = typename BiasType_::Layout;
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;
    using TileRowBroadcastAdd = TileRowBroadcastAdd_;
    using TileElemWiseCastTemp = TileElemWiseCastTemp_;
    using TileElemWiseCastC = TileElemWiseCastC_; // 这个需要进行新增
    using EpilogueTileSwizzle = EpilogueTileSwizzle_;
    using CopyGmToUbX = typename TileCopy_::CopyGmToUbX;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbScale;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbPerTokenScale;
    using CopyGmToUbBias = typename TileCopy_::CopyGmToUbBias;
    using CopyUbToGmC = typename TileCopy_::CopyUbToGmC;
    using TileShape = typename TileRowBroadcastMul::TileShape;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    static constexpr uint32_t STAGES = 2; // 两个AIV核
    const uint32_t UBSize = ArchTag::UB_SIZE;
    static constexpr bool RowOrColumn = std::is_same<LayoutC, AscendCT::layout::RowMajor>::value && std::is_same<LayoutX, AscendCT::layout::RowMajor>::value;

    struct Params{
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR ptrBias;
        LayoutBias layoutBias;
        GM_ADDR ptrC;
        LayoutC layoutC;
        
        ASCENDCT_HOST_DEVICE
        Params() {}

        ASCENDCT_HOST_DEVICE
        Params(GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale layoutPerTokenScale_,
                GM_ADDR ptrBias_, LayoutBias layoutBias_, GM_ADDR ptrC_, LayoutC layoutC_)
            : ptrScale(ptrScale_), layoutScale(layoutScale_), ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
                ptrBias(ptrBias_), layoutBias(layoutBias_), ptrC(ptrC_), layoutC(layoutC_){}

    };

    ASCENDCT_DEVICE
    BlockEpilogue(arch::Resource<ArchTag> &resource, MatmulCoord blockShape_, Params const& params_, uint32_t ubByteStart = 0) : blockShape(blockShape_), params(params_){
        uint32_t maxMPerBlock = blockShape.m();
        uint32_t maxNPerBlock = blockShape.n();
        uint32_t tileSize = maxMPerBlock * maxNPerBlock / STAGES;
        uint32_t ubCSize = tileSize * sizeof(ElementC);
        uint32_t ubXSize = tileSize * sizeof(ElementX);
        uint32_t ubTempSize = tileSize * sizeof(ElementTemp); // cast的空间
        uint32_t ubScaleSize = maxMPerBlock * sizeof(ElementScale);
        uint32_t ubPerTokenScaleSize = maxNPerBlock * sizeof(ElementPerTokenScale);
        uint32_t ubBiasSize = maxNPerBlock * sizeof(ElementBias);
        ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubByteStart);
        ubByteStart += ubCSize;
        ubXTensor = resource.ubBuf.template GetBufferByByte<ElementX>(ubByteStart);
        ubByteStart += ubXSize;
        ubTempTensor = resource.ubBuf.template GetBufferByByte<ElementTemp>(ubByteStart);
        ubByteStart += ubTempSize;
        ubScaleTensor = resource.ubBuf.template GetBufferByByte<ElementScale>(ubByteStart);
        ubByteStart += ubScaleSize;
        ubPerTokenScaleTensor = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubByteStart);
        ubByteStart += ubPerTokenScaleSize;
        ubPerTokenScaleBrcbTensor = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubByteStart);
        ubByteStart += TileShape::ROW * BYTE_PER_BLK;
        ubBiasTensor = resource.ubBuf.template GetBufferByByte<ElementBias>(ubByteStart);
        ubByteStart += ubBiasSize;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    ASCENDCT_DEVICE
    ~BlockEpilogue(){
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    ASCENDCT_DEVICE
    void operator()(
        uint32_t offsetC, uint32_t offsetScale, uint32_t offsetPerTokenScale, uint32_t offsetBias,
        AscendC::GlobalTensor<ElementX> gmBlockX, LayoutX layoutX, MatmulCoord actualShape
    ){
        AscendC::GlobalTensor<ElementC> gmBlockC;
        gmBlockC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrC));
        AscendC::GlobalTensor<ElementScale> gmBlockScale;
        gmBlockScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(params.ptrScale));
        AscendC::GlobalTensor<ElementPerTokenScale> gmBlockPerTokenScale;
        gmBlockPerTokenScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale*>(params.ptrPerTokenScale));
        AscendC::GlobalTensor<ElementBias> gmBlockBias;
        gmBlockBias.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBias*>(params.ptrBias));
        if constexpr(RowOrColumn){
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); 
            uint32_t maxMPerBlock = blockShape.m() / STAGES; 
            uint32_t maxNPerBlock = blockShape.n(); 
            uint32_t aivIndex = AscendC::GetSubBlockIdx(); 
            uint32_t MActualAIV0 = (MActual < maxMPerBlock) ? MActual : maxMPerBlock;
            uint32_t MActualAIV1 = (MActual < maxMPerBlock) ? 0 : (MActual - maxMPerBlock);
            uint32_t MUbActual = aivIndex == 1 ? MActualAIV1 : MActualAIV0;
            uint32_t NUbActual = NActual;
            LayoutC layoutInUb{maxMPerBlock, maxNPerBlock};
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            auto layoutTileX = layoutX.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutXInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileXOffset{aivIndex * maxMPerBlock, 0};
            auto gmTileX = gmBlockX[layoutX.GetOffset(gmTileXOffset)];
            copyGmToUbX(ubXTensor, gmTileX, layoutXInUb, layoutTileX);
            // begin cast
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            tileElemWiseCastTemp(ubTempTensor, ubXTensor); // fp32
            AscendC::PipeBarrier<PIPE_V>();
            // copy
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            auto layoutTileScale = params.layoutScale.GetTileLayout(MakeCoord(NUbActual));
            auto layoutScaleInUb = params.layoutScale.GetTileLayout(MakeCoord(NUbActual));
            TensorCoord gmTileScaleOffset{0};
            auto gmTileScale = gmBlockScale[offsetScale + params.layoutScale.GetOffset(gmTileScaleOffset)];
            copyGmToUbScale(ubScaleTensor, gmTileScale, layoutScaleInUb, layoutTileScale);
            auto layoutTilePerTokenScale = params.layoutPerTokenScale.GetTileLayout(MakeCoord(MUbActual));
            auto layoutPerTokenScaleInUb = params.layoutPerTokenScale.GetTileLayout(MakeCoord(MUbActual));
            TensorCoord gmTilePerTokenScaleOffset{0};
            auto gmTilePerTokenScale = gmBlockPerTokenScale[offsetPerTokenScale + params.layoutPerTokenScale.GetOffset(gmTilePerTokenScaleOffset)];
            copyGmToUbPerTokenScale(ubPerTokenScaleTensor, gmTilePerTokenScale, layoutPerTokenScaleInUb, layoutTilePerTokenScale);
            auto layoutTileBias = params.layoutBias.GetTileLayout(MakeCoord(NUbActual));
            auto layoutBiasInUb = params.layoutBias.GetTileLayout(MakeCoord(NUbActual));
            TensorCoord gmTileBiasOffset{0};
            auto gmTileBias = gmBlockBias[offsetBias + params.layoutBias.GetOffset(gmTileBiasOffset)];
            copyGmToUbBias(ubBiasTensor, gmTileBias, layoutBiasInUb, layoutTileBias);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            // AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + 2);
            // start compute
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            tileRowBroadcastMul(ubTempTensor, ubTempTensor, ubScaleTensor);
            tileBroadcastOneBlk(ubPerTokenScaleBrcbTensor, ubPerTokenScaleTensor);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubTempTensor, ubTempTensor, ubPerTokenScaleBrcbTensor);
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastAdd(ubTempTensor, ubTempTensor, ubBiasTensor);
            AscendC::PipeBarrier<PIPE_V>();
            tileElemWiseCastC(ubCTensor, ubTempTensor); // bf16
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutCInGm = params.layoutC.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutTileC = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileCOffset{aivIndex * maxMPerBlock, 0}; 
            auto gmTileC = gmBlockC[offsetC + params.layoutC.GetOffset(gmTileCOffset)];
            copyUbToGmC(gmTileC, ubCTensor, layoutCInGm, layoutTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }else{

        }
    }

private:
    MatmulCoord blockShape;
    Params params;

    AscendC::LocalTensor<ElementC> ubCTensor;
    AscendC::LocalTensor<ElementTemp> ubTempTensor;
    AscendC::LocalTensor<ElementX> ubXTensor;
    AscendC::LocalTensor<ElementScale> ubScaleTensor;
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleTensor;
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleBrcbTensor;
    AscendC::LocalTensor<ElementBias> ubBiasTensor;

    CopyGmToUbX copyGmToUbX;
    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyGmToUbBias copyGmToUbBias;
    CopyUbToGmC copyUbToGmC;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;
    TileRowBroadcastAdd tileRowBroadcastAdd;
    TileElemWiseCastTemp tileElemWiseCastTemp;
    TileElemWiseCastC tileElemWiseCastC;
};
}

#endif // ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMM_DEQUANT_HPP