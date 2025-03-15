#ifndef ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP
#define ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/gemm/helper.hpp"

namespace acot::epilogue::block {
template<
    class CType_, // C 矩阵
    class XType_, // A * B 矩阵
    class DType_, // D 矩阵
    class TileElemWiseEpilogueAdd_, // 单块元素的处理 包括加法和乘法
    class TileElemWiseEpilogueMul_, // 乘法的后处理
    class TileCopy_
>
class BlockEpilogue<
    EpilogueAtlasA2ElemWiseOneSource,
    CType_,
    XType_, 
    DType_, 
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMul_,
    TileCopy_
>{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA2ElemWiseOneSource;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element; // 只是获取类型名
    using LayoutC = typename CType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
    using TileElemWiseEpilogueMul = TileElemWiseEpilogueMul_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbX = typename TileCopy_::CopyGmToUbX;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    
    static constexpr uint32_t STAGES = 2; // 两个AIV核
    const uint32_t UBSize = ArchTag::UB_SIZE;
    static constexpr bool RowOrColumn = std::is_same<LayoutC, acot::layout::RowMajor>::value && std::is_same<LayoutX, acot::layout::RowMajor>::value;
    static constexpr bool isNeedCast = std::is_same<ElementC, bfloat16_t>::value; // 只对bfloat16_t进行类型转换处理
    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueAdd::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    using ElementCompute = typename acot::gemm::helper::ElementAccumulatorSelector<ElementC, ElementD>::ElementAccumulator;
    using ElementScalar = ElementCompute; // 标量的数据类型

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueAdd::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMul::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    // Check if compute length is valid
    static_assert(COMPUTE_LENGTH * OPERANDS_NUM * sizeof(ElementCompute) <= ArchTag::UB_SIZE, "UB out of bounds");

    // 管理内存的
    ACOT_DEVICE
    BlockEpilogue(arch::Resource<ArchTag> &resource, MatmulCoord blockShape_, uint32_t ubByteStart = 0) : blockShape(blockShape_){
        uint32_t maxMPerBlock = blockShape.m(); // M 和 N方向都切一刀
        uint32_t maxNPerBlock = blockShape.n();
        uint32_t tileSize = maxMPerBlock * maxNPerBlock / STAGES;
        uint32_t ubCSize = tileSize * sizeof(ElementC);
        uint32_t ubXSize = tileSize * sizeof(ElementX);
        uint32_t ubDSize = tileSize * sizeof(ElementD);
        uint32_t ubCSizeCast = tileSize * sizeof(ElementCompute);
        uint32_t ubDSizeCast = tileSize * sizeof(ElementCompute);
        ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubByteStart);
        ubByteStart += ubCSize;
        ubXTensor = resource.ubBuf.template GetBufferByByte<ElementX>(ubByteStart); //half
        ubByteStart += ubXSize;
        ubDTensor = resource.ubBuf.template GetBufferByByte<ElementD>(ubByteStart);
        ubByteStart += ubDSize;
        // 转换的空间
        if constexpr (isNeedCast){
            ubCTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);
            ubByteStart += ubCSizeCast;
            ubDTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);;
            ubByteStart += ubDSizeCast;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }

    ACOT_DEVICE
    ~BlockEpilogue(){
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }

    ACOT_DEVICE
    void operator()(
        ElementScalar alpha, ElementScalar beta, 
        AscendC::GlobalTensor<ElementC> gmBlockC, LayoutC layoutC, 
        AscendC::GlobalTensor<ElementD> gmBlockD, LayoutD layoutD, 
        AscendC::GlobalTensor<ElementX> gmBlockX, LayoutX layoutX, 
        MatmulCoord actualShape
    ){ // 进行操作 先实现行优先
        if constexpr(RowOrColumn){
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); // 这里也要对齐
            uint32_t maxMPerBlock = blockShape.m() / STAGES; // 对着M方向进行切分 肯定会对齐32Byte
            uint32_t maxNPerBlock = blockShape.n(); 
            uint32_t aivIndex = AscendC::GetSubBlockIdx(); // 0 或 1
            uint32_t MActualAIV0 = (MActual < maxMPerBlock) ? MActual : maxMPerBlock;
            uint32_t MActualAIV1 = (MActual < maxMPerBlock) ? 0 : (MActual - maxMPerBlock);
            uint32_t MUbActual = aivIndex == 1 ? MActualAIV1 : MActualAIV0;
            uint32_t NUbActual = NActual;
            LayoutC layoutInUb{maxMPerBlock, maxNPerBlock};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            auto layoutTileC = layoutC.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutCInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileCOffset{aivIndex * maxMPerBlock, 0}; // gm中的偏移量
            auto gmTileC = gmBlockC[layoutC.GetOffset(gmTileCOffset)];
            copyGmToUbC(ubCTensor, gmTileC, layoutCInUb, layoutTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            if constexpr (isNeedCast){
                AscendC::Cast<ElementCompute, ElementC>(ubCTensorCast, ubCTensor, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
                AscendC::PipeBarrier<PIPE_V>();
                tileElemWiseEpilogueMul(ubCTensorCast,ubCTensorCast, (ElementCompute)beta);
            }else{
                tileElemWiseEpilogueMul(ubCTensor,ubCTensor, (ElementC)beta);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            auto layoutTileX = layoutX.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutXInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileXOffset{aivIndex * maxMPerBlock, 0}; // gm中的偏移量
            auto gmTileX = gmBlockX[layoutX.GetOffset(gmTileXOffset)];
            copyGmToUbX(ubXTensor, gmTileX, layoutXInUb, layoutTileX);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            tileElemWiseEpilogueMul(ubXTensor, ubXTensor, (ElementX)alpha);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            if constexpr (isNeedCast){
                tileElemWiseEpilogueAdd(ubDTensorCast,ubXTensor,ubCTensorCast);
            }else{
                tileElemWiseEpilogueAdd(ubDTensor,ubXTensor,ubCTensor);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            if constexpr (isNeedCast){
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)-1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)-1);
                AscendC::Cast<ElementD, ElementCompute>(ubDTensor, ubDTensorCast, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutDInGm = layoutD.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutTileD = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileDOffset{aivIndex * maxMPerBlock, 0}; // gm中的偏移量
            auto gmTileD = gmBlockD[layoutD.GetOffset(gmTileDOffset)];
            copyUbToGmD(gmTileD, ubDTensor, layoutDInGm, layoutTileD);
        }else{ // 优化手段不一样
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); // 这里也要对齐
            uint32_t maxMPerBlock = blockShape.m(); // 对着M方向进行切分 肯定会对齐32Byte
            uint32_t maxNPerBlock = blockShape.n() / STAGES; 
            uint32_t aivIndex = AscendC::GetSubBlockIdx(); // 0 或 1
            uint32_t NActualAIV0 = (NActual < maxNPerBlock) ? NActual : maxNPerBlock;
            uint32_t NActualAIV1 = (NActual < maxNPerBlock) ? 0 : (NActual - maxNPerBlock);
            uint32_t NUbActual = aivIndex == 1 ? NActualAIV1 : NActualAIV0;
            uint32_t MUbActual = MActual;
            LayoutC layoutInUb{maxMPerBlock, maxNPerBlock};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            auto layoutTileC = layoutC.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutCInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileCOffset{0, aivIndex * maxNPerBlock}; // gm中的偏移量
            auto gmTileC = gmBlockC[layoutC.GetOffset(gmTileCOffset)];
            copyGmToUbC(ubCTensor,gmTileC,layoutCInUb, layoutTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            if constexpr (isNeedCast){
                AscendC::Cast<ElementCompute, ElementC>(ubCTensorCast, ubCTensor, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
                AscendC::PipeBarrier<PIPE_V>();
                tileElemWiseEpilogueMul(ubCTensorCast,ubCTensorCast,(ElementCompute)beta);
            }else{
                tileElemWiseEpilogueMul(ubCTensor,ubCTensor,(ElementC)beta);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            auto layoutTileX = layoutX.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutXInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileXOffset{0, aivIndex * maxNPerBlock}; // gm中的偏移量
            auto gmTileX = gmBlockX[layoutX.GetOffset(gmTileXOffset)];
            copyGmToUbX(ubXTensor,gmTileX,layoutXInUb, layoutTileX);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            tileElemWiseEpilogueMul(ubXTensor,ubXTensor,(ElementX)alpha);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            if constexpr (isNeedCast){
                tileElemWiseEpilogueAdd(ubDTensorCast,ubXTensor,ubCTensorCast);
            }else{
                tileElemWiseEpilogueAdd(ubDTensor,ubXTensor,ubCTensor);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            if constexpr (isNeedCast){
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)-1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)-1);
                AscendC::Cast<ElementD, ElementCompute>(ubDTensor, ubDTensorCast, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutDInGm = layoutD.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutTileD = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            MatrixCoord gmTileDOffset{0, aivIndex * maxNPerBlock}; // gm中的偏移量
            auto gmTileD = gmBlockD[layoutD.GetOffset(gmTileDOffset)];
            copyUbToGmD(gmTileD,ubDTensor,layoutDInGm, layoutTileD);
        }
    }
private:
    MatmulCoord blockShape;

    AscendC::LocalTensor<ElementC> ubCTensor;
    AscendC::LocalTensor<ElementX> ubXTensor;
    AscendC::LocalTensor<ElementD> ubDTensor;
    AscendC::LocalTensor<ElementCompute> ubCTensorCast;
    AscendC::LocalTensor<ElementCompute> ubDTensorCast;

    // 搬运函数
    CopyGmToUbC copyGmToUbC;
    CopyGmToUbX copyGmToUbX;
    CopyUbToGmD copyUbToGmD;

    // 计算函数
    TileElemWiseEpilogueAdd tileElemWiseEpilogueAdd;
    TileElemWiseEpilogueMul tileElemWiseEpilogueMul;
};
}

#endif // ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP