#ifndef ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP
#define ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"
#include "acot/epilogue/helper.hpp"

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
    EpilogueAscendC910B3Gemm,
    CType_, // bf16
    XType_, // fp32
    DType_, // bf16
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMul_,
    TileCopy_
>{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAscendC910B3Gemm;
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
    
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES; // 开启双缓冲机制的 两个AIV核
    static constexpr uint32_t DOUBLESTAGES = DispatchPolicy::STAGES * DispatchPolicy::STAGES;
    const uint32_t UBSize = ArchTag::UB_SIZE;
    static constexpr bool RowOrColumn = std::is_same<LayoutC, acot::layout::RowMajor>::value && std::is_same<LayoutX, acot::layout::RowMajor>::value;
    static constexpr bool isNeedCast = std::is_same<ElementC, ElementX>::value;
    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueAdd::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    using ElementCompute =
        typename acot::epilogue::helper::ElementAccumulatorSelector<ElementC, ElementD>::ElementAccumulator;
    using ElementScalar = ElementCompute; // 标量的数据类型

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueAdd::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMul::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    // Check if compute length is valid
    static_assert(COMPUTE_LENGTH * OPERANDS_NUM * sizeof(ElementCompute) <= ArchTag::UB_SIZE, "UB out of bounds");

    typedef struct Params{
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrC;
        LayoutC layoutC; // 有stride的信息
        GM_ADDR ptrD;
        LayoutD layoutD;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrD_, LayoutD layoutD_)
        : alpha(alpha_), beta(beta_), ptrC(ptrC_), layoutC(layoutC_), ptrD(ptrD_), layoutD(layoutD_){}
    }Params;

    // 管理内存的
    ACOT_DEVICE
    BlockEpilogue(MatmulCoord blockShape_, Params params_) : blockShape(blockShape_), params(params_){
        cGm.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        dGm.SetGlobalBuffer((__gm__ ElementD*)params.ptrD);
        Resource();
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE2
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)i);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(i + STAGES));
        }
    }

    ACOT_DEVICE
    ~BlockEpilogue(){
        for(uint32_t i = 0; i < STAGES; i++){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)i);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(i + STAGES));
        }
        // 销毁内存
        pipe.Destroy();
    }

    ACOT_DEVICE
    void operator()(
        uint32_t offsetC, // A * B 矩阵
        uint32_t offsetD, // 最后结果
        AscendC::GlobalTensor<ElementX> gmBlockX,
        LayoutX layoutX,
        MatmulCoord actualShape
    ){ // 进行操作 先实现行优先
        // 分配内存
        xGm = gmBlockX;
        EpilogueGemm( // 使用第三种方法
            offsetC,
            offsetD,
            actualShape,
            layoutX
        );
    }
private:
    MatmulCoord blockShape;
    Params params;

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<ElementC> cGm;
    AscendC::GlobalTensor<ElementX> xGm;
    AscendC::GlobalTensor<ElementD> dGm;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> VECQueue; // 只开一个全空间的 
    // 开启双缓冲空间
    AscendC::LocalTensor<ElementC> ubCTensor[STAGES];
    AscendC::LocalTensor<ElementX> ubXTensor[STAGES];
    AscendC::LocalTensor<ElementD> ubDTensor[STAGES];
    AscendC::LocalTensor<ElementCompute> ubCTensorCast[STAGES];
    AscendC::LocalTensor<ElementCompute> ubDTensorCast[STAGES];

    // 搬运函数
    CopyGmToUbC copyGmToUbC;
    CopyGmToUbX copyGmToUbX;
    CopyUbToGmD copyUbToGmD;

    // 计算函数
    TileElemWiseEpilogueAdd tileElemWiseEpilogueAdd;
    TileElemWiseEpilogueMul tileElemWiseEpilogueMul;

    ACOT_DEVICE
    void Resource(){
        // 分配内存 分成三块
        uint32_t maxMPerBlock = blockShape.m() / STAGES; // M 和 N方向都切一刀
        uint32_t maxNPerBlock = blockShape.n() / STAGES;
        uint32_t tileSize = maxMPerBlock * maxNPerBlock;
        uint32_t ubCSize = tileSize * sizeof(ElementC);
        uint32_t ubXSize = tileSize * sizeof(ElementX);
        uint32_t ubDSize = tileSize * sizeof(ElementD);
        uint32_t ubCSizeCast = tileSize * sizeof(ElementCompute);
        uint32_t ubDSizeCast = tileSize * sizeof(ElementCompute);
        // 三个数据，要分成6份数据块
        uint32_t ubByteStart = 0;
        pipe.InitBuffer(VECQueue, 1, UBSize);
        AscendC::LocalTensor<uint8_t> ubTensor = VECQueue.AllocTensor<uint8_t>();
        ubCTensor[0] = ubTensor[ubByteStart].template ReinterpretCast<ElementC>(); 
        ubByteStart += ubCSize;
        ubCTensor[1] = ubTensor[ubByteStart].template ReinterpretCast<ElementC>(); 
        ubByteStart += ubCSize;
        ubXTensor[0] = ubTensor[ubByteStart].template ReinterpretCast<ElementX>();
        ubByteStart += ubXSize;
        ubXTensor[1] = ubTensor[ubByteStart].template ReinterpretCast<ElementX>();
        ubByteStart += ubXSize;
        ubDTensor[0] = ubTensor[ubByteStart].template ReinterpretCast<ElementD>();
        ubByteStart += ubDSize;
        ubDTensor[1] = ubTensor[ubByteStart].template ReinterpretCast<ElementD>();
        ubByteStart += ubDSize;
        // 转换的空间
        ubCTensorCast[0] = ubTensor[ubByteStart].template ReinterpretCast<ElementCompute>();
        ubByteStart += ubCSizeCast;
        ubCTensorCast[1] = ubTensor[ubByteStart].template ReinterpretCast<ElementCompute>();
        ubByteStart += ubCSizeCast;
        ubDTensorCast[0] = ubTensor[ubByteStart].template ReinterpretCast<ElementCompute>();
        ubByteStart += ubDSizeCast;
        ubDTensorCast[1] = ubTensor[ubByteStart].template ReinterpretCast<ElementCompute>();
        ubByteStart += ubDSizeCast;
    }
    
    ACOT_DEVICE
    void EpilogueRowMajor(
        uint32_t offsetC,
        uint32_t offsetD,
        MatmulCoord actualShape,
        LayoutX layoutX
    ){
        uint32_t MActual = actualShape.m();
        uint32_t NActual = actualShape.n(); // 这里也要对齐
        uint32_t maxMPerBlock = blockShape.m() / STAGES; // 对着M方向进行切分 肯定会对齐32Byte
        uint32_t maxNPerBlock = blockShape.n() / STAGES; 
        uint32_t aivIndex = AscendC::GetSubBlockIdx(); // 0 或 1
        uint32_t MActualAIV0 = (MActual < maxMPerBlock) ? MActual : maxMPerBlock;
        uint32_t MActualAIV1 = (MActual < maxMPerBlock) ? 0 : (MActual - maxMPerBlock);
        uint32_t MUbActual = aivIndex == 1 ? MActualAIV1 : MActualAIV0;
        uint32_t aivNum = AscendC::GetSubBlockNum(); // 910B3 AIV核为2
        uint32_t NLoops = CeilDiv(NActual, maxNPerBlock);
        for(uint32_t NIdx = 0; NIdx < NLoops; NIdx++){
            uint32_t NUbActual = (NIdx == NLoops - 1) ? (NActual - NIdx * maxNPerBlock) : maxNPerBlock;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)NIdx);
            auto layoutXInUb = layoutX.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutTileX = layoutX.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            copyGmToUbX(
                ubXTensor[NIdx % STAGES],
                xGm[aivIndex * maxMPerBlock * layoutX.stride(0) + NIdx * maxNPerBlock],
                layoutXInUb, layoutTileX
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)NIdx);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)NIdx);
            tileElemWiseEpilogueMul(
                ubXTensor[NIdx % STAGES],
                ubXTensor[NIdx % STAGES],
                params.alpha
            );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)NIdx);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(NIdx + STAGES));
            auto layoutTileC = params.layoutC.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            auto layoutCInUb = params.layoutC.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            copyGmToUbC(
                ubCTensor[NIdx % STAGES],
                cGm[offsetC + aivIndex * maxMPerBlock * params.layoutC.stride(0) + NIdx * maxNPerBlock],
                layoutCInUb, layoutTileC
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx + STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx + STAGES));
            if constexpr (!isNeedCast){
                AscendC::Cast<ElementCompute, ElementC>(ubCTensorCast[NIdx % STAGES], ubCTensor[NIdx % STAGES], AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx + STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx + STAGES));
                tileElemWiseEpilogueMul(
                    ubCTensorCast[NIdx % STAGES],
                    ubCTensorCast[NIdx % STAGES],
                    params.beta
                );
            }else{
                tileElemWiseEpilogueMul(
                    ubCTensor[NIdx % STAGES],
                    ubCTensor[NIdx % STAGES],
                    params.beta
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx + STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx + STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)NIdx);
            if constexpr (!isNeedCast){
                tileElemWiseEpilogueAdd(
                    ubDTensorCast[NIdx % STAGES],
                    ubXTensor[NIdx % STAGES],
                    ubCTensorCast[NIdx % STAGES]
                );
            }else{
                tileElemWiseEpilogueAdd(
                    ubDTensor[NIdx % STAGES],
                    ubXTensor[NIdx % STAGES],
                    ubCTensor[NIdx % STAGES]
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)NIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(NIdx + STAGES));
            if constexpr (!isNeedCast){
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast<ElementD, ElementCompute>(ubDTensor[NIdx % STAGES], ubDTensorCast[NIdx % STAGES], AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutTileD = params.layoutD.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutDInGm = params.layoutD.GetTileLayout(MakeCoord(MUbActual, NUbActual));
            copyUbToGmD(
                dGm[offsetD + aivIndex * maxMPerBlock * params.layoutD.stride(0) + NIdx * maxNPerBlock],
                ubDTensor[NIdx % STAGES],
                layoutDInGm, layoutTileD
            );
        }
    }

    ACOT_DEVICE
    void EpilogueColumnMajor(
        uint32_t offsetC,
        uint32_t offsetD,
        MatmulCoord actualShape,
        LayoutX layoutX
    ){
        uint32_t MActual = actualShape.m();
        uint32_t NActual = actualShape.n(); // 这里也要对齐
        uint32_t maxMPerBlock = blockShape.m() / STAGES; // 对着M方向进行切分 肯定会对齐32Byte
        uint32_t maxNPerBlock = blockShape.n() / STAGES; 
        uint32_t aivIndex = AscendC::GetSubBlockIdx();
        uint32_t NActualAIV0 = (NActual < maxNPerBlock) ? NActual : maxNPerBlock;
        uint32_t NActualAIV1 = (NActual < maxNPerBlock) ? 0 : (NActual - maxNPerBlock);
        uint32_t NUbActual = aivIndex == 1 ? NActualAIV1 : NActualAIV0;
        uint32_t aivNum = AscendC::GetSubBlockNum(); // 910B3 AIV核为2
        uint32_t MLoops = CeilDiv(MActual, maxMPerBlock);
        for(uint32_t MIdx = 0; MIdx < MLoops; MIdx++){
            uint32_t MUbActual = (MIdx == MLoops - 1) ? (MActual - MIdx * maxMPerBlock) : maxMPerBlock;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)MIdx);
            auto layoutXInUb = layoutX.GetTileLayout(MakeCoord(maxNPerBlock, maxMPerBlock));
            auto layoutTileX = layoutX.GetTileLayout(MakeCoord(NUbActual, MUbActual));
            copyGmToUbX(
                ubXTensor[MIdx % STAGES],
                xGm[aivIndex * maxNPerBlock * layoutX.stride(1) + MIdx * maxMPerBlock],
                layoutXInUb, layoutTileX
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)MIdx);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)MIdx);
            tileElemWiseEpilogueMul(
                ubXTensor[MIdx % STAGES],
                ubXTensor[MIdx % STAGES],
                params.alpha
            );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)MIdx);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(MIdx + STAGES));
            auto layoutTileC = params.layoutC.GetTileLayout(MakeCoord(NUbActual, MUbActual));
            auto layoutCInUb = params.layoutC.GetTileLayout(MakeCoord(maxNPerBlock, maxMPerBlock));
            copyGmToUbC(
                ubCTensor[MIdx % STAGES],
                cGm[offsetC + aivIndex * maxNPerBlock * params.layoutC.stride(1) + MIdx * maxMPerBlock],
                layoutCInUb, layoutTileC
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx + STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx + STAGES));
            if constexpr (!isNeedCast){
                AscendC::Cast<ElementCompute, ElementC>(ubCTensorCast[MIdx % STAGES], ubCTensor[MIdx % STAGES], AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx + STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx + STAGES));
                tileElemWiseEpilogueMul(
                    ubCTensorCast[MIdx % STAGES],
                    ubCTensorCast[MIdx % STAGES],
                    params.beta
                );
            }else{
                tileElemWiseEpilogueMul(
                    ubCTensor[MIdx % STAGES],
                    ubCTensor[MIdx % STAGES],
                    params.beta
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx + STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx + STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)MIdx);
            if constexpr (!isNeedCast){
                tileElemWiseEpilogueAdd(
                    ubDTensorCast[MIdx % STAGES],
                    ubXTensor[MIdx % STAGES],
                    ubCTensorCast[MIdx % STAGES]
                );
            }else{
                tileElemWiseEpilogueAdd(
                    ubDTensor[MIdx % STAGES],
                    ubXTensor[MIdx % STAGES],
                    ubCTensor[MIdx % STAGES]
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)MIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(MIdx + STAGES));
            if constexpr (!isNeedCast){
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast<ElementD, ElementCompute>(ubDTensor[MIdx % STAGES], ubDTensorCast[MIdx % STAGES], AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutTileD = params.layoutD.GetTileLayout(MakeCoord(maxNPerBlock, maxMPerBlock));
            auto layoutDInGm = params.layoutD.GetTileLayout(MakeCoord(NUbActual, MUbActual));
            copyUbToGmD(
                dGm[offsetD + aivIndex * maxNPerBlock * params.layoutD.stride(1) + MIdx * maxMPerBlock],
                ubDTensor[MIdx % STAGES],
                layoutDInGm, layoutTileD
            );
        }
    }
    
    ACOT_DEVICE
    void EpilogueGemm(
        uint32_t offsetC,
        uint32_t offsetD,
        MatmulCoord actualShape,
        LayoutX layoutX
    ){
        if constexpr (RowOrColumn){ // 开启双缓冲
            EpilogueRowMajor(offsetC, offsetD, actualShape, layoutX);
        }else{ // 对着N方向进行切分处理
            EpilogueColumnMajor(offsetC, offsetD, actualShape, layoutX);
        }
    }
};
}

#endif // ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP