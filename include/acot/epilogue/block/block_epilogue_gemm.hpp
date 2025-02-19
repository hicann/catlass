#ifndef ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP
#define ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/epilogue/dispatch_policy.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/epilogue/tile/tile_copy.hpp"

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
    CType_,
    XType_,
    DType_,
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMul_,
    TileCopy_
>{
public:
    // Type aliases
    using DispatchPolicy = EpilogueAscendC910B3Gemm;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementScalar = typename CType_::Element; // 标量的数据类型
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
    const uint32_t UBSize = ArchTag::UBSize;
    static constexpr bool RowOrColumn = std::is_same<LayoutC, acot::layout::RowMajor>::value && std::is_same<LayoutX, acot::layout::RowMajor>::value;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueAdd::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    // Check the element type of C, X and D
    static_assert(std::is_same_v<ElementC, ElementD> && std::is_same_v<ElementX, ElementD>,
        "Element type of C, X and D must be the same");
    using ElementCompute = ElementD;

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueAdd::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMul::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    // Check if compute length is valid
    static_assert(COMPUTE_LENGTH * OPERANDS_NUM * sizeof(ElementCompute) <= ArchTag::UBSize, "UB out of bounds");

    typedef struct Params{
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrX;
        LayoutX layoutX; // 有stride的信息
        GM_ADDR ptrD;
        LayoutD layoutD;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrX_, LayoutX layoutX_, GM_ADDR ptrD_, LayoutD layoutD_)
        : alpha(alpha_), beta(beta_), ptrX(ptrX_), layoutX(layoutX_), ptrD(ptrD_), layoutD(layoutD_){}
    }Params;

    // 管理内存的
    ACOT_DEVICE
    BlockEpilogue(MatmulCoord blockShape_, Params params_) : blockShape(blockShape_), params(params_){
        cGm.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
        dGm.SetGlobalBuffer((__gm__ ElementD*)params.ptrD);
        Resource();
        // 开启流水控制
        for(uint32_t i = 0; i < STAGES; i++){
            // 使能MTE2
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)i);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(i + 2));
        }
    }

    ACOT_DEVICE
    ~BlockEpilogue(){
        // 关闭流水
        for(uint32_t i = 0; i < STAGES; i++){
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)i);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(i + 2));
        }
        // 销毁内存
        pipe.Destroy();
    }

    ACOT_DEVICE
    void operator()(
        uint32_t offsetC, // A * B 矩阵
        // uint32_t offsetX, // C 矩阵
        uint32_t offsetD, // 最后结果
        AscendC::GlobalTensor<ElementCompute> gmBlockX,
        LayoutC layoutX,
        MatmulCoord actualShape
        // uint32_t singleIdx
    ){ // 进行操作 先实现行优先
        // 分配内存
        xGm = gmBlockX;
        AlphaDotBeta1( // 使用第三种方法
            offsetC,
            // offsetX,
            offsetD,
            actualShape,
            layoutX
            // singleIdx
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
        uint32_t maxMPerBlock = blockShape.m();
        uint32_t maxNPerBlock = blockShape.n() / STAGES;
        uint32_t tileSize = maxMPerBlock * maxNPerBlock;
        uint32_t ubCSize = tileSize * sizeof(ElementC);
        uint32_t ubXSize = tileSize * sizeof(ElementX);
        uint32_t ubDSize = tileSize * sizeof(ElementD);
        // 三个数据，要分成6份数据块
        uint32_t ubByteStart = 0;
        pipe.InitBuffer(VECQueue, 1, UBSize);
        AscendC::LocalTensor<uint8_t> ubTensor = VECQueue.AllocTensor<uint8_t>();
        // 双缓冲空间 128 * 128 * 4 * 3 = 196,608 = 192 * 1024 能存下 极限情况
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
    }

    ACOT_DEVICE
    void AlphaDotBeta0(
        uint32_t offsetC,
        uint32_t offsetX,
        uint32_t offsetD,
        MatmulCoord actualShape
    ){
        if constexpr (RowOrColumn){
            // 搬运时需要对齐32Byte的
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); // 这里也要对齐
            // 行优先部分对M进行切分处理
            uint32_t maxMPerBlock = blockShape.m() / STAGES; // 对着M方向进行切分 肯定会对齐32Byte
            uint32_t maxNPerBlock = blockShape.n(); 
            uint32_t MLoops = CeilDiv(MActual, maxMPerBlock);
            auto layoutCInUb = params.layoutC.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutXInUb = params.layoutX.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutDInGm = params.layoutD.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            for(uint32_t MIdx = 0; MIdx < MLoops; MIdx++){
                uint32_t MUbActual = (MIdx == MLoops - 1) ? (MActual - MIdx * maxMPerBlock) : maxMPerBlock;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(MIdx % STAGES));
                auto layoutTileX = params.layoutX.GetTileLayout(MakeCoord(MUbActual, NActual));
                copyGmToUbX(
                    ubXTensor[MIdx % STAGES],
                    xGm[offsetX + MIdx * maxMPerBlock * params.layoutX.stride(0)],
                    layoutXInUb, layoutTileX
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES + 4));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES + 4));
                // 流水控制
                // 进行计算
                // 先进行乘法 alpha * A * B
                tileElemWiseEpilogueMul(
                    ubXTensor[MIdx % STAGES],
                    ubXTensor[MIdx % STAGES],
                    params.alpha
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES));

                // 现在单纯测试C矩阵的搬运  AIV核出现问题
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(MIdx % STAGES + 2));
                auto layoutTileC = params.layoutC.GetTileLayout(MakeCoord(MUbActual, NActual));
                copyGmToUbC(
                    ubCTensor[MIdx % STAGES],
                    cGm[offsetC + MIdx * maxMPerBlock * params.layoutC.stride(0)],
                    layoutCInUb, layoutTileC
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES + 6));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES + 6));
                // 进行乘法 beta * C
                tileElemWiseEpilogueMul(
                    ubCTensor[MIdx % STAGES],
                    ubCTensor[MIdx % STAGES],
                    params.beta // 获取参数内容
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES + 2));
                // 等待计算
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(MIdx % STAGES + 2));
                // 进行加法 alpha * A * B + beta * C
                tileElemWiseEpilogueAdd( // ubXTensor 数据不干净导致的数据异常
                    ubDTensor[MIdx % STAGES],
                    ubXTensor[MIdx % STAGES],
                    ubCTensor[MIdx % STAGES]
                );
                // ubX和ubC使用完了
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(MIdx % STAGES));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(MIdx % STAGES + 2));
                // 流水控制
                // 搬到Gm中
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((int32_t)(MIdx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((int32_t)(MIdx % STAGES));
                auto layoutTileD = params.layoutD.GetTileLayout(MakeCoord(MUbActual, NActual));
                copyUbToGmD(
                    dGm[offsetD + MIdx * maxMPerBlock * params.layoutX.stride(0)],
                    ubDTensor[MIdx % STAGES],
                    // ubCTensor[MIdx % STAGES],
                    layoutDInGm, layoutTileD
                );
            }
        }else{ // 对着N方向进行切分处理
            // 搬运时需要对齐32Byte的
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); // 这里也要对齐
            // 行优先部分对M进行切分处理
            uint32_t maxMPerBlock = blockShape.m(); // 对着M方向进行切分 肯定会对齐32Byte
            uint32_t maxNPerBlock = blockShape.n() / STAGES; 
            uint32_t NLoops = CeilDiv(NActual, maxNPerBlock);
            auto layoutCInUb = params.layoutC.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutXInUb = params.layoutX.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutDInGm = params.layoutD.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            for(uint32_t NIdx = 0; NIdx < NLoops; NIdx++){
                uint32_t NUbActual = (NIdx == NLoops - 1) ? (NActual - NIdx * maxNPerBlock) : maxNPerBlock;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(NIdx % STAGES));
                auto layoutTileX = params.layoutX.GetTileLayout(MakeCoord(MActual, NUbActual));
                copyGmToUbX(
                    ubXTensor[NIdx % STAGES],
                    xGm[offsetX + NIdx * maxNPerBlock * params.layoutX.stride(1)],
                    layoutXInUb, layoutTileX
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES + 4));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES + 4));
                // 流水控制
                // 进行计算
                // 先进行乘法 alpha * A * B
                tileElemWiseEpilogueMul(
                    ubXTensor[NIdx % STAGES],
                    ubXTensor[NIdx % STAGES],
                    params.alpha
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES));

                // 现在单纯测试C矩阵的搬运  AIV核出现问题
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(NIdx % STAGES + 2));
                auto layoutTileC = params.layoutC.GetTileLayout(MakeCoord(MActual, NUbActual));
                copyGmToUbC(
                    ubCTensor[NIdx % STAGES],
                    cGm[offsetC + NIdx * maxNPerBlock * params.layoutC.stride(1)],
                    layoutCInUb, layoutTileC
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES + 6));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES + 6));
                // 进行乘法 beta * C
                tileElemWiseEpilogueMul(
                    ubCTensor[NIdx % STAGES],
                    ubCTensor[NIdx % STAGES],
                    params.beta // 获取参数内容
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES + 2));
                // 等待计算
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(NIdx % STAGES + 2));
                // 进行加法 alpha * A * B + beta * C
                tileElemWiseEpilogueAdd(
                    ubDTensor[NIdx % STAGES],
                    ubXTensor[NIdx % STAGES],
                    ubCTensor[NIdx % STAGES]
                );
                // ubX和ubC使用完了
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(NIdx % STAGES));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(NIdx % STAGES + 2));
                // 流水控制
                // 搬到Gm中
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((int32_t)(NIdx % STAGES));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((int32_t)(NIdx % STAGES));
                auto layoutTileD = params.layoutD.GetTileLayout(MakeCoord(MActual, NUbActual));
                copyUbToGmD(
                    dGm[offsetD + NIdx * maxNPerBlock * params.layoutD.stride(1)],
                    ubDTensor[NIdx % STAGES],
                    // ubCTensor[NIdx % STAGES],
                    layoutDInGm, layoutTileD
                );
            }
        }
    }

    ACOT_DEVICE
    void AlphaDotBeta1(
        uint32_t offsetC,
        // uint32_t offsetX,
        uint32_t offsetD,
        MatmulCoord actualShape,
        LayoutC layoutC
    ){
        if constexpr (RowOrColumn){
            // 搬运时需要对齐32Byte的
            // 搬运时需要对齐32Byte的
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); // 这里也要对齐
            // 行优先部分对M进行切分处理
            uint32_t maxMPerBlock = blockShape.m() / STAGES; // 对着M方向进行切分 肯定会对齐32Byte
            uint32_t maxNPerBlock = blockShape.n(); 
            auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutXInUb = params.layoutX.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutDInGm = params.layoutD.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            uint32_t aivIndex = AscendC::GetSubBlockIdx(); // 0 或 1
            uint32_t MActualAIV0 = (MActual < maxMPerBlock) ? MActual : maxMPerBlock;
            uint32_t MActualAIV1 = (MActual < maxMPerBlock) ? 0 : (MActual - maxMPerBlock);
            uint32_t MUbActual = aivIndex ? MActualAIV1 : MActualAIV0;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES));
            auto layoutTileX = params.layoutX.GetTileLayout(MakeCoord(MUbActual, NActual));
            copyGmToUbX(
                ubXTensor[aivIndex % STAGES],
                // xGm[offsetX + aivIndex * maxMPerBlock * params.layoutX.stride(0)],
                xGm[aivIndex * maxMPerBlock * params.layoutX.stride(0)],
                layoutXInUb, layoutTileX
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 4));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 4));
            // 流水控制
            // 进行计算
            // 先进行乘法 alpha * A * B
            tileElemWiseEpilogueMul(
                ubXTensor[aivIndex % STAGES],
                ubXTensor[aivIndex % STAGES],
                params.alpha
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES));

            // 现在单纯测试C矩阵的搬运  AIV核出现问题
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES + 2));
            auto layoutTileC = layoutC.GetTileLayout(MakeCoord(MUbActual, NActual));
            copyGmToUbC(
                ubCTensor[aivIndex % STAGES],
                cGm[offsetC + aivIndex * maxMPerBlock * layoutC.stride(0)],
                layoutCInUb, layoutTileC
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 6));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 6));
            // 进行乘法 beta * C
            tileElemWiseEpilogueMul(
                ubCTensor[aivIndex % STAGES],
                ubCTensor[aivIndex % STAGES],
                params.beta // 获取参数内容
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 2));
            // 等待计算
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 2));
            // 进行加法 alpha * A * B + beta * C
            tileElemWiseEpilogueAdd( // ubXTensor 数据不干净导致的数据异常
                ubDTensor[aivIndex % STAGES],
                ubXTensor[aivIndex % STAGES],
                ubCTensor[aivIndex % STAGES]
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            // ubX和ubC使用完了
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES + 2));
            // 流水控制
            // 搬到Gm中
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((int32_t)(aivIndex % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((int32_t)(aivIndex % STAGES));
            auto layoutTileD = params.layoutD.GetTileLayout(MakeCoord(MUbActual, NActual));
            copyUbToGmD(
                dGm[offsetD + aivIndex * maxMPerBlock * params.layoutX.stride(0)],
                ubDTensor[aivIndex % STAGES],
                // ubCTensor[MIdx % STAGES],
                layoutDInGm, layoutTileD
            );
        }else{ // 对着N方向进行切分处理
            // 搬运时需要对齐32Byte的
            uint32_t MActual = actualShape.m();
            uint32_t NActual = actualShape.n(); // 这里也要对齐
            // 行优先部分对M进行切分处理
            uint32_t maxMPerBlock = blockShape.m(); // 对着M方向进行切分 肯定会对齐32Byte
            uint32_t maxNPerBlock = blockShape.n() / STAGES; 
            auto layoutCInUb = layoutC.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutXInUb = params.layoutX.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            auto layoutDInGm = params.layoutD.GetTileLayout(MakeCoord(maxMPerBlock, maxNPerBlock));
            uint32_t aivIndex = AscendC::GetSubBlockIdx();
            uint32_t NActualAIV0 = (NActual < maxNPerBlock) ? NActual : maxNPerBlock;
            uint32_t NActualAIV1 = (NActual < maxNPerBlock) ? 0 : (NActual - maxNPerBlock);
            uint32_t NUbActual = aivIndex ? NActualAIV1 : NActualAIV0;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES));
            auto layoutTileX = params.layoutX.GetTileLayout(MakeCoord(MActual, NUbActual));
            copyGmToUbX(
                ubXTensor[aivIndex % STAGES],
                // xGm[offsetX + aivIndex * maxNPerBlock * params.layoutX.stride(1)],
                xGm[aivIndex * maxNPerBlock * params.layoutX.stride(1)],
                layoutXInUb, layoutTileX
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 4));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 4));
            // 流水控制
            // 进行计算
            // 先进行乘法 alpha * A * B
            tileElemWiseEpilogueMul(
                ubXTensor[aivIndex % STAGES],
                ubXTensor[aivIndex % STAGES],
                params.alpha
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES));

            // 现在单纯测试C矩阵的搬运  AIV核出现问题
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES + 2));
            auto layoutTileC = layoutC.GetTileLayout(MakeCoord(MActual, NUbActual));
            copyGmToUbC(
                ubCTensor[aivIndex % STAGES],
                cGm[offsetC + aivIndex * maxNPerBlock * layoutC.stride(1)],
                layoutCInUb, layoutTileC
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 6));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 6));
            // 进行乘法 beta * C
            tileElemWiseEpilogueMul(
                ubCTensor[aivIndex % STAGES],
                ubCTensor[aivIndex % STAGES],
                params.beta // 获取参数内容
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 2));
            // 等待计算
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((int32_t)(aivIndex % STAGES + 2));
            // 进行加法 alpha * A * B + beta * C
            tileElemWiseEpilogueAdd(
                ubDTensor[aivIndex % STAGES],
                ubXTensor[aivIndex % STAGES],
                ubCTensor[aivIndex % STAGES]
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            // ubX和ubC使用完了
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((int32_t)(aivIndex % STAGES + 2));
            // 流水控制
            // 搬到Gm中
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((int32_t)(aivIndex % STAGES));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((int32_t)(aivIndex % STAGES));
            auto layoutTileD = params.layoutD.GetTileLayout(MakeCoord(MActual, NUbActual));
            copyUbToGmD(
                dGm[offsetD + aivIndex * maxNPerBlock * params.layoutD.stride(1)],
                ubDTensor[aivIndex % STAGES],
                // ubCTensor[aivIndex % STAGES],
                layoutDInGm, layoutTileD
            );
        }
    }
    
};
}

#endif // ACOT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP