#ifndef ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP
#define ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP

#include "acot/acot.hpp"
#include "acot/arch/cross_core_sync.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"

using namespace acot;

namespace acot::gemm::kernel{
// 保持接口统一
template<
    class BlockGemm_,
    class BlockEpilogue_ ,// 在后处理阶段进行操作beta alpha操作
    class TileScheduler_ = void
>
class KernelGemmEpilogue{
public:
    using BlockGemm = BlockGemm_;
    using ArchTag = typename BlockGemm::ArchTag;
    using L1TileShape = typename BlockGemm::L1TileShape;
    using ElementA = typename BlockGemm::ElementA;
    using LayoutA = typename BlockGemm::LayoutA;
    using ElementB = typename BlockGemm::ElementB;
    using LayoutB = typename BlockGemm::LayoutB;
    using ElementC = typename BlockGemm::ElementC;
    using LayoutC = typename BlockGemm::LayoutC;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;

    static constexpr uint32_t STAGES = BlockGemm::STAGES; // 开启双缓冲机制的
    static constexpr bool RowOrColumn = std::is_same<LayoutA, acot::layout::RowMajor>::value && std::is_same<LayoutB, acot::layout::RowMajor>::value;
    using TileScheduler = TileScheduler_;

    typedef struct Params{
        MatmulCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrX;
        LayoutC layoutX;
        EpilogueParams epilogueParams;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(MatmulCoord problemShape_, GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrX_, LayoutC layoutX_, EpilogueParams epilogueParams_)
                : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
                ptrB(ptrB_), layoutB(layoutB_), ptrX(ptrX_), layoutX(layoutX_), epilogueParams(epilogueParams_){}
    }Params;

    ACOT_DEVICE
    KernelGemmEpilogue(){}

    ACOT_DEVICE
    ~KernelGemmEpilogue(){}

    template<int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params &params){}

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params &params){ // 先进行Matmul操作
        // // 先实例化BlockGemm对象
        // BlockGemm blockGemm(params.ptrA, params.ptrB, params.ptrX, params.layoutA, params.layoutB, params.layoutX);

        // uint32_t maxMPerBlock = L1TileShape::M;
        // uint32_t maxNPerBlock = L1TileShape::N;
        // uint32_t M = params.problemShape.m();
        // uint32_t N = params.problemShape.n();
        // uint32_t K = params.problemShape.k();
        // uint32_t cSize = maxMPerBlock * maxNPerBlock;
        // uint32_t l0CBlockNum = ArchTag::L0CSize / (cSize * sizeof(ElementAccumulator));
        // #pragma unroll
        // for(uint32_t i = 0; i < l0CBlockNum; i++){
        //     AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        // }
        // uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        // uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        // uint32_t coreLoops = MLoops * NLoops;
        // uint32_t singleIdx = 0;
        // for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
        //     uint32_t MGmBlockIdx = loopIdx / NLoops;
        //     uint32_t NGmBlockIdx = loopIdx % NLoops;
        //     uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
        //     uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
        //     MatmulCoord actualShape{MGmActual, NGmActual, K};
        //     AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
        //     // 这里进行特判操作，因为不熟悉coord getoffset的API,而且这个API好像不符合我的要求
        //     if constexpr (RowOrColumn){ // 这里需要进行核间流水 使用API  CrossCoreSetFlag  面向分离架构的API
        //         blockGemm(
        //             MGmBlockIdx * params.layoutA.stride(0) * maxMPerBlock,
        //             NGmBlockIdx * maxNPerBlock, // 将目前需要转移的数据块的首地址传入就行
        //             MGmBlockIdx * params.layoutX.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock,
        //             actualShape, singleIdx
        //         );
        //     }else{
        //         blockGemm(
        //             MGmBlockIdx * maxMPerBlock,
        //             NGmBlockIdx * maxNPerBlock * params.layoutB.stride(1), // 将目前需要转移的数据块的首地址传入就行
        //             MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * params.layoutX.stride(1),
        //             actualShape, singleIdx
        //         );
        //     }
        //     // 上面处理完，就放到AIV上进行处理
        //     sync.SetFlag((int8_t)(loopIdx % STAGES));
        //     AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
        //     singleIdx += 1;
        // }
        // #pragma unroll
        // for(uint32_t i = 0; i < l0CBlockNum; i++){
        //     AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        // }
    } 

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        MatmulCoord blockShape = L1TileShape::ToCoord();
        // 先实例化BlockEpilogue对象
        // epilogueParams 里面有X 、C 和 D 对象
        BlockEpilogue blockEpilogue(blockShape, params.epilogueParams); // 传入空间参数
        // AscendC::CrossCoreWaitFlag(0x0); // 开启AIV 这句话有问题
        // 写相应的逻辑
        // 这个时候ptrX对应地方已经有数据了 可以直接进行搬运 还是要进行切分工作

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        uint32_t coreLoops = MLoops * NLoops;
        for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){ // AIV部分 对着M和N进行切分处理 切分方式尽量和AIC切分的方式相同
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            MatmulCoord actualShape{MGmActual, NGmActual, K};
            // sync.WaitFlag((int8_t)(loopIdx % STAGES));
            if constexpr (RowOrColumn){ // 行优先
                blockEpilogue( // 传入偏移量
                    MGmBlockIdx * params.layoutX.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetX
                    MGmBlockIdx * params.layoutX.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetC
                    MGmBlockIdx * params.layoutX.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetD
                    actualShape
                );
            }else{ // 列优先
                blockEpilogue(
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * params.layoutX.stride(1),
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * params.layoutX.stride(1),
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * params.layoutX.stride(1),
                    actualShape
                );
            }
        }
    }
private:
    AscendC::TQueSync<PIPE_M, PIPE_V> sync;
};
}

#endif // ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP