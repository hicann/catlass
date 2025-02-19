#ifndef ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP
#define ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP

#include "acot/acot.hpp"
#include "acot/arch/cross_core_sync.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"

using namespace acot;

namespace acot::groupgemm::kernel{
// 保持接口统一
template<
    class BlockGemm_,
    class BlockEpilogue_ ,// 在后处理阶段进行操作beta alpha操作
    class TileScheduler_ = void
>
class KernelGroupGemmEpilogue{
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
        GM_ADDR gmWorkspace;
        EpilogueParams epilogueParams;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(MatmulCoord problemShape_, GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR gmWorkspace_, EpilogueParams epilogueParams_)
                : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
                ptrB(ptrB_), layoutB(layoutB_), gmWorkspace(gmWorkspace_), epilogueParams(epilogueParams_){}
    }Params;

    ACOT_DEVICE
    KernelGroupGemmEpilogue(){}

    ACOT_DEVICE
    ~KernelGroupGemmEpilogue(){}

    template<int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params &params){}

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params &params){ // 先进行Matmul操作
        // // 先实例化BlockGemm对象
        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        uint32_t cSize = maxMPerBlock * maxNPerBlock;
        uint32_t l0CBlockNum = ArchTag::L0CSize / (cSize * sizeof(ElementAccumulator));
        #pragma unroll
        for(uint32_t i = 0; i < l0CBlockNum; i++){
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        uint32_t coreLoops = MLoops * NLoops;
        uint32_t singleIdx = 0;
        if constexpr (RowOrColumn){
            layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
            BlockGemm blockGemm(params.ptrA, params.ptrB, params.gmWorkspace, params.layoutA, params.layoutB, layoutC);
            for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
                uint32_t MGmBlockIdx = loopIdx / NLoops;
                uint32_t NGmBlockIdx = loopIdx % NLoops;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                MatmulCoord actualShape{MGmActual, NGmActual, K};
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                // 这里进行特判操作，因为不熟悉coord getoffset的API,而且这个API好像不符合我的要求
                blockGemm(
                    MGmBlockIdx * params.layoutA.stride(0) * maxMPerBlock,
                    NGmBlockIdx * maxNPerBlock, // 将目前需要转移的数据块的首地址传入就行
                    MGmBlockIdx * layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock,
                    actualShape, singleIdx
                );
                // 上面处理完，就放到AIV上进行处理
                // sync.SetFlag((int8_t)(loopIdx % STAGES));
                arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                singleIdx += 1;
            }
        }else{
            layout::ColumnMajor layoutC(params.problemShape.m(), params.problemShape.n());
            BlockGemm blockGemm(params.ptrA, params.ptrB, params.gmWorkspace, params.layoutA, params.layoutB, layoutC);
            for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
                uint32_t MGmBlockIdx = loopIdx / NLoops;
                uint32_t NGmBlockIdx = loopIdx % NLoops;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                MatmulCoord actualShape{MGmActual, NGmActual, K};
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                // 这里进行特判操作，因为不熟悉coord getoffset的API,而且这个API好像不符合我的要求
                blockGemm(
                    MGmBlockIdx * maxMPerBlock,
                    NGmBlockIdx * maxNPerBlock * params.layoutB.stride(1), // 将目前需要转移的数据块的首地址传入就行
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutC.stride(1),
                    actualShape, singleIdx
                );
                // 上面处理完，就放到AIV上进行处理
                // sync.SetFlag((int8_t)(loopIdx % STAGES));
                arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                singleIdx += 1;
            }
        }
        #pragma unroll
        for(uint32_t i = 0; i < l0CBlockNum; i++){
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
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
        // 获取是AIC核的位置
        uint32_t aivNum = AscendC::GetSubBlockNum(); // 910B3 AIV核为2
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / aivNum;
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.gmWorkspace);
        for(uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){ // 换一下切分方式，一个AIC对应两个AIV核 blockNum是AIV核数 
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            MatmulCoord actualShape{MGmActual, NGmActual, K};
            // sync.WaitFlag((int8_t)(loopIdx % STAGES));
            arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
            if constexpr (RowOrColumn){ // 行优先  这个现在正确率高
                layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
                blockEpilogue( // 传入偏移量
                    MGmBlockIdx * layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetC
                    // MGmBlockIdx * layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetX
                    MGmBlockIdx * layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetD
                    gmC[MGmBlockIdx * layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock],
                    layoutC,
                    actualShape
                );
            }else{ // 列优先
                layout::ColumnMajor layoutC(params.problemShape.m(), params.problemShape.n());
                blockEpilogue(
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutC.stride(1),
                    // MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutC.stride(1),
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutC.stride(1),
                    gmC[MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutC.stride(1)],
                    layoutC,
                    actualShape
                );
            }
        }
    }
private:
    // AscendC::TQueSync<PIPE_M, PIPE_V> sync;
    static constexpr arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    // arch::Resource<ArchTag> resource;
};
}

#endif // ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP