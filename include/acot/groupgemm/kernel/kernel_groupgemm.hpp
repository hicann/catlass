#ifndef ACOT_GEMM_KERNEL_GEMM_HPP
#define ACOT_GEMM_KERNEL_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/matmul_coord.hpp"

using namespace acot;

namespace acot::groupgemm::kernel{
// 保持接口统一
template<
    class BlockGemm_,
    class BlockEpilogue_ = void,  // 在后处理阶段进行操作beta alpha操作
    class TileScheduler_ = void
>
class KernelGroupGemm{
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
    static constexpr bool RowOrColumn = std::is_same<LayoutA, acot::layout::RowMajor>::value && std::is_same<LayoutB, acot::layout::RowMajor>::value;
    using TileScheduler = TileScheduler_;

    const uint32_t maxMPerBlock = L1TileShape::M;
    const uint32_t maxNPerBlock = L1TileShape::N;
    const uint32_t cSize = maxMPerBlock * maxNPerBlock;
    const uint32_t l0CBlockNum = ArchTag::L0CSize / (cSize * sizeof(ElementAccumulator));

    typedef struct Params{
        MatmulCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(MatmulCoord problemShape_, GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
                : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
                ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_){}
    }Params;

    ACOT_DEVICE
    KernelGroupGemm(){
        // for(uint32_t i = 0; i < l0CBlockNum; i++){
        //     AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        // }
    }

    ACOT_DEVICE
    ~KernelGroupGemm(){
        // for(uint32_t i = 0; i < l0CBlockNum; i++){
        //     AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        // }
    }

    template<int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params &params){}

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params &params){
        // 先实例化BlockGemm对象
        BlockGemm blockGemm(params.ptrA, params.ptrB, params.ptrC, params.layoutA, params.layoutB, params.layoutC);

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
        for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            MatmulCoord actualShape{MGmActual, NGmActual, K};
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
            // 这里进行特判操作，因为不熟悉coord getoffset的API,而且这个API好像不符合我的要求
            if constexpr (RowOrColumn){
                blockGemm(
                    MGmBlockIdx * params.layoutA.stride(0) * maxMPerBlock,
                    NGmBlockIdx * maxNPerBlock, // 将目前需要转移的数据块的首地址传入就行
                    MGmBlockIdx * params.layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock,
                    actualShape, singleIdx
                );
            }else{
                blockGemm(
                    MGmBlockIdx * maxMPerBlock,
                    NGmBlockIdx * maxNPerBlock * params.layoutB.stride(1), // 将目前需要转移的数据块的首地址传入就行
                    MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * params.layoutC.stride(1), 
                    actualShape, singleIdx
                );
            }
            // AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
            singleIdx += 1;
        }
        #pragma unroll
        for(uint32_t i = 0; i < l0CBlockNum; i++){
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
    } 

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        
    }
};
}

#endif // ACOT_GEMM_KERNEL_GEMM_HPP