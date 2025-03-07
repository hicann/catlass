#ifndef ACOT_GEMM_KERNEL_GEMM_HPP
#define ACOT_GEMM_KERNEL_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/gemm_shape.hpp"

using namespace acot;

namespace acot::gemm::kernel{
template<
    class BlockGemm_,
    class BlockEpilogue_ = void
>
class KernelGemm{
public:
    using BlockGemm = BlockGemm_;
    using ArchTag = typename BlockGemm::ArchTag;
    using ElementA = typename BlockGemm::ElementA;
    using LayoutA = typename BlockGemm::LayoutA;
    using ElementB = typename BlockGemm::ElementB;
    using LayoutB = typename BlockGemm::LayoutB;
    using ElementC = typename BlockGemm::ElementC;
    using LayoutC = typename BlockGemm::LayoutC;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;

    typedef struct Params{
        GemmShape problemShape;
        L1TileShape l1TileShape;
        GemmShapeStride strideShape;
        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementC* ptrC;
        LayoutC layoutC;

        ACOT_DEVICE
        Params(GemmShape problemShape_, L1TileShape l1TileShape_, GemmShapeStride strideShape_, __gm__ ElementA* ptrA_, LayoutA layoutA_,
            __gm__ ElementB* ptrB_, LayoutB layoutB_, __gm__ ElementC* ptrC_, LayoutC layoutC_)
                : problemShape(problemShape_), l1TileShape(l1TileShape_), strideShape(strideShape_), ptrA(ptrA_), layoutA(layoutA_), 
                ptrB(ptrB_), layoutB(layoutB_),ptrC(ptrC_), layoutC(layoutC_){}
    }Params;

    ACOT_DEVICE
    KernelGemm(){}

    template<int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params &params){}

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params &params){
        // 初始化全局内存
        // 这个部分不需要修改
        BlockGemm blockGemm(params.ptrA, params.ptrB, params.ptrC);

        uint32_t M = params.problemShape.M;
        uint32_t N = params.problemShape.N;
        uint32_t K = params.problemShape.K;
        uint32_t maxMPerBlock = params.l1TileShape.l1MaxM;
        uint32_t maxNPerBlock = params.l1TileShape.l1MaxN;
        uint32_t maxKPerBlock = params.l1TileShape.l1MaxK;
        uint32_t strideA = params.strideShape.strideA;
        uint32_t strideB = params.strideShape.strideB;
        uint32_t strideC = params.strideShape.strideC;

        // 开启流水
        uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
        uint32_t l0CBlockNum = ArchTag::L0CSize / cSize;
        for(uint32_t i = 0; i < l0CBlockNum; i++){
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }

        // 行优先部分
        if constexpr (std::is_same<LayoutA, acot::layout::RowMajor>::value && std::is_same<LayoutB, acot::layout::RowMajor>::value){
            // 已经集中了4种数据类型的判断了
            uint32_t MGmAlignment = C0_NUM_PER_FRACTAL;
            uint32_t NGmAlignment = BYTE_PER_C0 / sizeof(ElementB);
            if constexpr (std::is_same<ElementB, float>::value){
                NGmAlignment = C0_NUM_PER_FRACTAL; // N方向向16对齐
            }
            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t NLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = MLoops * NLoops;
            uint32_t singleIdx = 0;
            // 这边一定是M方向和N方向的切分操作
            for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
                uint32_t MGmBlockIdx = loopIdx / NLoops;
                uint32_t NGmBlockIdx = loopIdx % NLoops;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                uint32_t MGmRound = RoundUp(MGmActual, MGmAlignment);
                uint32_t NGmRound = RoundUp(NGmActual, NGmAlignment);
                Gemm_Kernel2Block_Params KernelParams;
                KernelParams.MActual = MGmActual;
                KernelParams.NActual = NGmActual;
                KernelParams.MRound = MGmRound;
                KernelParams.NRound = NGmRound;
                KernelParams.K = K;
                KernelParams.strideA = strideA;
                KernelParams.strideB = strideB;
                KernelParams.strideC = strideC;
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                blockGemm(
                    MGmBlockIdx * strideA * maxMPerBlock,
                    NGmBlockIdx * maxNPerBlock, // 将目前需要转移的数据块的首地址传入就行
                    MGmBlockIdx * strideC * maxMPerBlock  + NGmBlockIdx * maxNPerBlock,
                    KernelParams, singleIdx
                );
                singleIdx++;
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
            }
        // 列优先部分
        }else if constexpr (std::is_same<LayoutA, acot::layout::ColumnMajor>::value && std::is_same<LayoutB, acot::layout::ColumnMajor>::value){
            
        }
    }

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        
    }
};
}

#endif // ACOT_GEMM_KERNEL_GEMM_HPP