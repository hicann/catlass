#ifndef ACOT_GEMM_KERNEL_GEMM_HPP
#define ACOT_GEMM_KERNEL_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/arch/resource.hpp"
#include "acot/coord.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"

using namespace acot;

namespace acot::gemm::kernel{
// 保持接口统一
template<
    class BlockGemm_,
    class BlockEpilogue_ = void,  // 在后处理阶段进行操作beta alpha操作
    class TileScheduler_ = void
>
class KernelGemm{
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
    const uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;

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
    KernelGemm(){}

    ACOT_DEVICE
    ~KernelGemm(){}

    template<int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params &params){}

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params &params){
        // 先实例化BlockGemm对象
        arch::Resource<ArchTag> resource;
        BlockGemm blockGemm(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
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
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(singleIdx % l0CBlockNum));
            if constexpr (RowOrColumn){
                blockGemm(
                    gmA[MGmBlockIdx * params.layoutA.stride(0) * maxMPerBlock], params.layoutA,
                    gmB[NGmBlockIdx * maxNPerBlock], params.layoutB,
                    gmC[MGmBlockIdx * params.layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock], params.layoutC,
                    actualShape, singleIdx % l0CBlockNum
                );
            }else{
                blockGemm(
                    gmA[MGmBlockIdx * maxMPerBlock], params.layoutA,
                    gmB[NGmBlockIdx * maxNPerBlock * params.layoutB.stride(1)], params.layoutB,
                    gmC[MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * params.layoutC.stride(1)], params.layoutC,
                    actualShape, singleIdx % l0CBlockNum
                );
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(singleIdx % l0CBlockNum));
            singleIdx++;
        }
        #pragma unroll
        for(uint32_t i = 0; i < l0CBlockNum; i++){
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
    } 

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params &params){}
};
}

#endif // ACOT_GEMM_KERNEL_GEMM_HPP