#ifndef ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP
#define ACOT_GEMM_KERNEL_GEMM_EPILOGUE_HPP

#include "acot/acot.hpp"
#include "acot/arch/cross_core_sync.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"

using namespace acot;

namespace acot::gemm::kernel{

    namespace detail {

        template <class T>
        ACOT_DEVICE
        void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len)
        {
            for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
                reinterpret_cast<uint64_t *>(dst)[i] = reinterpret_cast<__gm__ uint64_t *>(src)[i];
            }
        }
        
        }  // namespace detail

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
    using ElementX = typename BlockGemm::ElementC;
    using LayoutC = typename BlockGemm::LayoutC;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;
    static constexpr uint32_t MAX_TENSOR_COUNT = 256;

    static constexpr uint32_t STAGES = BlockGemm::STAGES; // 开启双缓冲机制的
    static constexpr bool RowOrColumn = std::is_same<LayoutA, acot::layout::RowMajor>::value && std::is_same<LayoutB, acot::layout::RowMajor>::value;
    using TileScheduler = TileScheduler_;

    typedef struct Params{
        // Data members
        uint32_t problemCount;
        GM_ADDR ptrProblemShape;
        GM_ADDR alpha;
        GM_ADDR beta;
        GM_ADDR ptrA;
        GM_ADDR ptrLayoutA;
        GM_ADDR ptrB;
        GM_ADDR ptrLayoutB;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrLayoutWorkspace;
        EpilogueParams epilogueParams;
        

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(
            uint32_t problemCount_, GM_ADDR ptrProblemShape_,
            GM_ADDR alpha_, GM_ADDR beta_,
            GM_ADDR ptrA_, GM_ADDR ptrLayoutA_,
            GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrLayoutWorkspace_, EpilogueParams epilogueParams_)
                :problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
                alpha(alpha_), beta(beta_),
                ptrA(ptrA_), ptrLayoutA(ptrLayoutA_),
                ptrB(ptrB_), ptrLayoutB(ptrLayoutB_),
                ptrWorkspace(ptrWorkspace_), ptrLayoutWorkspace(ptrLayoutWorkspace_), epilogueParams(epilogueParams_){}
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
        MatmulCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutWorkspaceList[MAX_TENSOR_COUNT];

        // Get matmul information from parameters
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutWorkspaceList, params.ptrLayoutWorkspace, params.problemCount);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t inGroupOffsetA = 0;
        int64_t inGroupOffsetB = 0;
        int64_t inGroupOffsetWorkspace = 0;

        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx ;
        

        uint32_t maxMPerBlock = L1TileShape::M;
        uint32_t maxNPerBlock = L1TileShape::N;
        uint32_t cSize = maxMPerBlock * maxNPerBlock;
        uint32_t l0CBlockNum = ArchTag::L0CSize / (cSize * sizeof(ElementAccumulator));

        
        uint32_t singleIdx = 0;
        for(uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx){
            // // 先实例化BlockGemm对象
            MatmulCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutWorkspace = layoutWorkspaceList[groupIdx];

            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();

            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t NLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = MLoops * NLoops;

            // Determine the starting loopIdx of the current core under the current groupIdx
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            
            #pragma unroll
            for(uint32_t i = 0; i < l0CBlockNum; i++){
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
            
            if constexpr (RowOrColumn){
                BlockGemm blockGemm(params.ptrA, params.ptrB, params.ptrWorkspace, layoutA, layoutB, layoutWorkspace);
                for(uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum){
                    uint32_t MGmBlockIdx = loopIdx / NLoops;
                    uint32_t NGmBlockIdx = loopIdx % NLoops;
                    uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                    uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                    MatmulCoord actualShape{MGmActual, NGmActual, K};
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                    // 这里进行特判操作，因为不熟悉coord getoffset的API,而且这个API好像不符合我的要求
                    blockGemm(
                        inGroupOffsetA+MGmBlockIdx * layoutA.stride(0) * maxMPerBlock,
                        inGroupOffsetB+NGmBlockIdx * maxNPerBlock, // 将目前需要转移的数据块的首地址传入就行
                        inGroupOffsetWorkspace+MGmBlockIdx * layoutWorkspace.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock,
                        actualShape, singleIdx
                    );
                    // 上面处理完，就放到AIV上进行处理
                    // sync.SetFlag((int8_t)(loopIdx % STAGES));
                    arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                    singleIdx += 1;
                    // AscendC::PipeBarrier<PIPE_ALL>();
                }
            }else{
                BlockGemm blockGemm(params.ptrA, params.ptrB, params.ptrWorkspace, layoutA, layoutB, layoutWorkspace);
                for(uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
                    uint32_t MGmBlockIdx = loopIdx / NLoops;
                    uint32_t NGmBlockIdx = loopIdx % NLoops;
                    uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                    uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                    MatmulCoord actualShape{MGmActual, NGmActual, K};
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                    // 这里进行特判操作，因为不熟悉coord getoffset的API,而且这个API好像不符合我的要求
                    blockGemm(
                        inGroupOffsetA+MGmBlockIdx * maxMPerBlock,
                        inGroupOffsetB+NGmBlockIdx * maxNPerBlock * layoutB.stride(1), // 将目前需要转移的数据块的首地址传入就行
                        inGroupOffsetWorkspace+MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutWorkspace.stride(1),
                        actualShape, singleIdx
                    );
                    // 上面处理完，就放到AIV上进行处理
                    // sync.SetFlag((int8_t)(loopIdx % STAGES));
                    arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)(loopIdx % l0CBlockNum));
                    singleIdx += 1;
                    // AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            inGroupOffsetA += problemShape.m() * problemShape.k();
            inGroupOffsetB += problemShape.k() * problemShape.n();
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            
            #pragma unroll
            for(uint32_t i = 0; i < l0CBlockNum; i++){
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
        }
        // #pragma unroll
        // for(uint32_t i = 0; i < l0CBlockNum; i++){
        //     AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        // }
    } 

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        MatmulCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutWorkspaceList[MAX_TENSOR_COUNT];
        ElementAccumulator alphaList[MAX_TENSOR_COUNT];
        ElementAccumulator betaList[MAX_TENSOR_COUNT];

        // Get matmul information from parameters
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutWorkspaceList, params.ptrLayoutWorkspace, params.problemCount);
        detail::UnpackListParam(alphaList, params.alpha, params.problemCount);
        detail::UnpackListParam(betaList, params.beta, params.problemCount);
        
        //存疑
        uint32_t coreIdx = AscendC::GetBlockIdx()/2;
        uint32_t coreNum = AscendC::GetBlockNum();

        int64_t inGroupOffsetWorkspace = 0;

        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;

        MatmulCoord blockShape = L1TileShape::ToCoord();
        // 先实例化BlockEpilogue对象
        // epilogueParams 里面有X 、C 和 D 对象
        for(uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx){
            // // 先实例化BlockGemm对象
            MatmulCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutWorkspace = layoutWorkspaceList[groupIdx];
            ElementAccumulator alpha_ = alphaList[groupIdx];
            ElementAccumulator beta_ = betaList[groupIdx];

            uint32_t maxMPerBlock = L1TileShape::M;
            uint32_t maxNPerBlock = L1TileShape::N;

            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            // 这个时候ptrX对应地方已经有数据了 可以直接进行搬运 还是要进行切分工作
            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t NLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = MLoops * NLoops;

            // Determine the starting loopIdx of the current core under the current groupIdx
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            params.epilogueParams.layoutC = layoutWorkspace;
            params.epilogueParams.layoutD = layoutWorkspace;
            params.epilogueParams.alpha = alpha_;
            params.epilogueParams.beta = beta_;
            
            BlockEpilogue blockEpilogue(blockShape, params.epilogueParams); // 传入空间参数
        
            // 获取是AIC核的位置
            AscendC::GlobalTensor<ElementX> gmX;
            gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrWorkspace);
            //存疑
            for(uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){ // 换一下切分方式，一个AIC对应两个AIV核 blockNum是AIV核数 
                uint32_t MGmBlockIdx = loopIdx / NLoops;
                uint32_t NGmBlockIdx = loopIdx % NLoops;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                MatmulCoord actualShape{MGmActual, NGmActual, K};
                // sync.WaitFlag((int8_t)(loopIdx % STAGES));
                arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
                // AscendC::PRINTF("asdasd");
                if constexpr (RowOrColumn){ // 行优先  这个现在正确率高
                    // layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
                    uint32_t inLoopOffset = MGmBlockIdx * layoutWorkspace.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock;
                    blockEpilogue( // 传入偏移量
                        inGroupOffsetWorkspace+inLoopOffset, // offsetC
                        // MGmBlockIdx * layoutC.stride(0) * maxMPerBlock  + NGmBlockIdx * maxNPerBlock, // offsetX
                        inGroupOffsetWorkspace+inLoopOffset, // offsetD
                        gmX[inGroupOffsetWorkspace+inLoopOffset],
                        layoutWorkspace,
                        actualShape
                    );
                }else{ // 列优先
                    // layout::ColumnMajor layoutC(params.problemShape.m(), params.problemShape.n());
                    uint32_t inLoopOffset = MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutWorkspace.stride(1);
                    blockEpilogue(
                        inGroupOffsetWorkspace+inLoopOffset,
                        // MGmBlockIdx * maxMPerBlock + NGmBlockIdx * maxNPerBlock * layoutC.stride(1),
                        inGroupOffsetWorkspace+inLoopOffset,
                        gmX[inGroupOffsetWorkspace+inLoopOffset],
                        layoutWorkspace,
                        actualShape
                    );
                }
                // AscendC::PipeBarrier<PIPE_ALL>();
            }
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
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