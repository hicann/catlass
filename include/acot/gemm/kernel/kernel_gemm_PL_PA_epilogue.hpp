#ifndef ACOT_GEMM_KERNEL_GEMM_PL_PA_EPILOGUE_HPP
#define ACOT_GEMM_KERNEL_GEMM_PL_PA_EPILOGUE_HPP

#include "acot/acot.hpp"
#include "acot/arch/cross_core_sync.hpp"
#include "acot/matmul_coord.hpp"
#include "acot/matrix_coord.hpp"
#include "acot/epilogue/tile/copy_gm_to_ub.hpp"
#include "acot/epilogue/tile/copy_ub_to_gm.hpp"
#include "acot/gemm/helper.hpp"

using namespace acot;

namespace acot::gemm::kernel{
// 进行padding操作
template<
    class ArchTag_,
    class Element_,
    class Layout_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrix {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using Layout = Layout_;
    using CopyGm2Ub = acot::epilogue::tile::CopyGm2Ub<
        ArchTag, matmul::MatmulType<Element, acot::layout::RowMajor>>;
    using CopyUb2Gm = acot::epilogue::tile::CopyUb2Gm<
        ArchTag, matmul::MatmulType<Element, acot::layout::RowMajor>>;
    using ComputeLayout = acot::layout::RowMajor; // 都是RowMajor处理

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    ACOT_DEVICE
    PaddingMatrix(arch::Resource<ArchTag> &resource){
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) { // 
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element)); // ubBuf是int8_t的内容
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    ACOT_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout){ // 最后都是RowMajor 
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0)); // Row Column stride
    }

    ACOT_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout){ // 最后都是RowMajor
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1)); // Column Row stride  相当于进行了转置处理
    }

    ACOT_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    Layout layoutDst, Layout layoutSrc
    ){
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum(); // 20 * 2 = 40
        uint32_t aivId = AscendC::GetBlockIdx(); // 获取当前AIV的核心号 0 ~ 39

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0); // Row
        uint32_t tileLen = computeLayoutSrc.shape(1); // Col
        uint32_t paddingStride = computeLayoutDst.stride(0); // 目标padding stride

        uint32_t tilesPerAiv = tilesNum / aivNum; // 方便批次处理
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++; // 前面的核多处理一行
        }
        uint32_t mIdx = aivId * tilesPerAiv; // 第几轮
        if (aivId >= tileRemain) {
            mIdx += tileRemain; // 均分任务
        }
        MatrixCoord blockOffset(mIdx, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]); // 相当于构造函数中
        uint32_t coreLoops{ 0 };
        if (paddingStride > COMPUTE_LENGTH) { // 一次处理不了 只能一行一行处理
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH); // 处理轮次 横向切割
            coreLoops = tilesPerAiv * loopsPerTile;  // 总轮次数
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) { // 相当于处理到了一行的最后一个
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum)); // row column 
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;
                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                bufferIndex = 1 - bufferIndex;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / paddingStride; // 可以同时处理几批次
            coreLoops = CeilDiv(tilesPerAiv, tilesPerLoop);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) { // 可以多次处理内容
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) { // 最后一次太够了，减少
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen)); // Row column
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout &ubLayout = dstLayout;
                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                bufferIndex = 1 - bufferIndex;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]); // 相当于析构函数中
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    ACOT_DEVICE
    ~PaddingMatrix() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

// 保持接口统一
template<
    class BlockGemm_,
    class BlockEpilogue_ ,
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
    using ElementX = typename BlockGemm::ElementX;
    using LayoutX = typename BlockGemm::LayoutX;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementC = typename BlockEpilogue::ElementC;
    using ElementD = typename BlockEpilogue::ElementD;

    const uint32_t maxMPerBlock = L1TileShape::M;
    const uint32_t maxNPerBlock = L1TileShape::N;
    const uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
    const uint32_t l0XBlockNum = ArchTag::L0C_SIZE / cSize;
    using ElementCompute =
        typename acot::gemm::helper::ElementAccumulatorSelector<ElementC, ElementD>::ElementAccumulator;
    using ElementScalar = ElementCompute; // 标量的数据类型
    static constexpr uint32_t STAGES = BlockGemm::STAGES; // 开启双缓冲机制的
    // static constexpr bool RowOrColumn = std::is_same<LayoutA, acot::layout::RowMajor>::value && std::is_same<LayoutB, acot::layout::RowMajor>::value;
    using TileScheduler = TileScheduler_;
    
    // 进行Padding操作
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA); // UB_SIZE 192 * 1024 / 2 的内容 对半分
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    typedef struct Params{
        MatmulCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR gmWorkspace;
        GM_ADDR ptrWA;
        LayoutA layoutWA; // padding
        GM_ADDR ptrWB;
        LayoutB layoutWB;
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrC;
        GM_ADDR ptrD;

        // Methods
        ACOT_DEVICE
        Params() {}

        ACOT_DEVICE
        Params(MatmulCoord problemShape_, GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR gmWorkspace_, 
            GM_ADDR ptrWA_, LayoutA layoutWA_, GM_ADDR ptrWB_, LayoutB layoutWB_, 
            ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrC_, GM_ADDR ptrD_)
                : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
                ptrB(ptrB_), layoutB(layoutB_), gmWorkspace(gmWorkspace_), 
                ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_), 
                alpha(alpha_), beta(beta_), ptrC(ptrC_), ptrD(ptrD_){}
    }Params;

    ACOT_DEVICE
    KernelGemmEpilogue(){}

    ACOT_DEVICE
    ~KernelGemmEpilogue(){}
    
    // 比较两者的步长
    ACOT_DEVICE
    bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
    {
        return layout1.stride(0) == layout2.stride(0);
    }
    ACOT_DEVICE
    bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
    {
        return layout1.stride(1) == layout2.stride(1);
    }

    template<int32_t CORE_TYPE = g_coreType>
    ACOT_DEVICE
    void operator()(Params &params){}

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIC>(Params &params){
        // 等待Padding操作 只padding stride padding操作没问题
        if (!IsSameStride(params.layoutWA, params.layoutA) || !IsSameStride(params.layoutWB, params.layoutB)) {
            arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }
        // 先实例化BlockGemm对象
        arch::Resource<ArchTag> resource;
        BlockGemm blockGemm(resource);
        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.gmWorkspace);
        uint32_t M = params.problemShape.m(); // 这些参数都没有进行padding操作，只有stride进行padding操作
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        #pragma unroll
        for(uint32_t i = 0; i < l0XBlockNum; i++){
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        uint32_t coreLoops = MLoops * NLoops;
        uint32_t singleIdx = 0;
        LayoutX layoutX(params.problemShape.m(), params.problemShape.n());
        for(uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            MatmulCoord nextActualShape;
            uint32_t MNextGmBlockIdx = 0; uint32_t NNextGmBlockIdx = 0;
            if(loopIdx + AscendC::GetBlockNum() < coreLoops){
                hasNextBlock = true;
                uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                MNextGmBlockIdx = nextLoopIdx / NLoops;
                NNextGmBlockIdx = nextLoopIdx % NLoops;
                uint32_t MNextGmActual = (MNextGmBlockIdx == MLoops - 1) ? (M - MNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NNextGmActual = (NNextGmBlockIdx == NLoops - 1) ? (N - NNextGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                nextActualShape = MakeCoord(MNextGmActual, NNextGmActual, K); // 构建下一次的形状
            }
            MatmulCoord actualShape{MGmActual, NGmActual, K};
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
            MatrixCoord gmTileAOffset{MGmBlockIdx * maxMPerBlock, 0}; // gm中的偏移量
            auto gmTileA = gmA[params.layoutWA.GetOffset(gmTileAOffset)];
            MatrixCoord gmTileBOffset{0, NGmBlockIdx * maxNPerBlock}; // gm中的偏移量
            auto gmTileB = gmB[params.layoutWB.GetOffset(gmTileBOffset)];
            MatrixCoord gmTileXOffset{MGmBlockIdx * maxMPerBlock, NGmBlockIdx * maxNPerBlock}; // gm中的偏移量
            auto gmTileX = gmX[layoutX.GetOffset(gmTileXOffset)];
            MatrixCoord gmTileNextAOffset{MNextGmBlockIdx * maxMPerBlock, 0}; // gm中的偏移量
            auto gmTileNextA = gmA[params.layoutWA.GetOffset(gmTileNextAOffset)];
            MatrixCoord gmTileNextBOffset{0, NNextGmBlockIdx * maxNPerBlock}; // gm中的偏移量
            auto gmTileNextB = gmB[params.layoutWB.GetOffset(gmTileNextBOffset)];
            blockGemm(
                gmTileA, params.layoutWA, // row col stride不一样了
                gmTileB, params.layoutWB,
                gmTileX, layoutX,
                gmTileNextA, gmTileNextB,
                actualShape, nextActualShape, isFirstBlock, hasNextBlock, singleIdx
            );
            arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
            singleIdx = (singleIdx + 1) % l0XBlockNum;
        }
        #pragma unroll
        for(uint32_t i = 0; i < l0XBlockNum; i++){
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
        }
    } 

    template<>
    ACOT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        arch::Resource<ArchTag> resource;
        // 首先进行Padding操作
        if (!IsSameStride(params.layoutWA, params.layoutA)) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA); // 两个AIV核
        }

        if (!IsSameStride(params.layoutWB, params.layoutB)) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if (!IsSameStride(params.layoutWA, params.layoutA) || !IsSameStride(params.layoutWB, params.layoutB)) {
            acot::arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            acot::arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
        // 后进行后处理过程
        MatmulCoord blockShape = L1TileShape::ToCoord();
        BlockEpilogue blockEpilogue(resource, blockShape);
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        uint32_t MLoops = CeilDiv(M, maxMPerBlock);
        uint32_t NLoops = CeilDiv(N, maxNPerBlock);
        uint32_t coreLoops = MLoops * NLoops;
        // 获取是AIC核的位置
        uint32_t aivNum = AscendC::GetSubBlockNum();
        uint32_t aivIndex = AscendC::GetBlockIdx();
        uint32_t aicoreIndex = aivIndex / aivNum;
        AscendC::GlobalTensor<ElementX> gmX; // fp32
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.gmWorkspace);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer((__gm__ ElementD*)params.ptrD);
        for(uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
            uint32_t MGmBlockIdx = loopIdx / NLoops;
            uint32_t NGmBlockIdx = loopIdx % NLoops;
            uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
            uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
            MatmulCoord actualShape{MGmActual, NGmActual, K};
            LayoutX layoutX(params.problemShape.m(), params.problemShape.n());
            arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore); // 没这个变量
            MatrixCoord gmTileOffset{MGmBlockIdx * maxMPerBlock, NGmBlockIdx * maxNPerBlock}; // gm中的偏移量
            auto offsetX = layoutX.GetOffset(gmTileOffset);
            blockEpilogue( // 传入偏移量
                params.alpha, params.beta,
                gmC[offsetX], layoutX,
                gmD[offsetX], layoutX,
                gmX[offsetX], layoutX,
                actualShape
            );
        }
    }
private:
    // AIC同步
    static constexpr arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    // AIV同步
    static constexpr arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};
}

#endif // ACOT_GEMM_KERNEL_GEMM_PL_PA_EPILOGUE_HPP