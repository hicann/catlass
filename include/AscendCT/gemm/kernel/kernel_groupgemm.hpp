/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDCT_GROUPGEMM_KERNEL_GROUPGEMM_HPP
#define ASCENDCT_GROUPGEMM_KERNEL_GROUPGEMM_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/cross_core_sync.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/matrix_coord.hpp"
#include "AscendCT/epilogue/tile/copy_gm_to_ub.hpp"
#include "AscendCT/epilogue/tile/copy_ub_to_gm.hpp"
#include "AscendCT/gemm/helper.hpp"

using namespace AscendCT;

namespace AscendCT::gemm::kernel{

    namespace detail {

        template <class T>
        ASCENDCT_DEVICE
        void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len)
        {
            for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
                reinterpret_cast<uint64_t *>(dst)[i] = reinterpret_cast<__gm__ uint64_t *>(src)[i];
            }
        }
        
    }  // namespace detail

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
    using CopyGm2Ub = AscendCT::epilogue::tile::CopyGm2Ub<
        ArchTag, gemm::MatmulType<Element, AscendCT::layout::RowMajor>>;
    using CopyUb2Gm = AscendCT::epilogue::tile::CopyUb2Gm<
        ArchTag, gemm::MatmulType<Element, AscendCT::layout::RowMajor>>;
    using ComputeLayout = AscendCT::layout::RowMajor; // 都是RowMajor处理

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    ASCENDCT_DEVICE
    PaddingMatrix(arch::Resource<ArchTag> &resource){
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) { // 
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element)); // ubBuf是int8_t的内容
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    ASCENDCT_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout){ // 最后都是RowMajor 
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0)); // Row Column stride
    }

    ASCENDCT_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout){ // 最后都是RowMajor
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1)); // Column Row stride  相当于进行了转置处理
    }

    ASCENDCT_DEVICE
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

    ASCENDCT_DEVICE
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
    using ElementX = typename BlockGemm::ElementX;
    using LayoutX = typename BlockGemm::LayoutX;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;
    
    using BlockEpilogue = BlockEpilogue_;
    // using EpilogueParams = typename BlockEpilogue::Params;
    using ElementC = typename BlockEpilogue::ElementC;
    using ElementD = typename BlockEpilogue::ElementD;
    using ElementCompute =
        typename AscendCT::gemm::helper::ElementAccumulatorSelector<ElementC, ElementD>::ElementAccumulator;
    using ElementScalar = ElementCompute; // 标量的数据类型
    static constexpr uint32_t MAX_TENSOR_COUNT = 32;

    const uint32_t maxMPerBlock = L1TileShape::M;
    const uint32_t maxNPerBlock = L1TileShape::N;
    const uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;

    static constexpr uint32_t STAGES = BlockGemm::STAGES; // 开启双缓冲机制的
    // static constexpr bool RowOrColumn = std::is_same<LayoutA, AscendCT::layout::RowMajor>::value && std::is_same<LayoutB, AscendCT::layout::RowMajor>::value;
    using TileScheduler = TileScheduler_;
    
    // 进行Padding操作
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA); // UB_SIZE 192 * 1024 / 2 的内容 对半分
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

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
        GM_ADDR ptrWA;
        GM_ADDR ptrlayoutWA; // padding
        GM_ADDR ptrWB;
        GM_ADDR ptrlayoutWB;
        GM_ADDR ptrC;
        GM_ADDR ptrD;
        

        // Methods
        ASCENDCT_DEVICE
        Params() {}

        ASCENDCT_DEVICE
        Params(
            uint32_t problemCount_, GM_ADDR ptrProblemShape_,
            GM_ADDR alpha_, GM_ADDR beta_,
            GM_ADDR ptrA_, GM_ADDR ptrLayoutA_,
            GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrLayoutWorkspace_,
            GM_ADDR ptrWA_,GM_ADDR ptrlayoutWA_, 
            GM_ADDR ptrWB_,GM_ADDR ptrlayoutWB_, GM_ADDR ptrC_, GM_ADDR ptrD_)
                :problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
                alpha(alpha_), beta(beta_),
                ptrA(ptrA_), ptrLayoutA(ptrLayoutA_),
                ptrB(ptrB_), ptrLayoutB(ptrLayoutB_),
                ptrWorkspace(ptrWorkspace_), ptrLayoutWorkspace(ptrLayoutWorkspace_), 
                ptrWA(ptrWA_), ptrlayoutWA(ptrlayoutWA_), 
                ptrWB(ptrWB_), ptrlayoutWB(ptrlayoutWB_),ptrC(ptrC_), ptrD(ptrD_){}
    }Params;

    ASCENDCT_DEVICE
    KernelGroupGemmEpilogue(){

    }

    ASCENDCT_DEVICE
    ~KernelGroupGemmEpilogue(){
        
    }

    // 比较两者的步长
    ASCENDCT_DEVICE
    bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
    {
        return layout1.stride(0) == layout2.stride(0);
    }
    ASCENDCT_DEVICE
    bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
    {
        return layout1.stride(1) == layout2.stride(1);
    }
    ASCENDCT_DEVICE
    size_t GetWorkspaceLen(layout::RowMajor layout)
    {
        return layout.shape(0) * layout.stride(0);
    }
    ASCENDCT_DEVICE
    size_t GetWorkspaceLen(layout::ColumnMajor layout)
    {
        return layout.shape(1) * layout.stride(1);
    }

    template<int32_t CORE_TYPE = g_coreType>
    ASCENDCT_DEVICE
    void operator()(Params &params){}

    template<>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIC>(Params &params){
        MatmulCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutX layoutWorkspaceList[MAX_TENSOR_COUNT];
        //增加WA和WB
        LayoutA layoutWAList[MAX_TENSOR_COUNT];
        LayoutB layoutWBList[MAX_TENSOR_COUNT];

        // Get matmul information from parameters
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutWorkspaceList, params.ptrLayoutWorkspace, params.problemCount);
        //增加WA和WB
        detail::UnpackListParam(layoutWAList, params.ptrlayoutWA, params.problemCount);
        detail::UnpackListParam(layoutWBList, params.ptrlayoutWB, params.problemCount);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint64_t inGroupOffsetA = 0;
        uint64_t inGroupOffsetB = 0;
        uint64_t inGroupOffsetWorkspace = 0;

        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;
        
        arch::Resource<ArchTag> resource;
        BlockGemm blockGemm(resource);
        
        for(uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx){
            // // 先实例化BlockGemm对象
            MatmulCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutX layoutWorkspace = layoutWorkspaceList[groupIdx];
            //增加WA和WB
            LayoutA layoutWA = layoutWAList[groupIdx];
            LayoutB layoutWB = layoutWBList[groupIdx];
            // 等待Padding操作 只padding stride padding操作没问题
            if (!IsSameStride(layoutWA, layoutA) || !IsSameStride(layoutWB, layoutB)) {
                arch::CrossCoreWaitFlag(flagAivFinishPadding);
            }
            // 先实例化BlockGemm对象
            AscendC::GlobalTensor<ElementX> gmX;
            gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrWorkspace);
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrWA);
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            #pragma unroll
            for(uint32_t i = 0; i < l0CBlockNum; i++){
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }

            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t NLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = MLoops * NLoops;

            // Determine the starting loopIdx of the current core under the current groupIdx
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            uint32_t singleIdx = 0;
            for(uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
                uint32_t MGmBlockIdx = loopIdx / NLoops;
                uint32_t NGmBlockIdx = loopIdx % NLoops;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                bool isFirstBlock = (loopIdx == startLoopIdx);
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
                auto gmTileA = gmA[inGroupOffsetA + layoutWA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{0, NGmBlockIdx * maxNPerBlock}; // gm中的偏移量
                auto gmTileB = gmB[inGroupOffsetB + layoutWB.GetOffset(gmTileBOffset)];
                MatrixCoord gmTileXOffset{MGmBlockIdx * maxMPerBlock, NGmBlockIdx * maxNPerBlock}; // gm中的偏移量
                auto gmTileX = gmX[inGroupOffsetWorkspace + layoutWorkspace.GetOffset(gmTileXOffset)];
                MatrixCoord gmTileNextAOffset{MNextGmBlockIdx * maxMPerBlock, 0}; // gm中的偏移量
                auto gmTileNextA = gmA[inGroupOffsetA + layoutWA.GetOffset(gmTileNextAOffset)];
                MatrixCoord gmTileNextBOffset{0, NNextGmBlockIdx * maxNPerBlock}; // gm中的偏移量
                auto gmTileNextB = gmB[inGroupOffsetB + layoutWB.GetOffset(gmTileNextBOffset)];
                blockGemm(
                    gmTileA, layoutWA, // row col stride不一样了
                    gmTileB, layoutWB,
                    gmTileX, layoutWorkspace,
                    gmTileNextA, gmTileNextB,
                    actualShape, nextActualShape, isFirstBlock, hasNextBlock, singleIdx
                );
                arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
                singleIdx = (singleIdx + 1) % l0CBlockNum;
            }
            inGroupOffsetA += GetWorkspaceLen(layoutWA);
            inGroupOffsetB += GetWorkspaceLen(layoutWB);
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            #pragma unroll
            for(uint32_t i = 0; i < l0CBlockNum; i++){
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
        }
    } 

    template<>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        MatmulCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutX layoutWorkspaceList[MAX_TENSOR_COUNT];
        ElementScalar alphaList[MAX_TENSOR_COUNT];
        ElementScalar betaList[MAX_TENSOR_COUNT];
        LayoutA layoutWAList[MAX_TENSOR_COUNT];
        LayoutB layoutWBList[MAX_TENSOR_COUNT];
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutWorkspaceList, params.ptrLayoutWorkspace, params.problemCount);
        detail::UnpackListParam(alphaList, params.alpha, params.problemCount);
        detail::UnpackListParam(betaList, params.beta, params.problemCount);
        detail::UnpackListParam(layoutWAList, params.ptrlayoutWA, params.problemCount);
        detail::UnpackListParam(layoutWBList, params.ptrlayoutWB, params.problemCount);
        uint32_t coreIdx = AscendC::GetBlockIdx() / 2;
        uint32_t coreNum = AscendC::GetBlockNum();
        uint64_t inGroupOffsetA = 0;
        uint64_t inGroupOffsetWA = 0;
        uint64_t inGroupOffsetB = 0;
        uint64_t inGroupOffsetWB = 0;
        uint64_t inGroupOffsetWorkspace = 0;
        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;
        MatmulCoord blockShape = L1TileShape::ToCoord();
        arch::Resource<ArchTag> resource;
        BlockEpilogue blockEpilogue(resource, blockShape);
        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrWorkspace);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer((__gm__ ElementD*)params.ptrD);
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementA> gmWA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
        gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrWA));
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementB> gmWB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
        gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));
        for(uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx){
            MatmulCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutX layoutWorkspace = layoutWorkspaceList[groupIdx];
            ElementScalar alpha_ = alphaList[groupIdx];
            ElementScalar beta_ = betaList[groupIdx];
            LayoutA layoutWA = layoutWAList[groupIdx];
            LayoutB layoutWB = layoutWBList[groupIdx];
            if (!IsSameStride(layoutWA, layoutA)) {
                PaddingA paddingA(resource);
                paddingA(gmWA[inGroupOffsetWA], gmA[inGroupOffsetA], layoutWA, layoutA); // 两个AIV核
            }

            if (!IsSameStride(layoutWB, layoutB)) {
                PaddingB paddingB(resource);
                paddingB(gmWB[inGroupOffsetWB], gmB[inGroupOffsetB], layoutWB, layoutB);
            }
            if (!IsSameStride(layoutWA, layoutA) || !IsSameStride(layoutWB, layoutB)) {
                AscendCT::arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
                AscendCT::arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
            }
            
            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            uint32_t MLoops = CeilDiv(M, maxMPerBlock);
            uint32_t NLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = MLoops * NLoops;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            for(uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()){
                uint32_t MGmBlockIdx = loopIdx / NLoops;
                uint32_t NGmBlockIdx = loopIdx % NLoops;
                uint32_t MGmActual = (MGmBlockIdx == MLoops - 1) ? (M - MGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t NGmActual = (NGmBlockIdx == NLoops - 1) ? (N - NGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                MatmulCoord actualShape{MGmActual, NGmActual, K};
                arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore); // 没这个变量
                MatrixCoord gmTileOffset{MGmBlockIdx * maxMPerBlock, NGmBlockIdx * maxNPerBlock}; // gm中的偏移量
                auto offsetX = layoutWorkspace.GetOffset(gmTileOffset);
                blockEpilogue( // 传入偏移量
                    alpha_, beta_,
                    gmC[inGroupOffsetWorkspace + offsetX], layoutWorkspace,
                    gmD[inGroupOffsetWorkspace + offsetX], layoutWorkspace,
                    gmX[inGroupOffsetWorkspace + offsetX], layoutWorkspace,
                    actualShape
                );
            }
            inGroupOffsetA += problemShape.m() * problemShape.k();
            inGroupOffsetWA += GetWorkspaceLen(layoutWA);
            inGroupOffsetB += problemShape.k() * problemShape.n();
            inGroupOffsetWB += GetWorkspaceLen(layoutWB);
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
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

#endif // ASCENDCT_GROUPGEMM_KERNEL_GROUPGEMM_PL_PA_EPILOGUE_HPP}