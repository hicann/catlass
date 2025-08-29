/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "catlass/catlass.hpp"
 #include "catlass/arch/arch.hpp"
 #include "catlass/layout/layout.hpp"
 
 #include "catlass/gemm/block/block_mmad.hpp"
 #include "catlass/gemm/dispatch_policy.hpp"
 #include "catlass/gemm/gemm_type.hpp"
 
 #include "catlass/arch/cross_core_sync.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/epilogue/block/block_epilogue.hpp"
 #include "catlass/epilogue/dispatch_policy.hpp"
 #include "catlass/debug.hpp"
 
 #include "kernel_common.hpp"
 #include "kernel_operator.h"
 using namespace Catlass;
 
 
 template <
     class BlockMmadQK,
     class BlockMmadPV,
     class EpilogueOnlineSoftmax,
     class EpilogueRescaleO,
     bool PAGED_CACHE_FLAG>
 class FAInferKernel {
 public:
     using ArchTag = typename BlockMmadQK::ArchTag;
     using L1TileShape = typename BlockMmadQK::L1TileShape;
     using ElementQ = typename BlockMmadQK::ElementA;
     using LayoutQ = typename BlockMmadQK::LayoutA;
     using ElementK = typename BlockMmadQK::ElementB;
     using LayoutK = typename BlockMmadQK::LayoutB;
     using ElementS = typename BlockMmadQK::ElementC;
     using LayoutS = typename BlockMmadQK::LayoutC;
 
     using ElementP = typename BlockMmadPV::ElementA;
     using LayoutP = typename BlockMmadPV::LayoutA;
     using ElementV = typename BlockMmadPV::ElementB;
     using LayoutV = typename BlockMmadPV::LayoutB;
 
     using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
     using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;
 
     using ElementO = typename EpilogueRescaleO::ElementOutput;
     using LayoutO = typename EpilogueRescaleO::LayoutOutput;
 
     using ElementOTmp = typename EpilogueRescaleO::ElementInput;
     using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;
 
     /// Parameters structure
     struct Params {
         // Data members
         GM_ADDR q;
         GM_ADDR k;
         GM_ADDR v;
         GM_ADDR mask;
         GM_ADDR blockTables;
         GM_ADDR actualQseqlen;
         GM_ADDR actualKvseqlen;
         GM_ADDR o;
         GM_ADDR s;
         GM_ADDR p;
         GM_ADDR oTemp;
         GM_ADDR oUpdate;
         GM_ADDR tiling;
 
         // Methods
         CATLASS_DEVICE
         Params() {}
 
         CATLASS_DEVICE
         Params(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR s_, GM_ADDR p_, GM_ADDR oTemp_, GM_ADDR oUpdate_, GM_ADDR tiling_)
             : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
               actualKvseqlen(actualKvseqlen_), o(o_), s(s_), p(p_), oTemp(oTemp_), oUpdate(oUpdate_), tiling(tiling_) {}
     };
 
     struct FATilingData {
         uint32_t numHeads = 0;
         uint32_t embeddingSize = 0;
         uint32_t numBlocks = 0;
         uint32_t blockSize = 0;
         uint32_t maxKvSeqlen = 0;
         uint32_t kvHeads = 0;
         uint32_t batch = 0;
         uint32_t maxNumBlocksPerBatch = 0;
         uint32_t firstBatchTaskNum = 0;
         uint32_t totalTaskNum = 0;
         uint32_t maskType = 0;
         uint64_t mm1OutSize = 0;
         uint64_t smOnlineOutSize = 0;
         uint64_t mm2OutSize = 0;
         uint64_t UpdateSize = 0;
         uint64_t workSpaceSize = 0;
         float scaleValue = 0.0;
     };
 
     // Methods
     CATLASS_DEVICE
     FAInferKernel() {}
 
     CATLASS_DEVICE
     uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
     {
         uint32_t qNBlockTile = (128 / qSeqlen) / 2 * 2;
         qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
         qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
         return qNBlockTile;
     }
 
     CATLASS_DEVICE
     uint32_t GetQSBlockTile(uint32_t kvSeqlen)
     {
         uint32_t qSBlockTile = 128;
         return qSBlockTile;
     }
 
     template <int32_t CORE_TYPE = g_coreType>
     CATLASS_DEVICE void operator()(Params const &params);
 
         template <>
     CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
     {
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
         AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
         AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
         AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
         BlockMmadQK blockMmadQK(resource);
         static constexpr uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) + 
                                              BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
         BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
 
         __gm__ FATilingData *fATilingData = reinterpret_cast<__gm__ FATilingData *>(params.tiling);
         uint64_t mm1OutSize = fATilingData->mm1OutSize;
         uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
         uint32_t batch = fATilingData->batch;
         uint32_t qHeads = fATilingData->numHeads;
         uint32_t kvHeads = fATilingData->kvHeads;
         uint32_t embed = fATilingData->embeddingSize;
         uint32_t pagedBlockSize = fATilingData->blockSize;
         uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
         uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
         uint32_t totalTaskNum = fATilingData->totalTaskNum;
         uint32_t blockSize = fATilingData->blockSize;
         uint32_t maskType = fATilingData->maskType;
         float scaleValue = fATilingData->scaleValue;
 
         AscendC::GlobalTensor<ElementQ> gQ;
         gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
         // AscendC::DumpTensor(gQ, 11, 128);
         AscendC::GlobalTensor<ElementK> gK;
         gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
         AscendC::GlobalTensor<ElementK> gV;
         gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
         AscendC::GlobalTensor<int32_t> gBlockTable;
         gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
         AscendC::GlobalTensor<int64_t> gActualQseqlen;
         gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
         AscendC::GlobalTensor<int64_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
 
         uint64_t strideQO = qHeads * embed;
         uint64_t strideKV = kvHeads * embed;
         uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
         uint32_t groupSize = qHeads / kvHeads;
 
         uint32_t coreIdx = AscendC::GetBlockIdx();
         uint32_t coreNum = AscendC::GetBlockNum();
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint64_t qBOffset = 0;
         uint64_t kBOffset = 0;
         uint64_t vBOffset = 0;
         uint64_t blockBOffset = 0;
 
         uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
         uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
         uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
         uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         uint32_t curTotalTaskNum = firstBatchTaskNum;
 
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
             while (taskIdx >= curTotalTaskNum) {
                 ++curBatch;
                 preTotalTaskNum = curTotalTaskNum;
                 qBOffset += qSeqlen * strideQO;
                 if constexpr (!PAGED_CACHE_FLAG) {
                     kBOffset += kvSeqlen * strideKV;
                     vBOffset += kvSeqlen * strideKV;
                 } else {
                     blockBOffset += maxNumBlocksPerBatch;
                 }
                 qSeqlen = reinterpret_cast<int64_t>(gActualQseqlen.GetValue(curBatch));
                 kvSeqlen = reinterpret_cast<int64_t>(gActualKvseqlen.GetValue(curBatch));
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
             uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint64_t gmQOffset = qBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;
             uint64_t gmKOffset = kBOffset + kvHeadIdx * embed;
             uint64_t gmVOffset = vBOffset + kvHeadIdx * embed;
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1)) ?
                 (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = RoundUp(rowNum, BLOCK_SIZE);
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
             if (maskType != 0) {
                 uint32_t diffS = kvSeqlen - qSeqlen;
                 noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                 noSkipKvS = Min((uint32_t)kvSeqlen, noSkipKvS);
                 noMaskKvS = noSkipKvS - qSBlockSize;
                 noMaskTailS = noMaskKvS % pagedBlockSize;
             }
             uint32_t maskedKvS = qSBlockSize;
             uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             uint32_t blockStackNum = 4;
             uint32_t stackSeqTile;
             uint32_t stackSeqTileRound = blockStackNum * 128;
             int32_t preLaunch = 2;
             int32_t totalStackSeqNum = (maskType != 0) ?
                 (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1) :
                 CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
             int32_t stackSeqCount = 0;
             LayoutQ layoutQTemp(rowNum, embed);
             LayoutK layoutKTemp(strideKV, blockStackNum * pagedBlockSize);
             LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideKV);
             blockMmadQK.loadQGM(gQ[gmQOffset], layoutQTemp, rowNum, qNBlockSize, qHeads);
             for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
                 if (kvSIdx < kvSLoopNumNoMask) {
                     if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                     uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     blockMmadQK(gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset],
                         gBlockTable[blockBOffset], layoutQTemp, layoutKTemp, actualBlockShapeQK,
                         kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV);
                     if constexpr (!PAGED_CACHE_FLAG) {
                         gmKOffset += stackSeqTile * strideKV;
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                 }
                 if (kvSIdx >= preLaunch * blockStackNum) {
                     uint32_t nowkvSIdx = kvSIdx - preLaunch * blockStackNum;
                     if (nowkvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - nowkvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     Arch::CrossCoreWaitFlag(softmaxReady);
                     uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     uint64_t gmOTmpOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     blockMmadPV(gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset],
                         gBlockTable[blockBOffset], layoutPTemp, layoutVTemp, actualBlockShapePV,
                         nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV, softmaxReady);
                     if constexpr (!PAGED_CACHE_FLAG) {
                         gmVOffset += stackSeqTile * strideKV;
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                 }
                 stackSeqCount++;
             }
             /*
             * for the secondary loop
             * while masked, it deals the CV stage1(Qk^t/SMOnline) of the final base block(typical shape [128, 512]), and the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks
             * while unmasked, it deals the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks
             *  
             */
             // deal secondary loop
             uint32_t maskedStartIdx = (maskType != 0) ?
                 ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask) :
                 AlignUp(kvSLoopNumNoMask, blockStackNum);
             uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
             noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? 
                     noMaskTailInteStackNum : ((noMaskTailS != 0) ? 0:blockStackNum);
             uint32_t preLaunchStackNum = (maskType != 0) ?
                 ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum) : (preLaunch * blockStackNum);
 
             // masked kvSeqlen loop
             for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum; ){
                 if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)){
                     stackSeqTile = maskedKvS;
                     uint32_t SWorkSpacePingPongFlag = stackSeqCount % (preLaunch + 1);
                     uint64_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     if constexpr (PAGED_CACHE_FLAG) {
                         blockMmadQK(gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset],
                             gBlockTable[blockBOffset], layoutQTemp, layoutKTemp, actualBlockShapeQK,
                             kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS, strideKV, noMaskTailS, 1);
                     } else {
                         gmKOffset += stackSeqTile * strideKV;
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                     
                 }
                 if (kvSIdx >= preLaunchStackNum){
                     uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)){
                         stackSeqTile = maskedKvS;
                     }
                     else if(delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1){
                         stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                     }
                     else{
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     uint32_t PVWorkSpacePingPongFlag = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint64_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     uint64_t gmOTmpOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                     LayoutP layoutPTemp(rowNum, stackSeqTileRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     Arch::CrossCoreWaitFlag(softmaxReady);
 
                     if ((stackSeqCount - preLaunch == totalStackSeqNum - 1) && (maskType != 0)) {
                         blockMmadPV(gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset],
                             gBlockTable[blockBOffset], layoutPTemp, layoutVTemp, actualBlockShapePV,
                             delayedKvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS, strideKV, softmaxReady, noMaskTailS, 1);
                         if constexpr (!PAGED_CACHE_FLAG) {
                             gmVOffset += stackSeqTile * strideKV;
                         }
                     } else {
                         blockMmadPV(gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset],
                             gBlockTable[blockBOffset], layoutPTemp, layoutVTemp, actualBlockShapePV,
                             delayedKvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV, softmaxReady);
                         if constexpr (!PAGED_CACHE_FLAG) {
                             gmVOffset += stackSeqTile * strideKV;
                         }
                     }
                     Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                 }
                 if((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)){
                     kvSIdx += noMaskTailInteStackNum;
                 }
                 else{
                     kvSIdx += blockStackNum;
                 }
                 stackSeqCount++;
             }
         }
 
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
         AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
 
         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
         AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
     }
 
     template <>
     CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
     {
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
 
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
         AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
 
         // Get tiling parameters
         __gm__ FATilingData *fATilingData = reinterpret_cast<__gm__ FATilingData *>(params.tiling);
         uint64_t mm1OutSize = fATilingData->mm1OutSize;
         uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
         uint64_t mm2OutSize = fATilingData->mm2OutSize;
         uint32_t batch = fATilingData->batch;
         uint32_t qHeads = fATilingData->numHeads;
         uint32_t kvHeads = fATilingData->kvHeads;
         uint32_t embed = fATilingData->embeddingSize;
         uint32_t pagedBlockSize = fATilingData->blockSize;
         uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
         uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
         uint32_t totalTaskNum = fATilingData->totalTaskNum;
         uint32_t maskType = fATilingData->maskType;
         float scaleValue = fATilingData->scaleValue;
         // Get the memory offset address of the input on Global Memory
         AscendC::GlobalTensor<ElementMask> gMask;
         gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
         AscendC::GlobalTensor<int64_t> gActualQseqlen;
         gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
         AscendC::GlobalTensor<int64_t> gActualKvseqlen;
         gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
         AscendC::GlobalTensor<ElementO> gO;
         gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
         AscendC::GlobalTensor<ElementS> gS;
         gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
         AscendC::GlobalTensor<ElementP> gP;
         gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
         AscendC::GlobalTensor<ElementOTmp> gOTmp;
         gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
         AscendC::GlobalTensor<ElementOTmp> gOUpdate;
         gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
 
         uint32_t groupSize = qHeads / kvHeads;
         uint32_t embedRound = RoundUp(embed, BLOCK_SIZE);
 
         EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
         EpilogueRescaleO epilogueRescaleO(resource);
 
         
         // uint32_t curTotalTaskNum = 0;
         uint32_t preTotalTaskNum = 0;
         uint32_t curBatch = 0;
         uint32_t oBatchOffset = 0;
         uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
         uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
         uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
         uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
         uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
         uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
         uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
         uint32_t curTotalTaskNum = firstBatchTaskNum;
 
         uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
         uint32_t coreNum = AscendC::GetBlockNum();
         // Go through each task.
         for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
             // Get the offset of each core on the GM.
             while (taskIdx >= curTotalTaskNum) {
                 curBatch++;
                 oBatchOffset += qSeqlen * qHeads * embed;
                 preTotalTaskNum = curTotalTaskNum;
                 qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                 kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                 curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                 qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                 curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                 curQSBlockTile = GetQSBlockTile(kvSeqlen);
                 curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                 curTotalTaskNum += curQNBlockNum * curQSBlockNum;
             }
             uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
             uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
             uint32_t qNBlockIdx = taskIdxCurBatch % curQNBlockNum;
             uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
             
             uint32_t oSOffset = qSBlockIdx * curQSBlockTile * qHeads * embed;
             uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
             uint32_t qStartNIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
             uint32_t oNOffset = qStartNIdx * embed;
             uint32_t gmOffsetO = oBatchOffset + oSOffset + oNOffset;
 
             uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ?
                 (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
             uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1)) ?
                 (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
             uint32_t rowNum = qSBlockSize * qNBlockSize;
             uint32_t rowNumRound = RoundUp(rowNum, BLOCK_SIZE);
 
             uint32_t noSkipKvS = kvSeqlen;
             uint32_t noMaskKvS = kvSeqlen;
             uint32_t noMaskTailS = 0;
             if (maskType != 0) {
                 uint32_t diffS = kvSeqlen - qSeqlen;
                 noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                 noSkipKvS = Min(kvSeqlen, noSkipKvS);
                 noMaskKvS = noSkipKvS - qSBlockSize;
                 noMaskTailS = noMaskKvS % pagedBlockSize;
             }
             uint32_t maskedKvS = qSBlockSize;
             uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
             uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
             uint32_t blockStackNum = 4;
             uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
             uint32_t stackSeqTile;
             int32_t preLaunch = 2;
             int32_t totalStackSeqNum = (maskType != 0) ?
                 (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1) :
                 CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
             int32_t stackSeqCount = 0;
 
             // no mask kvSeqlen loop
             for (uint32_t kvSIdx = 0;
                 kvSIdx < kvSLoopNumNoMask;
                 kvSIdx += blockStackNum) {
 
                 if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                     stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                 } else {
                     stackSeqTile = pagedBlockSize * blockStackNum;
                 }
                 uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                 LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                 LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                 GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                 uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                 uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                     curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                 // vec core offset will be processed within epilogue block
                 uint32_t gmOffsetP = gmOffsetS;
                 // AscendC::printf("stackSeqCount:%d\n", stackSeqCount);
                 Arch::CrossCoreWaitFlag(qkReady);
                 // online softmax
                 epilogueOnlineSoftmax(
                     gP[gmOffsetP],
                     gS[gmOffsetS],
                     layOutP,
                     layOutS,
                     actualBlockShapeQK,
                     (stackSeqCount == 0),
                     qSBlockSize,
                     qNBlockSize,
                     curStackTileMod);
                 Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
 
                 if (kvSIdx >= preLaunch * blockStackNum) {
                     uint32_t delayedKvSIdx = kvSIdx - preLaunch * blockStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                         stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                     } else {
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     LayoutO layoutO(qSeqlen, embed * qHeads);
                     LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) +
                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                     Arch::CrossCoreWaitFlag(pvReady);
                     // rescale O
                     epilogueRescaleO(
                         gO[gmOffsetO],
                         gOTmp[gmOffsetOTmp],
                         layoutO,
                         layoutOTmp,
                         actualBlockShapePV,
                         qSBlockSize,
                         qNBlockSize,
                         (stackSeqCount - preLaunch == 0),
                         0,
                         curStackTileMod);
                 }
                 stackSeqCount++;
             }
             /*
             * for the secondary loop
             * while masked, it deals the CV stage1(Qk^t/SMOnline) of the final base block(typical shape [128, 512]), and the CV stage2(PV/rescaleO) of the last (prelaunch+1) base blocks
             * while unmasked, it deals the CV stage1(Qk^t/SMOnline) of the last (prelaunch+1) base blocks
             */
            // deal secondary loop conditions
            uint32_t maskedStartIdx = (maskType != 0) ?
             ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask) :
             AlignUp(kvSLoopNumNoMask, blockStackNum);
             uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
             noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ?
                noMaskTailInteStackNum : ((noMaskTailS != 0) ? 0 : blockStackNum);
             uint32_t preLaunchStackNum = (maskType != 0) ?
                 ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum) : (preLaunch * blockStackNum);
             // masked kvSeqlen loop
             for (uint32_t kvSIdx = maskedStartIdx;
                 kvSIdx < kvSLoopNumTotal + preLaunchStackNum; ){
                 if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)){
                     stackSeqTile = maskedKvS;
                     uint32_t stackSeqTileRound = RoundUp(stackSeqTile, BLOCK_SIZE);
                     LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
                     LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                     LayoutMask layOutMask(1024, 1024, 1024);
                     GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                     uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                     uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + // cube core offset
                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                     // vec core offset will be processed within epilogue block
                     uint32_t gmOffsetP = gmOffsetS;
                     // online softmax
                     epilogueOnlineSoftmax(
                         gP[gmOffsetP],
                         gS[gmOffsetS],
                         gMask,
                         layOutP,
                         layOutS,
                         layOutMask,
                         actualBlockShapeQK,
                         (stackSeqCount == 0),
                         qSBlockSize,
                         qNBlockSize,
                         curStackTileMod,
                         qkReady);
                     Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                 }
                 if (kvSIdx >= preLaunchStackNum){
                     uint32_t delayedKvSIdx = kvSIdx - preLaunchStackNum;
                     if (delayedKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && (maskType != 0)){
                         stackSeqTile = maskedKvS;
                     }
                     else if(delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1){
                         stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                     }
                     else{
                         stackSeqTile = pagedBlockSize * blockStackNum;
                     }
                     LayoutO layoutO(qSBlockSize, embed * qHeads);
                     LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                     GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                     uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                     uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) +
                         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                     Arch::CrossCoreWaitFlag(pvReady);
                     // rescale O
                     epilogueRescaleO(
                         gO[gmOffsetO],
                         gOTmp[gmOffsetOTmp],
                         layoutO,
                         layoutOTmp,
                         actualBlockShapePV,
                         qSBlockSize,
                         qNBlockSize,
                         (stackSeqCount - preLaunch == 0),
                         (stackSeqCount - preLaunch == totalStackSeqNum - 1),
                         curStackTileMod);
                 }
                 if((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)){
                     kvSIdx += noMaskTailInteStackNum;
                 }
                 else{
                     kvSIdx += blockStackNum;
                 }
                 stackSeqCount++;
             }
         }
 
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
 
         
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
 
 
     }
 
 private:
     Arch::Resource<ArchTag> resource;
     Arch::CrossCoreFlag qkReady{QK_READY_ID};
     Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
     Arch::CrossCoreFlag pvReady{PV_READY_ID};
 };
 
 CATLASS_GLOBAL void FAInferFp16(uint64_t fftsAddr,
                                 GM_ADDR q,
                                 GM_ADDR k,
                                 GM_ADDR v,
                                 GM_ADDR mask,
                                 GM_ADDR blockTables,
                                 GM_ADDR o,
                                 GM_ADDR actualQseqlen,
                                 GM_ADDR actualKvseqlen,
                                 GM_ADDR s,
                                 GM_ADDR p,
                                 GM_ADDR oTemp,
                                 GM_ADDR oUpdate,
                                 GM_ADDR tiling)
 {
     // Set FFTS address
     AscendC::SetSyncBaseAddr(fftsAddr);
 
     using ArchTag = Arch::AtlasA2;
     using ElementQ = half;
     using LayoutQ = layout::RowMajor;
     using ElementK = half;
     using LayoutK = layout::ColumnMajor;
     using ElementV = half;
     using LayoutV = layout::RowMajor;
     using ElementS = float;
     using LayoutS = layout::RowMajor;
     using ElementP = half;
     using LayoutP = layout::RowMajor;
     using ElementO = half;
     using LayoutO = layout::RowMajor;
     using ElementMask = half;
     using LayoutMask = layout::RowMajor;
     using ElementOTmp = float;
     using LayoutOTmp = layout::RowMajor;
     using ElementUpdate = float;
     using LayoutUpdate = layout::RowMajor;
     // L1TileShape::K must be embdding
     using L1TileShape = GemmShape<128, 128, 128>;
     using L0TileShape = L1TileShape;
     // GEMM Block模块，实现Flash Attention Infer的Q * K^T
     using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true>;
     using QType = Gemm::GemmType<ElementQ, LayoutQ>;
     using KType = Gemm::GemmType<ElementK, LayoutK>;
     using SType = Gemm::GemmType<ElementS, LayoutS>;
     using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;
     // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
     using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax;
     using PType = Gemm::GemmType<ElementP, LayoutP>;
     using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
     using EpilogueOnlineSoftmax =
         Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;
 
     // GEMM Block模块，实现Flash Attention Infer的P * V
     using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true>;
     using VType = Gemm::GemmType<ElementV, LayoutV>;
     using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
     using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;
 
     // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
     using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO;
     using OType = Gemm::GemmType<ElementO, LayoutO>;
     using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
     using EpilogueRescaleO =
         Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType>;
 
 
     // Kernel level
     using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, true>;
     typename FAInferKernel::Params params{q, k, v, mask, blockTables,
                                           actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling};
 
     // call kernel
     FAInferKernel flashAttnInfer;
     flashAttnInfer(params);
 }
 
 CATLASS_GLOBAL void FAInferBf16(uint64_t fftsAddr,
                                 GM_ADDR q,
                                 GM_ADDR k,
                                 GM_ADDR v,
                                 GM_ADDR mask,
                                 GM_ADDR blockTables,
                                 GM_ADDR o,
                                 GM_ADDR actualQseqlen,
                                 GM_ADDR actualKvseqlen,
                                 GM_ADDR s,
                                 GM_ADDR p,
                                 GM_ADDR oTemp,
                                 GM_ADDR oUpdate,
                                 GM_ADDR tiling)
 {
     // Set FFTS address
     AscendC::SetSyncBaseAddr(fftsAddr);
 
     using ArchTag = Arch::AtlasA2;
     using ElementQ = bfloat16_t;
     using LayoutQ = layout::RowMajor;
     using ElementK = bfloat16_t;
     using LayoutK = layout::ColumnMajor;
     using ElementV = bfloat16_t;
     using LayoutV = layout::RowMajor;
     using ElementS = float;
     using LayoutS = layout::RowMajor;
     using ElementP = bfloat16_t;
     using LayoutP = layout::RowMajor;
     using ElementO = bfloat16_t;
     using LayoutO = layout::RowMajor;
     using ElementMask = bfloat16_t;
     using LayoutMask = layout::RowMajor;
     using ElementOTmp = float;
     using LayoutOTmp = layout::RowMajor;
     using ElementUpdate = float;
     using LayoutUpdate = layout::RowMajor;
     // L1TileShape::K must be embdding
     using L1TileShape = GemmShape<128, 128, 128>;
     using L0TileShape = L1TileShape;
     // GEMM Block模块，实现Flash Attention Infer的Q * K^T
     using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true>;
     using QType = Gemm::GemmType<ElementQ, LayoutQ>;
     using KType = Gemm::GemmType<ElementK, LayoutK>;
     using SType = Gemm::GemmType<ElementS, LayoutS>;
     using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;
     // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
     using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax;
     using PType = Gemm::GemmType<ElementP, LayoutP>;
     using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
     using EpilogueOnlineSoftmax =
         Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;
     // GEMM Block模块，实现Flash Attention Infer的P * V
     using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true>;
     using VType = Gemm::GemmType<ElementV, LayoutV>;
     using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
     using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;
     // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
     using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO;
     using OType = Gemm::GemmType<ElementO, LayoutO>;
     using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
     using EpilogueRescaleO =
         Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType>;
 
     // Kernel Level
     using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, true>;
     typename FAInferKernel::Params params{q, k, v, mask, blockTables,
         actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling};
     // call kernel
     FAInferKernel flashAttnInfer;
     flashAttnInfer(params);
 }