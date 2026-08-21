/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "rfa_tilingdata.h"
#include "rfa_kernel_utils.h"

using namespace Catlass;
using namespace tla;

/**
 * @brief Rain Fusion Attention Inference Kernel
 *
 * This kernel implements rain fusion attention where attention is computed only on
 * selected KV blocks specified by selectIdx. This reduces computation for long sequences
 * by focusing on relevant tokens.
 *
 * @tparam BlockMmadQK, Block-level QK matmul module
 * @tparam EpilogueOnlineSoftmax, Online softmax epilogue
 * @tparam BlockMmadPV, Block-level PV matmul module
 * @tparam EpilogueRescaleO, Output rescaling epilogue
 * @tparam PAGED_CACHE_FLAG, Whether to use paged KV cache
 * @tparam QUERY_LAYOUT, Query tensor layout (0=TND, 1=BNSD)
 * @tparam KV_CACHE_LAYOUT, KV cache layout (0=TND, 1=BNSD)
 */
template <
    class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, bool PAGED_CACHE_FLAG,
    uint32_t QUERY_LAYOUT, uint32_t KV_CACHE_LAYOUT>
class RfaKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;
    using ElementLse = typename EpilogueRescaleO::ElementLse;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    static constexpr uint32_t Q_L1_BUF_NUM = BlockMmadQK::L1A_BUF_NUM;
    static constexpr uint32_t K_L1_BUF_NUM = BlockMmadQK::L1B_BUF_NUM;
    static constexpr uint32_t P_L1_BUF_NUM = BlockMmadPV::L1A_BUF_NUM;
    static constexpr uint32_t V_L1_BUF_NUM = BlockMmadPV::L1B_BUF_NUM;

    static constexpr uint32_t L1_P_BUF_SIZE = BlockMmadPV::L1A_BUF_SIZE;
    static constexpr uint32_t L1_QK_SIZE = BlockMmadQK::BLOCK_L1_SIZE;
    static constexpr uint32_t L1_PV_SIZE = BlockMmadPV::BLOCK_L1_SIZE;
    static_assert(L1_QK_SIZE + L1_PV_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
    static_assert(P_L1_BUF_NUM == PRE_LAUNCH + 1, "P L1 buffers num must be equal to PRE_LAUNCH + 1!");

    __aicore__ inline RfaKernelArch35()
    {}

    __aicore__ inline void operator()(RfaKernelParamsArch35 const& params)
    {
        __gm__ RfaTilingData* tilingData = reinterpret_cast<__gm__ RfaTilingData*>(params.tiling);
        GetTilingData(tilingData);
        CalcBlockMmadL0Stages();

        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV*)params.v);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualKvseqlen);
        AscendC::GlobalTensor<int64_t> gSelectIdx;
        gSelectIdx.SetGlobalBuffer((__gm__ int64_t*)params.selectIdx);
        AscendC::GlobalTensor<int64_t> gSelectNumIdx;
        gSelectNumIdx.SetGlobalBuffer((__gm__ int64_t*)params.selectNumIdx);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);

        // local buffers
        AscendC::LocalTensor<ElementP> l1PTensor[P_L1_BUF_NUM];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        // set reverse sync flags
        InitSyncFlags<4, 4, 4>();

#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif
        // Calculate strides based on layout
        // For TND: [T, N, D], stride = N * D
        // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
        int64_t strideQO = 0;
        int64_t strideKV = 0;
        int64_t strideQOB = 0; // BNSD batch stride for Q
        int64_t strideQON = 0; // BNSD head stride for Q
        int64_t strideQOS = 0; // BNSD seq stride for Q
        int64_t strideKVB = 0; // BNSD batch stride for KV
        int64_t strideKVN = 0; // BNSD head stride for KV
        int64_t strideKVS = 0; // BNSD seq stride for KV

        if constexpr (QUERY_LAYOUT == 1) {
            // BNSD: [B, N, S, D]
            strideQOB = static_cast<int64_t>(qHeads_) * maxQSeqlen_ * embed_; // batch stride
            strideQON = static_cast<int64_t>(maxQSeqlen_) * embed_;           // head stride
            strideQOS = embed_;                                               // seq stride
        } else {
            // TND: [T, N, D]
            strideQO = static_cast<int64_t>(qHeads_) * embed_;
        }
        if constexpr (KV_CACHE_LAYOUT == 1) {
            // BNSD: [B, N, S, D]
            strideKVB = static_cast<int64_t>(kvHeads_) * maxKvSeqlen_ * embed_; // batch stride
            strideKVN = static_cast<int64_t>(maxKvSeqlen_) * embed_;            // head stride
            strideKVS = embed_;                                                 // seq stride
        } else {
            // TND: [T, N, D]
            strideKV = static_cast<int64_t>(kvHeads_) * embed_;
        }

        uint32_t qSTileNumInXBlock = (blockShapeX_ + qBaseTile_ - 1) / qBaseTile_;
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t bOffsetQO = 0;
        int64_t bOffsetKV = 0;
        uint32_t curBatch = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t preTotalQBlockNum = 0;
        int64_t qSeqlen = isVarLen_ ? gActualQseqlen.GetValue(curBatch) : maxQSeqlen_;
        int64_t kvSeqlen = isVarLen_ ? gActualKvseqlen.GetValue(curBatch) : maxKvSeqlen_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        uint32_t curQXBlockNum = (qSeqlen + blockShapeX_ - 1) / blockShapeX_;
        uint32_t curTotalQBlockNum = firstQBlockNum_;

        // Go through each task
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            // Get the offset of each core on the GM
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                preTotalQBlockNum = curTotalQBlockNum;

                // Update offsets based on layout
                if constexpr (QUERY_LAYOUT == 1) {
                    // BNSD: [B, N, S, D], batch offset = batch * strideB
                    bOffsetQO = static_cast<int64_t>(curBatch) * strideQOB;
                } else {
                    // TND
                    bOffsetQO += static_cast<int64_t>(qSeqlen) * strideQO;
                }
                if constexpr (KV_CACHE_LAYOUT == 1) {
                    // BNSD: [B, N, S, D], batch offset = batch * strideB
                    bOffsetKV = static_cast<int64_t>(curBatch) * strideKVB;
                } else {
                    // TND
                    bOffsetKV += static_cast<int64_t>(kvSeqlen) * strideKV;
                }

                qSeqlen = isVarLen_ ? gActualQseqlen.GetValue(curBatch) : maxQSeqlen_;
                kvSeqlen = isVarLen_ ? gActualKvseqlen.GetValue(curBatch) : maxKvSeqlen_;
                curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
                curTotalTaskNum += qHeads_ * curQSTileNum; // batch0～当前batch 的task总数
                curQXBlockNum = (qSeqlen + blockShapeX_ - 1) / blockShapeX_;
                curTotalQBlockNum += qHeads_ * curQXBlockNum; // batch0～当前batch 的Block总数
            }

            // Q task splitting按照[qNBlockNum, qHead]
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSTileIdx = taskIdxCurBatch / qHeads_;
            uint32_t qXBlockIdx = qSTileIdx / qSTileNumInXBlock;
            uint32_t qXBlockInnerIdx = qSTileIdx - qXBlockIdx * qSTileNumInXBlock;
            uint32_t qHeadIdx = taskIdxCurBatch - qSTileIdx * qHeads_;
            uint32_t xTailBlockLen = qSeqlen - (curQXBlockNum - 1) * blockShapeX_;
            uint32_t kvHeadIdx = qHeadIdx / groupSize; // 当前 task 对应在kvHeads的索引

            uint32_t curSelectNumIdx = preTotalQBlockNum + qXBlockIdx * qHeads_ + qHeadIdx;
            uint32_t curSelectNum = static_cast<uint32_t>(gSelectNumIdx.GetValue(curSelectNumIdx));
            // skip this task
            if (curSelectNum == 0) {
                continue;
            }

            uint32_t lastSelectIdx =
                static_cast<int32_t>(gSelectIdx.GetValue(curSelectNumIdx * maxKvBlockNum_ + curSelectNum - 1));
            uint32_t curKvYBlockNum = (kvSeqlen + blockShapeY_ - 1) / blockShapeY_;
            uint32_t selectKvSeqLen = (lastSelectIdx == curKvYBlockNum - 1 && kvSeqlen % blockShapeY_ != 0) ?
                                          blockShapeY_ * (curSelectNum - 1) + kvSeqlen % blockShapeY_ :
                                          blockShapeY_ * curSelectNum;

            // Calculate offsets based on layout
            int64_t gmOffsetQO = 0;
            int64_t gmOffsetKV = 0;
            int64_t qSeqOffset = qXBlockIdx * blockShapeX_ + qXBlockInnerIdx * qBaseTile_;
            if constexpr (QUERY_LAYOUT == 1) { // BNSD: [B, N, S, D]
                // offset = batch * strideB + head * strideN + seq * strideS
                gmOffsetQO = bOffsetQO + qHeadIdx * strideQON + qSeqOffset * strideQOS;
            } else {
                // TND: [T, N, D]
                gmOffsetQO = bOffsetQO + qSeqOffset * strideQO + qHeadIdx * embed_;
            }
            if constexpr (KV_CACHE_LAYOUT == 1) { // BNSD: [B, N, S, D]
                // offset = batch * strideB + head * strideN
                // kv seq offset will be handled in blockMmadQK/blockMmadPV based on selectIdx
                gmOffsetKV = bOffsetKV + kvHeadIdx * strideKVN;
            } else {
                // TND: [T, N, D]
                gmOffsetKV = bOffsetKV + kvHeadIdx * embed_;
            }

            // 当前task处理的 actual qSeq Tile size
            uint32_t qSTileSizeAct =
                (qXBlockIdx < curQXBlockNum - 1) ?
                    ((qXBlockInnerIdx == qSTileNumInXBlock - 1) ? blockShapeX_ - qXBlockInnerIdx * qBaseTile_ :
                                                                  qBaseTile_) :
                    ((qXBlockInnerIdx == CeilDiv(xTailBlockLen, qBaseTile_) - 1) ?
                         xTailBlockLen - qXBlockInnerIdx * qBaseTile_ :
                         qBaseTile_);

            uint32_t rowNum = qSTileSizeAct;
            uint32_t rowNumRound = RoundUp(rowNum, 16);

            uint32_t kvSLoopNumTotal = (selectKvSeqLen + kvBaseTile_ - 1) / kvBaseTile_;
            uint32_t kvSTileSizeAct = kvBaseTile_;

#ifdef __DAV_CUBE__
            int64_t qShapeCol = 0;
            int64_t kvShapeCol = 0;
            if constexpr (QUERY_LAYOUT == 1) { // BNSD
                qShapeCol = strideQOS;         // embed
            } else {                           // TND
                qShapeCol = strideQO;          // qHeads * embed
            }
            if constexpr (KV_CACHE_LAYOUT == 1) {
                kvShapeCol = strideKVS; // embed
            } else {
                kvShapeCol = strideKV; // kvHeads * embed
            }

            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, qShapeCol);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQO], gmQLayoutTla, Arch::PositionGM{});
            blockMmadQK.loadQGM(gmQTensorTla, rowNum, embed_);

            auto gmKLayoutTla = tla::MakeLayout(
                tla::MakeShape(kvShapeCol, kvBaseTile_), tla::MakeStride(tla::Int<1>{}, (int64_t)kvShapeCol),
                tla::MakeShape(kvShapeCol, kvSeqlen));
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetKV], gmKLayoutTla, Arch::PositionGM{});
            auto gmVLayoutTla = tla::MakeLayout(
                tla::MakeShape(kvBaseTile_, kvShapeCol), tla::MakeStride((int64_t)kvShapeCol, tla::Int<1>{}),
                tla::MakeShape(kvSeqlen, kvShapeCol));
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetKV], gmVLayoutTla, Arch::PositionGM{});
#endif
#ifdef __DAV_VEC__
            int64_t oShapeCol = 0;
            if constexpr (QUERY_LAYOUT == 1) { // BNSD
                oShapeCol = strideQOS;         // embed
            } else {                           // TND
                oShapeCol = strideQO;          // qHeads * embed
            }
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetQO], gmOLayoutTla, Arch::PositionGM{});
#endif

            // task内(核内) 对 selectKvSeqLen 切为多个块, 循环处理
            // 总共 kvSLoopNumTotal 个块, 每个循环 (即每个 BlockMmad) 会处理 kv_seq 方向的长度为 kvSTileSizeAct
            for (uint32_t kvSTileIdx = 0; kvSTileIdx < kvSLoopNumTotal + PRE_LAUNCH; kvSTileIdx++) {
                if (kvSTileIdx < kvSLoopNumTotal) {
                    if (kvSTileIdx == kvSLoopNumTotal - 1) {
                        kvSTileSizeAct = selectKvSeqLen - kvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }

                    // Stage 1: QK matmul
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvSTileIdx % UB_S_OTMP_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(mm1ToSmFlagId); // flag: 0,1
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages = CalcMm1Mm2PrefixSumL0ABStages(
                        kvSTileIdx, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNumTotal, true);
                    uint64_t prefixSumL0BStages = CalcMm1Mm2PrefixSumL0ABStages(
                        kvSTileIdx, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNumTotal, true);
                    blockMmadQK(
                        gmKTensorTla, ubSTensorTla, gSelectIdx[curSelectNumIdx * maxKvBlockNum_], actualBlockShapeQK,
                        kvSTileIdx, kvSeqlen, kvBaseTile_, blockShapeY_, curSelectNum, curKvYBlockNum,
                        prefixSumL0AStages, prefixSumL0BStages, mm1ToSmFlag);
                    if (kvSTileIdx == kvSLoopNumTotal - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif

#ifdef __DAV_VEC__
                    // Stage 2: Online softmax
                    uint32_t l1PBufId = kvSTileIdx % P_L1_BUF_NUM;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES; // flag: 2,3,4
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, Catlass::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});

                    epilogueOnlineSoftmax(
                        l1PTensorTla, actualBlockShapeQK, (kvSTileIdx == 0), ubSBufId, l1PBufId, mm1ToSmFlag,
                        smToMm2Flag);
#endif
                }

                if (kvSTileIdx >= PRE_LAUNCH) {
                    uint32_t kvSIdxActual = kvSTileIdx - PRE_LAUNCH;
                    if (kvSIdxActual == kvSLoopNumTotal - 1) {
                        kvSTileSizeAct = selectKvSeqLen - kvSIdxActual * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // Stage 3: PV matmul
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvSIdxActual % UB_S_OTMP_BUF_STAGES;
                    // 核间同步flagId规律：每份dst对应一个id，从小到大顺序依次为 qk->sm, sm->pv, pv->rescale
                    uint32_t mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + P_L1_BUF_NUM; // flag: 5,6
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvSIdxActual % P_L1_BUF_NUM;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES; // flag: 2,3,4
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});

                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(mm2ToReFlagId);
                    uint64_t prefixSumL0AStages = CalcMm1Mm2PrefixSumL0ABStages(
                        kvSIdxActual, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNumTotal, false);
                    uint64_t prefixSumL0BStages = CalcMm1Mm2PrefixSumL0ABStages(
                        kvSIdxActual, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNumTotal, false);
                    blockMmadPV(
                        gmVTensorTla, ubOTmpTensorTla, gSelectIdx[curSelectNumIdx * maxKvBlockNum_], actualBlockShapePV,
                        kvSIdxActual, kvSeqlen, kvBaseTile_, blockShapeY_, curSelectNum, curKvYBlockNum,
                        prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag, mm2ToReFlag);
#endif
#ifdef __DAV_VEC__
                    // Stage 4: rescale O
                    Arch::CrossCoreFlag mm2ToReFlag(mm2ToReFlagId); // flag: 5,6
                    uint32_t curTileMod = kvSIdxActual % (PRE_LAUNCH + 1);
                    epilogueRescaleO(
                        gmOTensorTla, actualBlockShapePV, curTileMod, kvSIdxActual, (kvSIdxActual == 0),
                        (kvSIdxActual == kvSLoopNumTotal - 1), mm2ToReFlag);
#endif
                }
            } // end of for(kvSLoopNumTotal)
        } // end of for(totalTaskNum_)

        // release reverse sync flags
        ReleaseSyncFlags<4, 4, 4>();
    }

    __aicore__ inline void GetTilingData(__gm__ RfaTilingData* tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        firstQBlockNum_ = tilingData->firstQBlockNum;
        firstBatchTaskNum_ = tilingData->firstBatchTaskNum;
        totalTaskNum_ = tilingData->totalTaskNum;
        maxKvBlockNum_ = tilingData->maxKvBlockNum;
        blockShapeX_ = tilingData->blockShapeX;
        blockShapeY_ = tilingData->blockShapeY;
        scaleValue_ = tilingData->scaleValue;

        // basic tile
        qBaseTile_ = tilingData->qBaseTile;
        kvBaseTile_ = tilingData->kvBaseTile;

        maxQSeqlen_ = tilingData->maxQSeqlen;
        maxKvSeqlen_ = tilingData->maxKvSeqlen;
        isVarLen_ = tilingData->isVarLen;
    }

    __aicore__ inline void CalcBlockMmadL0Stages()
    {
        mm1L0ATotalStages_ = CeilDiv<BlockMmadQK::L0_TILE_M>(qBaseTile_) * CeilDiv<BlockMmadQK::L0_TILE_K>(embed_);
        mm1L0BTotalStages_ = CeilDiv<BlockMmadQK::L0_TILE_N>(kvBaseTile_) * CeilDiv<BlockMmadQK::L0_TILE_K>(embed_);
        mm2L0ATotalStages_ = CeilDiv<BlockMmadPV::L0_TILE_M>(qBaseTile_) * CeilDiv<BlockMmadPV::L0_TILE_K>(kvBaseTile_);
        mm2L0BTotalStages_ = CeilDiv<BlockMmadPV::L0_TILE_K>(kvBaseTile_) * CeilDiv<BlockMmadPV::L0_TILE_N>(embed_);
    }

    __aicore__ inline uint64_t CalcMm1Mm2PrefixSumL0ABStages(
        uint32_t kvSTileIdx, uint32_t mm1L0Stages, uint32_t mm2L0Stages, uint32_t kvSLoopNum, bool isMm1)
    {
        uint64_t prefixSumStages = 0;
        if (isMm1) {
            prefixSumStages = (kvSTileIdx <= PRE_LAUNCH) ?
                                  kvSTileIdx * mm1L0Stages :
                                  kvSTileIdx * mm1L0Stages + (kvSTileIdx - PRE_LAUNCH) * mm2L0Stages;
        } else {
            prefixSumStages = (kvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
                                  (kvSTileIdx + PRE_LAUNCH + 1) * mm1L0Stages + kvSTileIdx * mm2L0Stages :
                                  kvSLoopNum * mm1L0Stages + kvSTileIdx * mm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementP> (&l1PTensor)[P_L1_BUF_NUM],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
        AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < P_L1_BUF_NUM; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(L1_QK_SIZE + L1_P_BUF_SIZE * i);
        }

        uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        uint32_t rescaleCol = EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM;
        uint32_t spElemNum = rowNumPerSubCore * colNumPerSubCore;
        uint32_t rescaleElemNum = rowNumPerSubCore * rescaleCol;
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(spElemNum * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                spElemNum * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
                spElemNum * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES + rescaleElemNum * sizeof(ElementOTmp) * i);
        }
    }

    template <uint8_t MM1_SM_MODE, uint8_t MM2_RE_MODE, uint8_t SM_MM2_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // cross core sync
        if constexpr (SM_MM2_MODE == 4) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == 4) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(1);
        }
        if constexpr (MM2_RE_MODE == 4) {
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(6);
        }
#endif
    }

    template <uint8_t MM1_SM_MODE, uint8_t MM2_RE_MODE, uint8_t SM_MM2_MODE>
    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (MM1_SM_MODE == 4) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (MM2_RE_MODE == 4) {
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == 4) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    // tiling info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstQBlockNum_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t maxKvBlockNum_;
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    // the basic tile of Gemm::Block
    uint32_t qBaseTile_;  // min(blockShapeX_, 128)
    uint32_t kvBaseTile_; // 256
    bool isVarLen_;
    int64_t maxQSeqlen_;
    int64_t maxKvSeqlen_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
};

template <
    typename InputDtype = half,   // half, bfloat16_t
    typename SoftmaxDtype = half, // half, bfloat16_t
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
    uint32_t QueryLayout = 0,   // 0=TND, 1=BNSD
    uint32_t KvCacheLayout = 0> // 0=TND, 1=BNSD
CATLASS_GLOBAL void RainFusionAttention950(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR selectIdx,    // [queryBlockNum, headNum, maxKvBlockNum], int64_t
    GM_ADDR selectNumIdx, // [queryBlockNum, headNum], int64_t
    GM_ADDR blockShape, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR mask, GM_ADDR blockTable, GM_ADDR o,
    GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InputDtype;
    using ElementK = InputDtype;
    using ElementV = InputDtype;
    using ElementS = SoftmaxDtype;
    using ElementP = InputDtype;
    using ElementO = InputDtype;
    using ElementOTmp = float; // RescaleO dtype

    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1(ub2L1)
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    //  layout of PV result. OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;

    // QK matmul
    using L1TileShapeQK = tla::Shape<Int<128>, Int<256>, Int<128>>;
    using L0TileShapeQK = tla::Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5RainQK<ArchTag>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK, ElementS, void, TileCopyQK>;

    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA5OnlineSoftmax;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    // PV matmul
    using L1TileShapePV = tla::Shape<Int<128>, Int<128>, Int<256>>;
    using L0TileShapePV = tla::Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5RainPV<ArchTag>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;

    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5RescaleO<lseMode>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementK, TileCopyRescaleO>;

    using RfaKernelArch35 = RfaKernelArch35<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, false, QueryLayout, KvCacheLayout>;
    RfaKernelParamsArch35 params{
        q, k, v, mask, blockTable, actualQseqlen, actualKvseqlen, selectIdx, selectNumIdx, o, lse, workspace, tiling};
    RfaKernelArch35 rfaKernelArch35;
    rfaKernelArch35(params);
}
