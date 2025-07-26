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

#include "kernel_common.hpp"

using namespace Catlass;

/*
This example demonstrates how to compute fa.
*/
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueOnlineSoftmax,
    class EpilogueRescaleO>
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

    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;


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
        GM_ADDR workSpace;
        GM_ADDR tiling;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
               GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR workSpace_, GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
              actualKvseqlen(actualKvseqlen_), o(o_), workSpace(workSpace_), tiling(tiling_) {}
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
        uint32_t totalTaskNum = 0;
        uint32_t firstBatchTaskNum = 0;
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
        uint32_t qRowNumCeil = 128;
        uint32_t qNBlockTile = qRowNumCeil / qSeqlen;
        qNBlockTile = std::min(qNBlockTile, groupSize);
        qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
        return qNBlockTile;
    }

    CATLASS_DEVICE
    uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = (kvSeqlen <= 4096) ? 128 : 256;
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
        uint32_t curTotalTaskNum = fATilingData->firstBatchTaskNum;
        uint32_t totalTaskNum = fATilingData->totalTaskNum;
        uint32_t blockSize = fATilingData->blockSize;
        float scaleValue = fATilingData->scaleValue;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.workSpace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.workSpace + mm1OutSize);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.workSpace + mm1OutSize + smOnlineOutSize);

        uint64_t strideQO = qHeads * embed;
        uint64_t strideKV = kvHeads * embed;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t qBOffset = 0;
        uint32_t kBOffset = 0;
        uint32_t vBOffset = 0;
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint32_t curQNBlockTile;
        uint32_t qNBlockPerGroup;
        uint32_t curQNBlockNum;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
            while (taskIdx >= curTotalTaskNum) {
                curBatch++;
                preTotalTaskNum = curTotalTaskNum;
                qSeqlen = reinterpret_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                kvSeqlen = reinterpret_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                curQSBlockTile = GetQSBlockTile(kvSeqlen);
                curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize, curQSBlockTile);
                qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                qBOffset += qSeqlen * strideQO;
                if constexpr (!PAGED_CACHE_FLAG) {
                    kBOffset += kvSeqlen * strideKV;
                    vBOffset += kvSeqlen * strideKV;
                }
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
            uint32_t gmQOffset = qBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;
            uint32_t gmKOffset = kBOffset;
            uint32_t gmVOffset = vBOffset;
            if constexpr (!PAGED_CACHE_FLAG) {
                gmKOffset = kBOffset + kvHeadIdx * embed;
                gmVOffset = vBOffset + kvHeadIdx * embed;
            }
            uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1)) ?
                (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = AlignUp(rowNum, BLOCK_SIZE);
            uint32_t noSkipKvS = kvSeqlen;
            uint32_t noMaskKvS = kvSeqlen;
            uint32_t noMaskTailS;
            if (params.mask != nullptr) {
                uint32_t diffS = kvSeqlen - qSeqlen;
                noMaskKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                noSkipKvS = noMaskKvS + qSBlockSize;
            }
            uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
            uint32_t kvSLoopNumTotal = kvSLoopNumNoMask + CeilDiv(qSBlockSize, pagedBlockSize);
            uint32_t blockStackNum = (kvSeqlen <= 4096) ? 2 : 4;
            uint32_t stackSeqCount = 0;
            uint32_t stackSeqTile;
            uint32_t preLaunch = 1;
            uint32_t stackSeqTileRound = blockStackNum * 128;
            uint32_t preKVNum = preLaunch * blockStackNum;
            LayoutQ layoutQ(qSeqlen, strideQO);
            LayoutK layoutK(strideKV, blockStackNum * pagedBlockSize);
            LayoutV layoutV(blockStackNum * pagedBlockSize, strideKV);
            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
                if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                    stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                } else {
                    stackSeqTile = pagedBlockSize * blockStackNum;
                }
                uint32_t SWorkSpacePingPongFlag = kvSIdx % (preLaunch + 1);
                uint32_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                LayoutS layOutS(rowNum, stackSeqTile, stackSeqTileRound);
                if constexpr (PAGED_CACHE_FLAG) {
                    blockMmadQK(gQ[gmQOffset], gB, gS[gmSOffset],
                        gBlockTable, LayoutQ, LayoutK, layOutS,
                        kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV);
                } else {
                    blockMmadQK(gQ[gmQOffset], gB[gmKOffset], gS[gmSOffset],
                        gBlockTable, LayoutQ, LayoutK, layOutS,
                        kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV);
                    gmKOffset += stackSeqTile * strideKV;
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(qkReady);
                if (kvSIdx >= preKVNum) {
                    uint32_t nowkvSIdx = kvSIdx - preKVNum;
                    if (nowkvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    uint32_t PVWorkSpacePingPongFlag = nowkvSIdx % (preLaunch + 1);
                    uint32_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    uint32_t gmOTmpOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (preLaunch + 1) + PVWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    if constexpr (PAGED_CACHE_FLAG) {
                        blockMmadPV(gP[gmPOffset], gV, gOTmp[gmOTmpOffset],
                            gBlockTable, layoutP, LayoutV, LayoutOTmp,
                            nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV, softmaxReady);
                    } else {
                        blockMmadPV(gP[gmPOffset], gV[gmVOffset], gOTmp[gmOTmpOffset],
                            gBlockTable, layoutP, LayoutV, LayoutOTmp,
                            nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS, strideKV, softmaxReady);
                        gmVOffset += stackSeqTile * strideKV;
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(pvReady);
                }
            }
            uint32_t maskTailS = 0;
            for (uint32_t kvSIdx = kvSLoopNumNoMask; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                if (kvSIdx < kvSLoopNumTotal) {
                    if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    if constexpr (PAGED_CACHE_FLAG) {
                        if (kvSIdx == kvSLoopNumNoMask) {
                            stackSeqTile -= pagedBlockSize - (noMaskKvS - kvSLoopNumNoMask * pagedBlockSize);
                        }
                    }
                    uint32_t SWorkSpacePingPongFlag = kvSIdx % (preLaunch + 1);
                    uint32_t gmSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 + SWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTileRound);
                    if constexpr (PAGED_CACHE_FLAG) {
                        if (kvSIdx == kvSLoopNumNoMask) {
                            maskTailS = kvSLoopNumNoMask * pagedBlockSize - noMaskKvS;
                        } else {
                            maskTailS = 0;
                        }
                        blockMmadQK(gQ[gmQOffset], gB, gS[gmSOffset],
                            gBlockTable, LayoutQ, LayoutK, LayoutS,
                            kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noSkipKvS, strideKV, maskTailS);
                    } else {
                        blockMmadQK(gQ[gmQOffset], gB[gmKOffset], gS[gmSOffset],
                            gBlockTable, LayoutQ, LayoutK, LayoutS,
                            kvSIdx, kvSLoopNumNoMask, pagedBlockSize, noSkipKvS, strideKV);
                        gmKOffset += stackSeqTile * strideKV;
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(qkReady);
                }
                if (kvSIdx >= preKVNum) {
                    uint32_t nowkvSIdx = kvSIdx - preKVNum;
                    if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }
                    if constexpr (PAGED_CACHE_FLAG) {
                        if (nowkvSIdx == kvSLoopNumNoMask) {
                            stackSeqTile -= pagedBlockSize - (noMaskKvS - kvSLoopNumNoMask * pagedBlockSize);
                        }
                    }
                    uint32_t PWorkSpacePingPongFlag = nowkvSIdx % (preLaunch + 1);
                    uint32_t gmPOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 + // cube core offset
                                         PWorkSpacePingPongFlag * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    if constexpr (PAGED_CACHE_FLAG) {
                        if (nowkvSIdx == kvSLoopNumNoMask) {
                            maskTailS = kvSLoopNumNoMask * pagedBlockSize - noMaskKvS;
                        } else {
                            maskTailS = 0;
                        }
                        blockMmadPV(gQ[gmQOffset], gB, gP[gmPOffset],
                            gBlockTable, layoutP, LayoutV, LayoutOTmp,
                            nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noSkipKvS, strideKV, softmaxReady, maskTailS);
                    } else {
                        blockMmadPV(gQ[gmQOffset], gB[gmVOffset], gP[gmPOffset],
                            gBlockTable, layoutP, LayoutV, LayoutOTmp,
                            nowkvSIdx, kvSLoopNumNoMask, pagedBlockSize, noSkipKvS, strideKV, softmaxReady);
                        gmVOffset += stackSeqTile * strideKV;
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(pvReady);
                }
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
        // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
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
        float scaleValue = fATilingData->scaleValue;
        
        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementMask> gMask;
        gO.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gO.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gO.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.workSpace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.workSpace + mm1OutSize);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.workSpace + mm1OutSize + smOnlineOutSize);
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.workSpace + mm1OutSize + smOnlineOutSize + mm2OutSize);

        uint32_t groupSize = qHeads / kvHeads;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
        EpilogueRescaleO epilogueRescaleO(resource);

        
        uint32_t curTotalTaskNum = firstBatchTaskNum;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t oBatchOffset = 0;
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint32_t curQNBlockTile;
        uint32_t qNBlockPerGroup;
        uint32_t curQNBlockNum;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        // Go through each task.
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
            // Get the offset of each core on the GM.
            while (taskIdx >= curTotalTaskNum) {
                curBatch++;
                preTotalTaskNum = curTotalTaskNum;
                qSeqlen = reinterpret_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                kvSeqlen = reinterpret_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                curQSBlockTile = GetQSBlockTile(kvSeqlen);
                curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                oBatchOffset += qSeqlen * qHeads * embed;
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

            uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen % curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1)) ?
                (groupSize % curQNBlockTile) : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = AlignUp(rowNum, BLOCK_SIZE);

            uint32_t noSkipKvS = kvSeqlen;
            uint32_t noMaskKvS = kvSeqlen;
            uint32_t noMaskTailS;
            if (params.mask != nullptr) {
                uint32_t diffS = kvSeqlen - qSeqlen;
                noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                noSkipKvS = std::min(kvSeqlen, noSkipKvS);
                noMaskKvS = noSkipKvS - qSBlockSize;
                noMaskTailS = noMaskKvS % pagedBlockSize;
                // maskedHeadS = (pagedBlockSize - noMaskTailS) % pagedBlockSize;
            }
            uint32_t maskedKvS = qSBlockSize;
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
            uint32_t kvSLoopNumNoMask = CeilDiv(noMaskKvS, pagedBlockSize);
            uint32_t blockStackNum = (kvSeqlen <= 4096) ? 1 : 2;
            uint32_t stackSeqCount = 0;
            uint32_t stackSeqTile;
            uint32_t preLaunch = 1;
            // no mask kvSeqlen loop
            for (uint32_t kvSIdx = 0;
                kvSIdx < kvSLoopNumNoMask;
                kvSIdx += blockStackNum) {
                if (kvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                    stackSeqTile = noMaskKvS - kvSIdx * pagedBlockSize;
                } else {
                    stackSeqTile = pagedBlockSize * blockStackNum;
                }
                uint32_t stackSeqTileRound = AlignUp(stackSeqTile, BLOCK_SIZE);
                LayoutS layOutS(rowNum, stackSeqTile, stackSeqTileRound);
                LayoutP layOutP(rowNum, stackSeqTile, stackSeqTileRound);
                GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 + // cube core offset
                    curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                // vec core offset will be processed within epilogue block
                uint32_t gmOffsetP = gmOffsetS;
                Arch::CrossCoreWaitFlag(qkReady);
                // online softmax
                epilogueOnlineSoftmax(
                    gP[gmOffsetP],
                    gS[gmOffsetS],
                    layOutP,
                    layOutS,
                    actualBlockShapeQK,
                    kvSIdx,
                    qSBlockSize,
                    qNBlockSize,
                    curStackTileMod);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);

                if (kvSIdx >= preLaunch * blockStackNum) {
                    uint32_t delayedKvSIdx = kvSIdx - preLaunch * blockStackNum;
                    stackSeqTile = pagedBlockSize * blockStackNum;
                    LayoutO layoutO(qSBlockSize, embed * qHeads);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                    uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(
                        gO[gmOffsetO],
                        gOTmp[gmOffsetOTmp],
                        layoutO,
                        layoutOTmp,
                        actualBlockShapePV,
                        delayedKvSIdx,
                        qSBlockSize,
                        qNBlockSize,
                        (delayedKvSIdx == 0),
                        0,
                        curStackTileMod);
                }
                stackSeqCount++;
            }
            uint32_t maskedStartIdx = (noMaskTailS == 0) ? kvSLoopNumNoMask : kvSLoopNumNoMask - 1;
            uint32_t maskedEndIdx = noMaskKvS / pagedBlockSize;
            // masked kvSeqlen loop
            for (uint32_t kvSIdx = maskedStartIdx;
                kvSIdx <= maskedEndIdx + preLaunch * blockStackNum;
                kvSIdx += blockStackNum) {
                if ((kvSIdx <= maskedEndIdx) && (kvSIdx < kvSLoopNumTotal)) {
                    // if (kvSIdx + blockStackNum >= maskedEndIdx) {
                    //     stackSeqTile = maskedKvS;
                    // } else {
                    //     stackSeqTile = pagedBlockSize * blockStackNum;
                    // }
                    stackSeqTile = maskedKvS;
                    uint32_t stackSeqTileRound = AlignUp(stackSeqTile, BLOCK_SIZE);
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    uint32_t curStackTileMod = stackSeqCount % (preLaunch + 1);
                    uint32_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 + // cube core offset
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB; // single cube core db offset
                    // vec core offset will be processed within epilogue block
                    uint32_t gmOffsetP = gmOffsetS;
                    Arch::CrossCoreWaitFlag(qkReady);
                    // online softmax
                    epilogueOnlineSoftmax(
                        gP[gmOffsetP],
                        gS[gmOffsetS],
                        layOutP,
                        layOutS,
                        actualBlockShapeQK,
                        kvSIdx,
                        qSBlockSize,
                        qNBlockSize,
                        curStackTileMod);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                }

                uint32_t delayedKvSIdx = kvSIdx - preLaunch * blockStackNum;
                if ((delayedKvSIdx >= 0) && (delayedKvSIdx < kvSLoopNumTotal)) {
                    if (delayedKvSIdx + blockStackNum >= maskedEndIdx) {
                        stackSeqTile = maskedKvS;
                    } else if (delayedKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) {
                        stackSeqTile = noMaskKvS - delayedKvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = pagedBlockSize * blockStackNum;
                    }

                    LayoutO layoutO(qSBlockSize, embed * qHeads);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                    uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 +
                        curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(
                        gO[gmOffsetO],
                        gOTmp[gmOffsetOTmp],
                        layoutO,
                        layoutOTmp,
                        actualBlockShapePV,
                        kvSIdx,
                        qSBlockSize,
                        qNBlockSize,
                        (delayedKvSIdx == 0),
                        (delayedKvSIdx + blockStackNum >= maskedEndIdx),
                        curStackTileMod);
                }
                
                // if (kvSIdx + blockStackNum > kvSLoopNumTotal + preLaunch * blockStackNum - 1) {
                //     stackSeqTile = kvSeqlen - (kvSIdx - preLaunch * blockStackNum) * pagedBlockSize;
                // } else if (kvSIdx + blockStackNum > kvSLoopNumNoMask + preLaunch * blockStackNum - 1) {
                //     stackSeqTile = noMaskKvS - (kvSIdx - preLaunch * blockStackNum) * pagedBlockSize;
                // } else {
                //     stackSeqTile = pagedBlockSize * blockStackNum;
                // }
                // LayoutO layoutO(qSBlockSize, embed * qHeads);
                // LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                // GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                // uint32_t curStackTileMod = (stackSeqCount - preLaunch) % (preLaunch + 1);
                // uint32_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * 2 +
                //         curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                // Arch::CrossCoreWaitFlag(pvReady);
                // // rescale O
                // epilogueRescaleO(
                //     gO[gmOffsetO],
                //     gOTmp[gmOffsetOTmp],
                //     layoutO,
                //     layoutOTmp,
                //     actualBlockShapePV,
                //     kvSIdx,
                //     qSBlockSize,
                //     qNBlockSize,
                //     0,
                //     (kvSIdx + blockStackNum > kvSLoopNumTotal + preLaunch * blockStackNum - 1),
                //     curStackTileMod);
                stackSeqCount++;
            }
            
            
            
            // for (uint32_t nIdx = 0; nIdx < nLoop + 1; nIdx++) {
            //     if (nIdx != nLoop) {
            //         if (nIdx == (nLoop - 1)) {
            //             // Calculate the size of the tail block
            //             kSeqTile = (curKVSeqlen - nIdx * seqTile);
            //             kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
            //         }
            //         // Wait for Q * K calculation to complete
            //         Arch::CrossCoreWaitFlag(qkReady);
            //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);

            //         LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound);
            //         LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
            //         GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
            //         uint32_t softmaxPingPongFlag = nIdx % 2;
            //         uint64_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE + softmaxPingPongFlag * TMP_SIZE / 2;
            //         uint64_t gmOffsetS =
            //             (uint64_t)coreIdx * TMP_SIZE_DECODER + softmaxPingPongFlag * TMP_SIZE_DECODER / 2;
            //         // Softmax one-stage calculation
            //         epilogueMLASoftmax(
            //             gP[gmOffsetP], gS[gmOffsetS],
            //             layoutP, layoutS,
            //             actualBlockShapeQK,
            //             nIdx, qHeadSplitSizeActual, softmaxPingPongFlag, glFlag);
            //         Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
            //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            //     }

            //     if (nIdx != 0) {
            //         // Calculate the size of the tail block
            //         if (nIdx == nLoop) {
            //             vSeqTile = (curKVSeqlen - (nIdx - 1) * seqTile);
            //             vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);
            //         }
            //         // Wait for P * V calculation to complete
            //         Arch::CrossCoreWaitFlag(pvReady);

            //         LayoutO layoutO(tokenNumPerHead, strideQO);
            //         LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
            //         LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
            //         GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
            //         uint32_t rescaleOPingPongFlag = (nIdx - 1) % 2;
            //         uint64_t gmOffsetOTmp = (uint64_t)(coreIdx * TMP_SIZE * 2 + rescaleOPingPongFlag * TMP_SIZE);
            //         uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * TMP_SIZE);
            //         uint32_t isLastNTile = (nIdx == nLoop) ? 1 : 0;
            //         // Softmax two-stage update
            //         epilogueMLARescaleO(
            //             gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO],
            //             gOCoreTmp[oFdOffset], gl[lOffset],
            //             layoutOTmp, layoutO, layoutUpdate,
            //             actualBlockShapePV,
            //             nIdx, isLastNTile, qHeadSplitSizeActual, rescaleOPingPongFlag, glFlag);
            //     }
            // }
        }

        
        // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
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
                                GM_ADDR workSpace,
                                GM_ADDR actualQseqlen,
                                GM_ADDR actualKvseqlen,
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
    using L1TileShape = GemmShape<128, 128, 576>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash Attention Infer的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
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
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2FARescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType>;


    // Kernel level
    using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO>;
    typename FAInferKernel::Params params{q, k, v, mask, blockTables,
                                          actualQseqlen, actualKvseqlen, o, workSpace, tiling};

    // call kernel
    FAInferKernel flashAttnInfer;
    flashAttnInfer(params);
}


// CATLASS_GLOBAL void MLABf16(uint64_t fftsAddr,
//                         GM_ADDR q,
//                         GM_ADDR qRope,
//                         GM_ADDR k,
//                         GM_ADDR kRope,
//                         GM_ADDR blockTables,
//                         GM_ADDR o,
//                         GM_ADDR s,
//                         GM_ADDR p,
//                         GM_ADDR oTmp,
//                         GM_ADDR oUpdate,
//                         GM_ADDR oCoreTmp,
//                         GM_ADDR l,
//                         GM_ADDR tiling)
// {
//     // Set FFTS address
//     AscendC::SetSyncBaseAddr(fftsAddr);

//     using ArchTag = Arch::AtlasA2;
//     using ElementQ = __bf16;
//     using LayoutQ = layout::RowMajor;
//     using ElementK = __bf16;
//     using LayoutK = layout::ColumnMajor;
//     using ElementV = __bf16;
//     using LayoutV = layout::RowMajor;
//     using ElementS = float;
//     using LayoutS = layout::RowMajor;
//     using ElementP = __bf16;
//     using LayoutP = layout::RowMajor;
//     using ElementO = __bf16;
//     using LayoutO = layout::RowMajor;
//     using ElementMask = __bf16;
//     using LayoutMask = layout::RowMajor;
//     using ElementOTmp = float;
//     using LayoutOTmp = layout::RowMajor;
//     using ElementUpdate = float;
//     using LayoutUpdate = layout::RowMajor;

//     // L1TileShape::K must be embdding
//     using L1TileShape = GemmShape<128, 128, 576>;
//     using L0TileShape = L1TileShape;

//     // GEMM Block模块，实现Flash MLA的Q * K^T
//     using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
//     using QType = Gemm::GemmType<ElementQ, LayoutQ>;
//     using KType = Gemm::GemmType<ElementK, LayoutK>;
//     using SType = Gemm::GemmType<ElementS, LayoutS>;
//     using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

//     // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
//     using PType = Gemm::GemmType<ElementP, LayoutP>;
//     using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
//     using EpilogueMLASoftmax =
//         Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

//     // GEMM Block模块，实现Flash MLA的P * V
//     using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
//     using VType = Gemm::GemmType<ElementV, LayoutV>;
//     using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
//     using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

//     // Epilogue Block模块，实现Flash MLA中当前O基块的更新
//     using OType = Gemm::GemmType<ElementO, LayoutO>;
//     using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
//     using EpilogueMLARescaleO =
//         Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

//     // Epilogue Block模块，实现Flash MLA中flash decoding
//     using OType = Gemm::GemmType<ElementO, LayoutO>;
//     using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
//     constexpr uint32_t ComputeEleNum = 6144;
//     using EpilogueMLAFDRescaleO =
//         Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

//     // Kernel level
//     using FAKernel = FAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax, EpilogueMLARescaleO>;
//     typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

//     // call kernel
//     MLAKernel mla;
//     mla(params);
// }