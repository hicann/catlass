/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/layout/layout.hpp"

#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/matmul_type.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"

#include "AscendCT/arch/resource.hpp"
#include "AscendCT/arch/cross_core_sync.hpp"
#include "AscendCT/epilogue/block/block_epilogue.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"

#include "AscendCT/status.hpp"
#include "AscendCT/gemm/device/matmul_universal_adapter.hpp"
#include "helper.hpp"

using namespace AscendCT;

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t TILING_HEAD_NUM = 16;
constexpr uint32_t TILING_PARA_NUM = 24;
constexpr uint32_t ALIGNED = 16;
constexpr uint32_t WORKSPACE_ELENUM = 32768;          // 128 * 256

/*
This example demonstrates how to compute fa.
*/
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueFASoftmax,
    class EpilogueFARescaleO
>
class FAKernel {
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
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using LayoutOTmp = typename BlockMmadPV::LayoutC;

    using ElementMask = typename EpilogueFASoftmax::ElementMask;
    using LayoutMask = typename EpilogueFASoftmax::LayoutMask;

    using ElementO = typename EpilogueFARescaleO::ElementOutput;
    using LayoutO = typename EpilogueFARescaleO::LayoutOutput;

    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR mask;
        GM_ADDR o;
        GM_ADDR s;
        GM_ADDR p;
        GM_ADDR oTmp;
        GM_ADDR tiling;

        // Methods
        ASCENDCT_HOST_DEVICE
        Params() {}

        ASCENDCT_HOST_DEVICE
        Params(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR o_, GM_ADDR s_,
               GM_ADDR p_, GM_ADDR oTmp_, GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), mask(mask_), o(o_), s(s_), p(p_), oTmp(oTmp_), tiling(tiling_) {}
    };

    struct Arguments {
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR mask;
        GM_ADDR o;
        GM_ADDR s;
        GM_ADDR p;
        GM_ADDR oTmp;
        GM_ADDR tiling;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.q, args.k, args.v, args.mask, args.o, args.s, args.p, args.oTmp, args.tiling};
        return params;
    }

    // Methods
    ASCENDCT_DEVICE
    FAKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    ASCENDCT_DEVICE
    void operator()(Params const &params);

    template <>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        // Represent the full gm
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);
        AscendC::GlobalTensor<float> gTilingFloat;
        gTilingFloat.SetGlobalBuffer((__gm__ float *)params.tiling);
        AscendC::GlobalTensor<int64_t> gTilingI64;
        gTilingI64.SetGlobalBuffer((__gm__ int64_t *)params.tiling);

        // get tiling
        uint32_t batchSize = gTiling.GetValue(0);
        uint32_t maxSeqlen = gTiling.GetValue(1);
        uint32_t qHead = gTiling.GetValue(2);
        uint32_t embd = gTiling.GetValue(3);
        uint32_t kvHead = gTiling.GetValue(4);
        half tor = (half)gTilingFloat.GetValue(5);
        uint32_t batchMask = gTiling.GetValue(6);
        uint32_t maskStride = gTiling.GetValue(7);
        uint32_t isTriuMask = gTiling.GetValue(8);
        uint32_t totalQTileNum = gTiling.GetValue(9);

        uint32_t embdAligned = RoundUp<ALIGNED>(embd);

        uint32_t headPerGroup = qHead / kvHead;
        int64_t strideQO = qHead * embd;
        int64_t strideKV = kvHead * embd;

        uint32_t mTile = L1TileShape::M;
        uint32_t nTile = L1TileShape::N;

        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, 4 * L1TileShape::M * L1TileShape::K * sizeof(ElementQ));

        uint32_t blockMmadpingpongFlag = 0;
        uint32_t sPingpongFlag = 0;
        uint32_t pPingpongFlag = 0;
        uint32_t oTmpPingpongFlag = 0;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        uint32_t curBatch = 0;
        uint32_t preBatchTotalQTileNum = 0;
        uint32_t tilingOffset = TILING_HEAD_NUM + curBatch * TILING_PARA_NUM;
        uint32_t curBatchTotalQTileNum = gTiling.GetValue(tilingOffset + 10);
        uint32_t processNum = totalQTileNum * qHead;
        for (uint32_t process = 0; process < processNum; process++) {
            if (process >= curBatchTotalQTileNum * qHead) {
                while (1) {
                    curBatch++;
                    preBatchTotalQTileNum = curBatchTotalQTileNum;
                    tilingOffset += TILING_PARA_NUM;
                    curBatchTotalQTileNum = gTiling.GetValue(tilingOffset + 10);
                    uint32_t qSeqlen = gTiling.GetValue(tilingOffset);
                    if (qSeqlen != 0) {
                        break;
                    }
                }
            }
            uint32_t curCoreIdx = process % coreNum;
            if (isTriuMask) {
                // 奇数轮反向分发
                if ((process / coreNum) % 2 == 1) {
                    curCoreIdx = coreNum - process % coreNum - 1;
                }
            }
            if (coreIdx != curCoreIdx) {
                continue;
            }

            // get tiling args
            uint32_t qSeqlen = gTiling.GetValue(tilingOffset);
            uint32_t kvSeqlen = gTiling.GetValue(tilingOffset + 1);
            int64_t qAddr = gTilingI64.GetValue((tilingOffset + 2) / 2);
            int64_t kAddr = gTilingI64.GetValue((tilingOffset + 4) / 2);
            int64_t vAddr = gTilingI64.GetValue((tilingOffset + 6) / 2);

            LayoutQ layoutQ(qSeqlen, strideQO);
            LayoutK layoutK(strideKV, kvSeqlen);
            LayoutV layoutV(kvSeqlen, strideKV);

            uint32_t processIdx = process - preBatchTotalQTileNum * qHead;
            uint32_t mIdx = processIdx / qHead;
            uint32_t headIdx = processIdx % qHead;

            // Compute initial location in logical coordinates
            MatrixCoord offsetQ{mIdx * mTile, headIdx * embd};
            MatrixCoord offsetK{headIdx / headPerGroup * embd, 0};
            MatrixCoord offsetV{0, headIdx / headPerGroup * embd};
            MatrixCoord blockOffsetK({0, nTile});
            MatrixCoord blockOffsetV({nTile, 0});

            uint32_t mLoop = (qSeqlen + mTile - 1) / mTile;
            uint32_t nLoop = (kvSeqlen + nTile - 1) / nTile;
            uint32_t m = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * mTile) : mTile;

            // preload
            uint32_t qkNIdx = 0;
            uint32_t qkN = (qkNIdx == (nLoop - 1)) ? kvSeqlen : nTile;
            uint32_t qkNRound = (qkN + ALIGNED - 1) / ALIGNED * ALIGNED;

            MatmulCoord actualBlockShapeQK{m, qkN, embd};
            LayoutS layoutS(m, qkN, qkNRound);
            int64_t gmOffsetQ = qAddr + layoutQ.GetOffset(offsetQ);
            int64_t gmOffsetK = kAddr + layoutK.GetOffset(offsetK);
            int64_t gmOffsetS = (int64_t)coreIdx * WORKSPACE_ELENUM + sPingpongFlag * WORKSPACE_ELENUM / 2;
            blockMmadQK(
                gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS],
                layoutQ, layoutK, layoutS,
                actualBlockShapeQK, blockMmadpingpongFlag, 1);
            arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
            sPingpongFlag = 1 - sPingpongFlag;
            qkNIdx++;

            // mainloop
            uint32_t nEnd = nLoop;
            if (isTriuMask) {
                nEnd = mIdx + 1;
            }
            uint32_t svN = nTile;
            uint32_t svNRound = nTile;
            for (uint32_t svNIdx = 0; svNIdx < nEnd; svNIdx++) {
                if (qkNIdx < nEnd) {
                    if (qkNIdx == (nLoop - 1)) {
                        qkN = kvSeqlen - qkNIdx * nTile;
                        qkNRound = (qkN + ALIGNED - 1) / ALIGNED * ALIGNED;
                    }

                    MatmulCoord actualBlockShapeQKNext{m, qkN, embd};
                    LayoutS layoutSNext(m, qkN, qkNRound);
                    offsetK += blockOffsetK;
                    gmOffsetK = kAddr + layoutK.GetOffset(offsetK);
                    gmOffsetS = (int64_t)coreIdx * WORKSPACE_ELENUM + sPingpongFlag * WORKSPACE_ELENUM / 2;
                    blockMmadQK(
                        gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS],
                        layoutQ, layoutK, layoutSNext,
                        actualBlockShapeQKNext, blockMmadpingpongFlag, 0);
                    arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                    sPingpongFlag = 1 - sPingpongFlag;
                    qkNIdx++;
                }

                if (svNIdx == (nLoop - 1)) {
                    svN = kvSeqlen - svNIdx * nTile;
                    svNRound = (svN + ALIGNED - 1) / ALIGNED * ALIGNED;
                }

                MatmulCoord actualBlockShapePV{m, embd, svN};
                LayoutP layoutP(m, svN, svNRound);
                LayoutOTmp layoutOTmp(m, embd, embdAligned);
                int64_t gmOffsetP = (int64_t)coreIdx * WORKSPACE_ELENUM + pPingpongFlag * WORKSPACE_ELENUM / 2;
                int64_t gmOffsetV = vAddr + layoutV.GetOffset(offsetV);
                int64_t gmOffsetOTmp = (int64_t)coreIdx * WORKSPACE_ELENUM + oTmpPingpongFlag * WORKSPACE_ELENUM / 2;
                blockMmadPV(
                    gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp],
                    layoutP, layoutV, layoutOTmp,
                    actualBlockShapePV, blockMmadpingpongFlag, softmaxReady);
                offsetV += blockOffsetV;
                pPingpongFlag = 1 - pPingpongFlag;
                oTmpPingpongFlag = 1 - oTmpPingpongFlag;
                arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
            }
        }
    }

    template <>
    ASCENDCT_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        // Represent the full gm
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);
        AscendC::GlobalTensor<float> gTilingFloat;
        gTilingFloat.SetGlobalBuffer((__gm__ float *)params.tiling);
        AscendC::GlobalTensor<int64_t> gTilingI64;
        gTilingI64.SetGlobalBuffer((__gm__ int64_t *)params.tiling);

        // get tiling
        uint32_t batchSize = gTiling.GetValue(0);
        uint32_t maxSeqlen = gTiling.GetValue(1);
        uint32_t qHead = gTiling.GetValue(2);
        uint32_t embd = gTiling.GetValue(3);
        uint32_t kvHead = gTiling.GetValue(4);
        half tor = (half)gTilingFloat.GetValue(5);
        uint32_t batchMask = gTiling.GetValue(6);
        uint32_t maskStride = gTiling.GetValue(7);
        uint32_t isTriuMask = gTiling.GetValue(8);
        uint32_t totalQTileNum = gTiling.GetValue(9);

        uint32_t embdAligned = RoundUp<ALIGNED>(embd);

        uint32_t headPerGroup = qHead / kvHead;
        int64_t strideQO = qHead * embd;
        int64_t strideKV = kvHead * embd;

        uint32_t mTile = L1TileShape::M;
        uint32_t nTile = L1TileShape::N;

        EpilogueFASoftmax epilogueFASoftmax(resource, tor);
        EpilogueFARescaleO epilogueFARescaleO(resource);

        LayoutMask layoutMask(maxSeqlen, maxSeqlen);
        uint32_t sPingpongFlag = 0;
        uint32_t pPingpongFlag = 0;
        uint32_t oTmpPingpongFlag = 0;

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        uint32_t curBatch = 0;
        uint32_t preBatchTotalQTileNum = 0;
        uint32_t tilingOffset = TILING_HEAD_NUM + curBatch * TILING_PARA_NUM;
        uint32_t curBatchTotalQTileNum = gTiling.GetValue(tilingOffset + 10);
        uint32_t processNum = totalQTileNum * qHead;
        for (uint32_t process = 0; process < processNum; process++) {
            if (process >= curBatchTotalQTileNum * qHead) {
                while (1) {
                    curBatch++;
                    preBatchTotalQTileNum = curBatchTotalQTileNum;
                    tilingOffset += TILING_PARA_NUM;
                    curBatchTotalQTileNum = gTiling.GetValue(tilingOffset + 10);
                    uint32_t qSeqlen = gTiling.GetValue(tilingOffset);
                    if (qSeqlen != 0) {
                        break;
                    }
                }
            }
            uint32_t curCoreIdx = process % coreNum;
            if (isTriuMask) {
                // 奇数轮反向分发
                if ((process / coreNum) % 2 == 1) {
                    curCoreIdx = coreNum - process % coreNum - 1;
                }
            }
            if (coreIdx != curCoreIdx) {
                continue;
            }

            // get tiling args
            uint32_t qSeqlen = gTiling.GetValue(tilingOffset);
            uint32_t kvSeqlen = gTiling.GetValue(tilingOffset + 1);
            int64_t oAddr = gTilingI64.GetValue((tilingOffset + 8) / 2);

            LayoutO layoutO(qSeqlen, strideQO);

            uint32_t processIdx = process - preBatchTotalQTileNum * qHead;
            uint32_t mIdx = processIdx / qHead;
            uint32_t headIdx = processIdx % qHead;

            uint32_t mLoop = (qSeqlen + mTile - 1) / mTile;
            uint32_t nLoop = (kvSeqlen + nTile - 1) / nTile;
            uint32_t m = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * mTile) : mTile;

            LayoutOTmp layoutOTmp(m, embd, embdAligned);
            int64_t maskBatchOffset = (batchMask == 1) ? curBatch * maskStride * maxSeqlen : 0;

            // preload
            uint32_t qkNIdx = 0;
            uint32_t qkN = (qkNIdx == (nLoop - 1)) ? kvSeqlen : nTile;
            uint32_t qkNRound = (qkN + ALIGNED - 1) / ALIGNED * ALIGNED;

            MatmulCoord actualBlockShapeQK{m, qkN, embd};
            LayoutP layoutP(m, qkN, qkNRound);
            LayoutS layoutS(m, qkN, qkNRound);
            int64_t gmOffsetMask = maskBatchOffset + mIdx * mTile * maxSeqlen;
            int64_t gmOffsetS = (int64_t)coreIdx * WORKSPACE_ELENUM + sPingpongFlag * WORKSPACE_ELENUM / 2;
            int64_t gmOffsetP = (int64_t)coreIdx * WORKSPACE_ELENUM + pPingpongFlag * WORKSPACE_ELENUM / 2;
            epilogueFASoftmax(
                gP[gmOffsetP], gS[gmOffsetS], gMask[gmOffsetMask],
                layoutP, layoutS, layoutMask,
                actualBlockShapeQK, qkNIdx, qkReady);
            sPingpongFlag = 1 - sPingpongFlag;
            pPingpongFlag = 1 - pPingpongFlag;
            qkNIdx++;
            arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);

            // mainloop
            uint32_t nEnd = nLoop;
            if (isTriuMask) {
                nEnd = mIdx + 1;
            }
            uint32_t svN = nTile;

            MatrixCoord offsetO{mIdx * mTile, headIdx * embd};
            int64_t gmOffsetO = oAddr + layoutO.GetOffset(offsetO);
            for (uint32_t svNIdx = 0; svNIdx < nEnd; svNIdx++) {
                if (qkNIdx < nEnd) {
                    if (qkNIdx == (nLoop - 1)) {
                        qkN = kvSeqlen - qkNIdx * nTile;
                        qkNRound = (qkN + ALIGNED - 1) / ALIGNED * ALIGNED;
                    }

                    MatmulCoord actualBlockShapeQKNext{m, qkN, embd};
                    LayoutP layoutPNext(m, qkN, qkNRound);
                    LayoutS layoutSNext(m, qkN, qkNRound);
                    gmOffsetMask += nTile;
                    gmOffsetS = (int64_t)coreIdx * WORKSPACE_ELENUM + sPingpongFlag * WORKSPACE_ELENUM / 2;
                    gmOffsetP = (int64_t)coreIdx * WORKSPACE_ELENUM + pPingpongFlag * WORKSPACE_ELENUM / 2;
                    epilogueFASoftmax(
                        gP[gmOffsetP], gS[gmOffsetS], gMask[gmOffsetMask],
                        layoutPNext, layoutSNext, layoutMask,
                        actualBlockShapeQKNext, qkNIdx, qkReady);
                    sPingpongFlag = 1 - sPingpongFlag;
                    pPingpongFlag = 1 - pPingpongFlag;
                    qkNIdx++;
                    arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                }

                if (svNIdx == (nLoop - 1)) {
                    svN = kvSeqlen - svNIdx * nTile;
                }

                MatmulCoord actualBlockShapePV{m, embd, svN};
                int64_t gmOffsetOTmp = (int64_t)coreIdx * WORKSPACE_ELENUM + oTmpPingpongFlag * WORKSPACE_ELENUM / 2;
                arch::CrossCoreWaitFlag(pvReady);
                epilogueFARescaleO(
                    gO[gmOffsetO], gOTmp[gmOffsetOTmp],
                    layoutO, layoutOTmp,
                    actualBlockShapePV, svNIdx, svNIdx == nEnd - 1);
                oTmpPingpongFlag = 1 - oTmpPingpongFlag;
            }
        }
    }

private:
    arch::Resource<ArchTag> resource;
    arch::CrossCoreFlag qkReady{QK_READY_ID};
    arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    arch::CrossCoreFlag pvReady{PV_READY_ID};
};
