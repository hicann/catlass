/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_SFMG_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_SFMG_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"
#include "fag_common/common_header.h"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TQue;

using namespace AscendC;
namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class UpdateType_,
    class InputType_>
class BlockEpilogue<
    EpilogueAtlasA2FAGSfmg,
    OutputType_,
    UpdateType_,
    InputType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGSfmg;
    using ArchTag = typename DispatchPolicy::ArchTag;

    

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, TPipe *pipe_in, __gm__ uint8_t *dout, __gm__ uint8_t *out,
    __gm__ uint8_t *cu_seq_qlen, __gm__ uint8_t *workspace, int32_t batchIn, __gm__ uint8_t * tiling_in)
    {
        batch = batchIn;
        cBlockIdx = GetBlockIdx();
        pipe = pipe_in;

        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)tiling_in);
        batch = tilingData.GetValue(TILING_B);
        total_q = tilingData.GetValue(TILING_T1);
        nheads_k = tilingData.GetValue(TILING_N2);
        g = tilingData.GetValue(TILING_G);
        headdim = tilingData.GetValue(TILING_D);

        int64_t sfmgWorkspaceOffset = tilingData.GetValue(TILING_SFMG_WORKSPACE_OFFSET);
        int64_t mm1WorkspaceOffset = tilingData.GetValue(TILING_MM1_WORKSPACE_OFFSET);
        int64_t mm2WorkspaceOffset = tilingData.GetValue(TILING_MM2_WORKSPACE_OFFSET);
        nheads = nheads_k * g;
        dAlign = (headdim + 15) / 16 * 16;
        cu_seq_qlen_addr = cu_seq_qlen;

        n_stride = (nheads - 1) * headdim * sizeof(half);

        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)tiling_in);;
        uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);

        softmaxGradTilingData.srcM = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2);
        softmaxGradTilingData.srcK = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2);
        softmaxGradTilingData.srcSize = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 2);
        softmaxGradTilingData.outMaxM = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 3);
        softmaxGradTilingData.outMaxK = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 4);
        softmaxGradTilingData.outMaxSize = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 5);
        softmaxGradTilingData.splitM = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 6);
        softmaxGradTilingData.splitK = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 7);
        softmaxGradTilingData.splitSize = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 8);
        softmaxGradTilingData.reduceM = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 9);
        softmaxGradTilingData.reduceK = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 10);
        softmaxGradTilingData.reduceSize = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 11);
        softmaxGradTilingData.rangeM = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 12);
        softmaxGradTilingData.tailM = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 13);
        softmaxGradTilingData.tailSplitSize = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 14);
        softmaxGradTilingData.tailReduceSize = tilingDataU32.GetValue(TILING_SOFTMAX_GRAD_TILING_DATA * CONST_2 + 15);

        // 计算 buffer 大小
        constexpr static uint32_t inputBufferLen = 24 * 1024; // castBuffer 24K*2=48K
        constexpr static uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
        uint32_t outputBufferLen = (castBufferLen + dAlign - 1) / dAlign * 8;
        uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

        // 计算单核的计算量
        int64_t normalAxisSize = total_q * nheads;
        normalCoreSize = (normalAxisSize + coreNum -1) / coreNum;
        usedCoreNum = (normalAxisSize + normalCoreSize -1) / normalCoreSize;

        // 计算单loop的计算量及loop次数
        singleLoopNBurstNum = inputBufferLen / sizeof(float) / dAlign;
        normalCoreLoopTimes = (normalCoreSize + singleLoopNBurstNum -1) / singleLoopNBurstNum;
        normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;

        int64_t tailCoreSize = normalAxisSize - (usedCoreNum - 1) * normalCoreSize;
        tailCoreLoopTimes = (tailCoreSize + singleLoopNBurstNum -1) / singleLoopNBurstNum;
        tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

        // 初始化 buffer
        pipe->InitBuffer(inBuffer1, inputBufferLen); // 24K
        pipe->InitBuffer(inBuffer2, inputBufferLen); // 24K
        pipe->InitBuffer(cast1Buf, castBufferLen); // 48K
        pipe->InitBuffer(cast2Buf, castBufferLen); // 48K
        pipe->InitBuffer(outBuffer1, outputBufferLen);
        pipe->InitBuffer(tmpBuf, tempBufferLen); // 40K - outputBufferLen

        // 初始化 GM
        doutGm.SetGlobalBuffer((__gm__ half *)dout);
        outGm.SetGlobalBuffer((__gm__ half *)out);
        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + sfmgWorkspaceOffset / sizeof(float));
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void InitIndex(int64_t startIdx, int64_t& curS, GM_ADDR seqS)
    {
        int64_t totalLen = 0;
        for (int64_t bDimIdx = bIdx; bDimIdx < batch; bDimIdx++) {
            totalLen = nheads * ((__gm__ int64_t *)seqS)[bDimIdx] * headdim;
            if (totalLen > startIdx) {
                bIdx = bDimIdx;
                curS = (bIdx == 0) ? ((__gm__ int64_t *)seqS)[bIdx] :
                                        (((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1]);
                int64_t bTail = startIdx - (totalLen - nheads * curS * headdim);
                nIdx = bTail / (curS * headdim);
                int64_t nTail = bTail % (curS * headdim);
                sIdx = nTail / headdim;
                break;
            }
        }
    }

    CATLASS_DEVICE
    void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset, GM_ADDR seqS)
    {
        int64_t srcOffset = 0;
        int64_t bOffset = bIdx == 0 ? 0 : nheads * ((__gm__ int64_t *)seqS)[bIdx - 1] * headdim;
        srcOffset = bOffset + (sIdx * nheads + nIdx) * headdim;

        DataCopyPad(input1Buf[dstOffset], doutGm[srcOffset],
                    {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(headdim * sizeof(half)),
                    static_cast<uint32_t>(n_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dAlign - headdim)), 0});
        DataCopyPad(input2Buf[dstOffset], outGm[srcOffset],
                    {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(headdim * sizeof(half)),
                    static_cast<uint32_t>(n_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dAlign - headdim)), 0});
    }

    CATLASS_DEVICE
    void CopyInSfmg(int64_t leftNburst, int64_t &curS, GM_ADDR seqS)
    {
        int64_t dstOffset = 0;
        while (leftNburst > 0) {
            int64_t curNburst = 0;
            if (curS - sIdx < leftNburst) { // 需要借N或借B
                curNburst = curS - sIdx;
                DoCopyIn(curS, curNburst, dstOffset, seqS);
                leftNburst = leftNburst - curNburst;
                sIdx = 0;
                if (nIdx < nheads - 1) { // 需要借N
                    nIdx += 1;
                } else {
                    nIdx = 0;
                    if (bIdx < batch - 1) { // 需要借B
                        bIdx += 1;
                        curS = ((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1];
                    } else { // 没有轴可以借了，end
                        leftNburst = 0;
                    }
                }
            } else {  // 当前S够用
                curNburst = leftNburst;
                DoCopyIn(curS, curNburst, dstOffset, seqS);
                sIdx = sIdx + leftNburst;
                leftNburst = 0;
            }
            dstOffset = dstOffset + curNburst * dAlign;
        }
    }
    
    CATLASS_DEVICE
    void operator()()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        event_t VWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t VWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        event_t Mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        uint32_t usedCoreNums = usedCoreNum;
        if (cBlockIdx < usedCoreNums) {
            LocalTensor<uint8_t> tempBuf = tmpBuf.Get<uint8_t>();
            LocalTensor<float> sfmgClc1 = cast1Buf.Get<float>();
            LocalTensor<float> sfmgClc2 = cast2Buf.Get<float>();

            int64_t singleCoreLoopTimes = normalCoreLoopTimes;
            int64_t singleCoreLastLoopNBurstNum = normalCoreLastLoopNBurstNum; // 普通单核最后一次loop处理多少个D
            if (cBlockIdx == usedCoreNums - 1) {
                singleCoreLoopTimes = tailCoreLoopTimes;
                singleCoreLastLoopNBurstNum = tailCoreLastLoopNBurstNum;
            }

            int64_t startIdx = cBlockIdx * normalCoreSize;
            int64_t nBurst = singleLoopNBurstNum;
            int64_t curS = 0;

            for (int64_t i = 0; i < singleCoreLoopTimes; i++) {
                if (i == singleCoreLoopTimes - 1) {
                    nBurst = singleCoreLastLoopNBurstNum;
                }

                // copyIn
                if (i == 0) {
                    input1Buf = inBuffer1.Get<half>();
                    input2Buf = inBuffer2.Get<half>();
                    InitIndex((startIdx + i * singleLoopNBurstNum) * headdim,
                            curS, cu_seq_qlen_addr);
                    CopyInSfmg(nBurst, curS, cu_seq_qlen_addr);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(VWaitMte2);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(VWaitMte2);

                // cast 1
                int64_t calcSize = nBurst * dAlign;
                Cast(sfmgClc1, input1Buf, RoundMode::CAST_NONE, calcSize);
                AscendC::PipeBarrier<PIPE_V>();

                // cast 2
                Cast(sfmgClc2, input2Buf, RoundMode::CAST_NONE, calcSize);
                AscendC::PipeBarrier<PIPE_V>();

                // pre copyIn next nBurst
                if (i < singleCoreLoopTimes - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Mte2WaitV);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Mte2WaitV);
                    int64_t nextNBurst = i == singleCoreLoopTimes - 2 ? singleCoreLastLoopNBurstNum : nBurst;
                    input1Buf = inBuffer1.Get<half>();
                    input2Buf = inBuffer2.Get<half>();
                    InitIndex((startIdx + (i + 1) * singleLoopNBurstNum) * headdim,
                            curS, cu_seq_qlen_addr);
                    CopyInSfmg(nextNBurst, curS, cu_seq_qlen_addr);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(VWaitMte2);
                }

                if (i > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(VWaitMte3);
                }

                // sfmg
                outputBuf = outBuffer1.Get<float>();
                Duplicate<float>(outputBuf, 0.0, nBurst * 8);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t shapeArray[] = {static_cast<uint32_t>(nBurst), static_cast<uint32_t>(dAlign)};
                sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
                sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
                uint32_t shapeArray1[] = {static_cast<uint32_t>(nBurst), BLOCK_BYTE_SIZE / sizeof(float)};
                outputBuf.SetShapeInfo(ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));

                bool isBasicBlock = (nBurst % SFMG_HIGH_PERF_N_FACTOR == 0) && (dAlign % SFMG_HIGH_PERF_D_FACTOR == 0);
                if (likely(isBasicBlock)) {
                    SoftmaxGradFront<float, true>(outputBuf, sfmgClc1, sfmgClc2, tempBuf, softmaxGradTilingData);
                } else {
                    SoftmaxGradFront<float, false>(outputBuf, sfmgClc1, sfmgClc2, tempBuf, softmaxGradTilingData);
                }
                AscendC::PipeBarrier<PIPE_V>();

                // copyOut
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

                int64_t sfmgOutputOffset = (startIdx + i * singleLoopNBurstNum) * BLOCK_SIZE;
                DataCopy(sfmgWorkspaceGm[sfmgOutputOffset], outputBuf, nBurst * BLOCK_SIZE);
                if (i < singleCoreLoopTimes - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(VWaitMte3);
                }
                
            }
        }
    }
protected:
    /// Data members
    constexpr static int64_t BLOCK_BYTE_SIZE = 32;
    constexpr static int64_t BLOCK_SIZE = 8;
    constexpr static int64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static int64_t SFMG_HIGH_PERF_D_FACTOR = 64;

    TPipe *pipe;
    uint32_t cBlockIdx;

    GlobalTensor<float> sfmgWorkspaceGm;
    GlobalTensor<half> doutGm;
    GlobalTensor<half> outGm;
    TBuf<QuePosition::VECIN> inBuffer1, inBuffer2;
    TBuf<> cast1Buf, cast2Buf, tmpBuf;
    TBuf<QuePosition::VECOUT> outBuffer1;

    int64_t batch;
    int64_t nheads;
    int64_t nheads_k;
    int64_t g;
    int64_t total_q;
    int64_t headdim;
    int64_t dAlign;
    GM_ADDR cu_seq_qlen_addr;

    int64_t bIdx = 0;
    int64_t nIdx = 0;
    int64_t sIdx = 0;

    int64_t dstOffset = 0;
    int64_t n_stride = 0;

    int64_t usedCoreNum;
    int64_t normalCoreSize;
    int64_t singleLoopNBurstNum;
    int64_t normalCoreLoopTimes;
    int64_t normalCoreLastLoopNBurstNum;
    int64_t tailCoreLoopTimes;
    int64_t tailCoreLastLoopNBurstNum;

    LocalTensor<half> input1Buf;
    LocalTensor<half> input2Buf;
    LocalTensor<float> outputBuf;

    SoftMaxTiling softmaxGradTilingData;
};
    
}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_SFMG_HPP
