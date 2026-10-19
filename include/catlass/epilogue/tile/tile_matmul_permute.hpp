/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_MATMUL_PERMUTE_HPP
#define CATLASS_EPILOGUE_TILE_TILE_MATMUL_PERMUTE_HPP

#include "catlass/catlass.hpp"
#include "catlass/epilogue/matmul_permute_policy.hpp"
#include "basic_api/kernel_operator_intf.h"
#include "tla/tuple.hpp"

namespace Catlass::Epilogue::Tile {

template <
    uint32_t S1_, uint32_t S2_, bool SplitM_, bool UseVectorPack_, uint32_t UB_STAGES_, class ElementOut_,
    class ElementIn_>
class TileMatmulPermute {
public:
    using DispatchPolicy = EpilogueAscend950MatmulPermute<S1_, S2_, SplitM_, UseVectorPack_, UB_STAGES_>;
    using ElementOut = ElementOut_;
    using ElementIn = ElementIn_;
    static constexpr uint32_t S1 = DispatchPolicy::S1;
    static constexpr uint32_t S2 = DispatchPolicy::S2;
    static constexpr bool SPLIT_M = DispatchPolicy::SPLIT_M;
    static constexpr bool USE_VECTOR_PACK = DispatchPolicy::USE_VECTOR_PACK;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static_assert(UB_STAGES > 0 && UB_STAGES <= 2, "MatmulPermute supports one or two UB stages.");

    CATLASS_DEVICE
    TileMatmulPermute() = default;

    template <class TensorOut, class TensorIn, class TensorScratch>
    CATLASS_DEVICE void operator()(
        TensorOut& tensorGmOut, TensorIn& tensorUbIn, TensorScratch& scratchUb, GM_ADDR workspace)
    {
        if (!SPLIT_M && AscendC::GetSubBlockIdx() > 0) {
            return;
        }

        uint32_t blockShapeM = static_cast<uint32_t>(tla::get<0>(tensorGmOut.shape()));
        uint32_t blockShapeN = static_cast<uint32_t>(tla::get<1>(tensorGmOut.shape()));
        uint32_t halfBlockShapeM = CeilDiv(blockShapeM, AscendC::GetTaskRation());
        uint32_t localMOffset = 0;
        if constexpr (SPLIT_M) {
            localMOffset = halfBlockShapeM * (AscendC::GetSubBlockIdx() & 0x1);
            blockShapeM = (static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL ?
                              (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                              halfBlockShapeM;
        }
        if (blockShapeM == 0 || blockShapeN == 0) {
            return;
        }

        uint32_t baseM = static_cast<uint32_t>(tla::get<0>(tensorGmOut.coord())) + localMOffset;
        uint32_t baseN = static_cast<uint32_t>(tla::get<1>(tensorGmOut.coord()));
        uint32_t globalN = static_cast<uint32_t>(tla::get<0>(tensorGmOut.stride()));
        uint32_t nOuter = globalN / S2_;
        uint32_t srcStrideM = static_cast<uint32_t>(tla::get<0>(tensorUbIn.stride()));
        // Keep the underlying AscendC GM/UB tensor handles, rather than passing
        // TLA element proxies to DataCopyPad (the 950 compiler rejects those).
        auto outputGm = tensorGmOut.data();
        auto inputUb = tensorUbIn.data();
        AscendC::GlobalTensor<ElementIn> workspaceGm;
        workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementIn*>(workspace));
        AscendC::DataCopyPadExtParams<ElementIn> noPadding{false, 0, 0, 0};
        uint32_t mOuter = baseM / S1_;
        uint32_t mInner = baseM - mOuter * S1_;
        if constexpr (USE_VECTOR_PACK) {
            constexpr uint32_t elementsPerBlock = Catlass::BYTE_PER_BLK / sizeof(ElementIn);
            uint32_t nOuterIdx = baseN % nOuter;
            if (nOuter >= elementsPerBlock && blockShapeN >= elementsPerBlock && blockShapeN % elementsPerBlock == 0 &&
                nOuterIdx + blockShapeN <= nOuter && mInner == 0 && blockShapeM % S1_ == 0 &&
                S1_ * srcStrideM >= blockShapeN) {
                uint32_t targetInner = baseN / nOuter;
                uint32_t groupCount = blockShapeM / S1_;
                if (targetInner < S2_ && groupCount > 0) {
                    uint32_t srcStrideBlocks =
                        (S1_ * srcStrideM - blockShapeN) * sizeof(ElementIn) / Catlass::BYTE_PER_BLK;
                    uint32_t dstStrideBytes = (S2_ * S1_ * nOuter - blockShapeN) * sizeof(ElementOut);
                    AscendC::DataCopyExtParams copyParams(
                        groupCount, blockShapeN * sizeof(ElementIn), srcStrideBlocks, dstStrideBytes, 0);
                    uint32_t dstBase = ((mOuter * S2_ + targetInner) * S1_) * nOuter + nOuterIdx;
                    for (uint32_t rr = 0; rr < S1_; ++rr) {
                        AscendC::DataCopyPad(outputGm[dstBase + rr * nOuter], inputUb[rr * srcStrideM], copyParams);
                    }
                    return;
                }
            }
        }
        if (nOuter <= 8 && baseN == 0 && blockShapeN == globalN && mInner == 0 && blockShapeM % S1_ == 0) {
            constexpr uint32_t elementsPerBlock = Catlass::BYTE_PER_BLK / sizeof(ElementIn);
            for (uint32_t row = 0; row < blockShapeM; row += S1_) {
                for (uint32_t targetInner = 0; targetInner < S2_; ++targetInner) {
                    for (uint32_t rr = 0; rr < S1_; ++rr) {
                        uint32_t dstOffset = (((mOuter * S2_ + targetInner) * S1_ + rr) * nOuter);
                        uint32_t srcOffset = (row + rr) * srcStrideM + targetInner * nOuter;
                        if (srcOffset % elementsPerBlock == 0) {
                            AscendC::DataCopyExtParams copyParams(1, nOuter * sizeof(ElementIn), 0, 0, 0);
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                outputGm[dstOffset], inputUb[srcOffset], copyParams);
                        } else {
                            if (workspace == nullptr) {
                                for (uint32_t nn = 0; nn < nOuter; ++nn) {
                                    outputGm.SetValue(
                                        dstOffset + nn, static_cast<ElementOut>(inputUb.GetValue(srcOffset + nn)));
                                }
                                continue;
                            }
                            uint32_t prefix = srcOffset % elementsPerBlock;
                            uint32_t alignedSrcOffset = srcOffset - prefix;
                            AscendC::DataCopyExtParams dirtyCopyParams(
                                1, (prefix + nOuter) * sizeof(ElementIn), 0, 0, 0);
                            AscendC::DataCopyExtParams cleanCopyParams(1, nOuter * sizeof(ElementIn), 0, 0, 0);
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                workspaceGm, inputUb[alignedSrcOffset], dirtyCopyParams);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                scratchUb, workspaceGm[prefix], cleanCopyParams, noPadding);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                outputGm[dstOffset], scratchUb, cleanCopyParams);
                        }
                    }
                }
                ++mOuter;
            }
            return;
        }
        for (uint32_t row = 0; row < blockShapeM;) {
            uint32_t remainRowsInGroup = S1_ - mInner;
            uint32_t remainRowsInBlock = blockShapeM - row;
            uint32_t groupRows = remainRowsInGroup < remainRowsInBlock ? remainRowsInGroup : remainRowsInBlock;
            for (uint32_t colStart = 0; colStart < blockShapeN;) {
                uint32_t globalCol = baseN + colStart;
                uint32_t targetInner = globalCol / nOuter;
                uint32_t nOuterIdx = globalCol - targetInner * nOuter;
                uint32_t remainColsInOuter = nOuter - nOuterIdx;
                uint32_t remainColsInBlock = blockShapeN - colStart;
                uint32_t segmentLen = remainColsInOuter < remainColsInBlock ? remainColsInOuter : remainColsInBlock;
                uint32_t dstOffset = (((mOuter * S2_ + targetInner) * S1_ + mInner) * nOuter + nOuterIdx);
                for (uint32_t rr = 0; rr < groupRows; ++rr) {
                    uint32_t rowSrcBase = (row + rr) * srcStrideM;
                    constexpr uint32_t elementsPerBlock = Catlass::BYTE_PER_BLK / sizeof(ElementIn);
                    uint32_t srcOffset = rowSrcBase + colStart;
                    if (srcOffset % elementsPerBlock == 0) {
                        AscendC::DataCopyExtParams copyParams(1, segmentLen * sizeof(ElementIn), 0, 0, 0);
                        AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                            outputGm[dstOffset + rr * nOuter], inputUb[srcOffset], copyParams);
                    } else {
                        if (colStart == 0 || workspace == nullptr) {
                            for (uint32_t nn = 0; nn < segmentLen; ++nn) {
                                outputGm.SetValue(
                                    dstOffset + rr * nOuter + nn,
                                    static_cast<ElementOut>(inputUb.GetValue(srcOffset + nn)));
                            }
                        } else {
                            uint32_t prefix = srcOffset % elementsPerBlock;
                            uint32_t alignedSrcOffset = srcOffset - prefix;
                            AscendC::DataCopyExtParams dirtyCopyParams(
                                1, (prefix + segmentLen) * sizeof(ElementIn), 0, 0, 0);
                            AscendC::DataCopyExtParams cleanCopyParams(1, segmentLen * sizeof(ElementIn), 0, 0, 0);
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                workspaceGm, inputUb[alignedSrcOffset], dirtyCopyParams);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                scratchUb, workspaceGm[prefix], cleanCopyParams, noPadding);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DataCopyPad<ElementIn, AscendC::PaddingMode::Compact>(
                                outputGm[dstOffset + rr * nOuter], scratchUb, cleanCopyParams);
                        }
                    }
                }
                colStart += segmentLen;
            }
            row += groupRows;
            mInner += groupRows;
            if (mInner >= S1_) {
                mInner = 0;
                ++mOuter;
            }
        }
    }
};

} // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_TILE_MATMUL_PERMUTE_HPP
