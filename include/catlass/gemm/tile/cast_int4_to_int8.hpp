/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_CAST_INT4_TO_INT8_HPP
#define CATLASS_GEMM_TILE_CAST_INT4_TO_INT8_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class SrcType_,
    class DstType_,
    uint32_t STAGES = 2
>
struct TileCastInt4ToInt8 {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;

    static constexpr uint32_t ELE_NUM_PER_BLK_INT8 = BYTE_PER_BLK / sizeof(ElementSrc);

    static constexpr uint32_t COMPUTE_LEN = 16 * 1024;

    struct Params {};

    /// Construct
    CATLASS_DEVICE
    TileCastInt4ToInt8(Arch::Resource<ArchTag> const &resource, Params const &params) {
        if constexpr (g_coreType == AscendC::AIV) {
            uint32_t ubOffset = 0;
            uint32_t ubInSize = COMPUTE_LEN * sizeof(ElementSrc) / 2;
            uint32_t ubOutSize = COMPUTE_LEN * sizeof(ElementDst);
            uint32_t ubWorkspaceSize = COMPUTE_LEN * sizeof(half);
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                ubInTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
                ubOffset += ubInSize;
                ubOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
                ubOffset += ubOutSize;
                ubWorkspaceList[i] = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
                ubOffset += ubWorkspaceSize;

                ubEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ubEventList[i]);
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~TileCastInt4ToInt8() {
        if constexpr (g_coreType == AscendC::AIV) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ubEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const &gmDst, LayoutDst const &layoutDst,
        AscendC::GlobalTensor<ElementSrc> const &gmSrc, LayoutSrc const &layoutSrc
    )
    {
        uint32_t tilesNum = layoutSrc.shape(0);
        uint32_t tileLen = layoutSrc.shape(1);
        uint32_t tileLenRoundInt8 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT8);
        uint32_t tileLenRoundInt4 = RoundUp((layoutSrc.shape(1) + 1) / 2, ELE_NUM_PER_BLK_INT8);
        uint64_t tileStrideSrc = layoutSrc.stride(0);
        uint64_t tileStrideDst = layoutDst.stride(0);
        if constexpr (std::is_same_v<LayoutSrc, layout::ColumnMajorInt4>) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            tileLenRoundInt8 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT8);
            tileLenRoundInt4 = RoundUp((layoutSrc.shape(0) + 1) / 2, ELE_NUM_PER_BLK_INT8);
            tileStrideSrc = layoutSrc.stride(1);
            tileStrideDst = layoutDst.stride(1);
        }
        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < (tilesNum % AscendC::GetSubBlockNum())) {
            tilesPerAiv++;
        }
        uint64_t taskOffsetSrc = AscendC::GetSubBlockIdx() * tilesPerAiv * ((tileStrideSrc + 1) / 2);
        uint64_t taskOffsetDst = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideDst;
        if (AscendC::GetSubBlockIdx() >= (tilesNum % AscendC::GetSubBlockNum())) {
            taskOffsetSrc += (tilesNum % AscendC::GetSubBlockNum()) * ((tileStrideSrc + 1) / 2);
            taskOffsetDst += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideDst;
        }
        uint32_t tilesPerLoop = 32;
        uint32_t loops = CeilDiv(tilesPerAiv, tilesPerLoop);
        uint32_t pingpong = 0;
        for (uint32_t loopIdx = 0; loopIdx < loops; loopIdx++) {
            uint32_t actualTiles = tilesPerLoop;
            if (loopIdx == loops - 1) {
                actualTiles = tilesPerAiv - loopIdx * tilesPerLoop;
            }
            uint64_t tileOffsetSrc = loopIdx * tilesPerLoop * ((tileStrideSrc + 1) / 2);
            AscendC::DataCopyExtParams dataCopyParamsIn(
                actualTiles, (tileLen + 1) / 2 * sizeof(ElementSrc),
                ((tileStrideSrc + 1) / 2 - (tileLen + 1) / 2) * sizeof(ElementSrc),
                (tileLenRoundInt4 - (tileLen + 1) / 2) / ELE_NUM_PER_BLK_INT8, 0
            );
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);
            AscendC::DataCopyPad(ubInTensorList[pingpong],
                gmSrc[taskOffsetSrc + tileOffsetSrc], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ubEventList[pingpong]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ubEventList[pingpong]);

            int4_workspace[pingpong] = ubInTensorList[pingpong].template ReinterpretCast<AscendC::int4b_t>();

            AscendC::SetFlag<AscendC::HardEvent::S_V>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(ubEventList[pingpong]);

            AscendC::Cast(ubWorkspaceList[pingpong], int4_workspace[pingpong],
                AscendC::RoundMode::CAST_NONE, actualTiles * tileLenRoundInt4 * 2);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);

            AscendC::Cast(ubOutTensorList[pingpong], ubWorkspaceList[pingpong],
                AscendC::RoundMode::CAST_NONE, actualTiles * tileLenRoundInt4 * 2);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);

            uint64_t tileOffsetDst = loopIdx * tilesPerLoop * tileStrideDst;
            AscendC::DataCopyExtParams dataCopyParamsOut(
                actualTiles, tileLen * sizeof(ElementDst),
                (tileLenRoundInt4 * 2 - tileLen) / ELE_NUM_PER_BLK_INT8,
                (tileStrideDst - tileLen) * sizeof(ElementDst), 0
            );
            AscendC::DataCopyPad(gmDst[taskOffsetDst + tileOffsetDst],
                ubOutTensorList[pingpong], dataCopyParamsOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ubEventList[pingpong]);

            pingpong = (pingpong + 1) % STAGES;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementSrc> ubInTensorList[STAGES];
    AscendC::LocalTensor<ElementDst> ubOutTensorList[STAGES];
    AscendC::LocalTensor<half> ubWorkspaceList[STAGES];
    AscendC::LocalTensor<AscendC::int4b_t> int4_workspace[STAGES];

    int32_t ubEventList[STAGES];
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_CAST_INT4_TO_INT8_HPP
