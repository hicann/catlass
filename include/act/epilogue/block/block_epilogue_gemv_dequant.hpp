/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_EPILOGUE_BLOCK_EPILOGUE_GEMV_DEQUANT_HPP
#define ACT_EPILOGUE_BLOCK_EPILOGUE_GEMV_DEQUANT_HPP

#include "act/act.hpp"
#include "act/arch/resource.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/matrix_coord.hpp"
#include "act/layout/layout.hpp"
#include "act/detail/callback.hpp"
#include "act/gemv/helper.hpp"
#include "act/gemv_coord.hpp"
#include "act/layout/layout.hpp"

namespace Act::Epilogue::Block {

template <
    uint32_t UB_STAGES_,
    class YType_,
    class ScaleType_,
    class BiasType_,                    // new add
    class ZType_,
    class TileRowBroadcastMul_,
    class TileElemWiseEpilogueAdd_,
    class TileElemWiseEpilogueMul_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequant<UB_STAGES_>,
    YType_,
    ScaleType_,
    BiasType_,                          // new add
    ZType_,
    TileRowBroadcastMul_,
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMul_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using ElementScale = bfloat16_t;
    using LayoutScale = typename ScaleType_::Layout;    //长度为m的vector
    using ElementPerTokenScale = float;
    //new add
    using ElementBias = bfloat16_t;    
    using LayoutBias = typename BiasType_::Layout;  //长度为m的vector

    using ElementZ = bfloat16_t;
    using LayoutZ = typename ZType_::Layout;
    using ElementCompute  = float;

    // Check data infos
    static_assert(
        std::is_same_v<ElementY, int32_t> && (std::is_same_v<ElementZ, half> || std::is_same_v<ElementZ, bfloat16_t>) &&
            std::is_same_v<ElementScale, ElementZ> && std::is_same_v<ElementPerTokenScale, float> &&
            std::is_same_v<ElementBias, ElementZ>,
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutY, layout::VectorLayout> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
            std::is_same_v<LayoutBias, layout::VectorLayout> && 
            std::is_same_v<LayoutZ, layout::VectorLayout>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    using LayoutComputeInUb = layout::VectorLayout;

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
    using TileElemWiseEpilogueMul = TileElemWiseEpilogueMul_;

    // Tile copy
    using CopyGmToUbY = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;    //存疑
    using CopyGmToUbBias = typename TileCopy_::CopyGmToUbX;     //存疑
    using CopyUbToGmZ = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueMul::COMPUTE_LENGTH;

    using TensorCoord = layout::VectorLayout::TensorCoord;

    struct Params {
        GM_ADDR ptrScale{nullptr};
        LayoutScale layoutScale{};
        ElementPerTokenScale PerTokenScale;     //标量
        GM_ADDR ptrBias{nullptr};
        LayoutBias layoutBias{};
        GM_ADDR ptrZ{nullptr};
        LayoutZ layoutZ{};

        ACT_HOST_DEVICE
        Params() {};

        ACT_HOST_DEVICE
        Params(
            GM_ADDR ptrScale_, LayoutScale const &layoutScale_,
            ElementPerTokenScale PerTokenScale_,
            GM_ADDR ptrBias_, LayoutBias const &layoutBias_,
            GM_ADDR ptrZ_, LayoutZ const &layoutZ_
        ) : ptrScale(ptrScale_), layoutScale(layoutScale_),
            PerTokenScale(PerTokenScale_),
            ptrBias(ptrBias_), layoutBias(layoutBias_),
            ptrZ(ptrZ_), layoutZ(layoutZ_) {}
    };

    ACT_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;

        ubY = resource.ubBuf.template GetBufferByByte<ElementY>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(ElementY);

        ubScale = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(ElementScale);

        ubBias = resource.ubBuf.template GetBufferByByte<ElementBias>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(ElementBias);

        ubZ = resource.ubBuf.template GetBufferByByte<ElementZ>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(ElementZ);

        ubYFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(float);

        ubScaleFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(float);

        ubBiasFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(float);

        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(float);

        ubZFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += COMPUTE_LENGTH * sizeof(float);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    ACT_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    ACT_DEVICE
    void operator() (
        TensorCoord const& blockOffsetMN,
        TensorCoord const &actualBlockShapeMN,
        AscendC::GlobalTensor<ElementY> const &gmBlockY,
        LayoutY const &layoutBlockY
    )
    {
        // Calculate the offset of the current block
        TensorCoord actualBlockShape = actualBlockShapeMN;
        TensorCoord blockOffset = blockOffsetMN;

        TensorCoord subblockShape{
            CeilDiv(actualBlockShape[0], static_cast<uint32_t>(AscendC::GetSubBlockNum()))
        };
        TensorCoord subblockCoord{static_cast<uint32_t>(AscendC::GetSubBlockIdx())};

        TensorCoord actualSubblockShape = TensorCoord::Min(subblockShape, actualBlockShape - subblockCoord * subblockShape);
        TensorCoord subblockOffset = subblockCoord * subblockShape;

        // Get the data and layout of Y
        auto gmSubblockY = gmBlockY[layoutBlockY.GetOffset(subblockOffset)];
        auto layoutSubblockY = layoutBlockY.GetTileLayout(actualSubblockShape);

        // Get the data and layout of gmScale
        AscendC::GlobalTensor<ElementScale> gmScale;    //m维向量
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(params.ptrScale));
        auto gmTileScale = gmScale[params.layoutScale.GetOffset(blockOffset + subblockOffset)];
        auto layoutGmTileScale = params.layoutScale.GetTileLayout(actualSubblockShape);


        //Get the data and layout of gmBias
        AscendC::GlobalTensor<ElementBias> gmBias;    //m维向量
        gmBias.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBias*>(params.ptrBias));
        auto gmTileBias = gmBias[params.layoutBias.GetOffset(blockOffset + subblockOffset)];
        auto layoutGmTileBias = params.layoutBias.GetTileLayout(actualSubblockShape);

        //Get the data and layout of Z
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementZ*>(params.ptrZ));
        auto gmSubblockZ = gmZ[params.layoutZ.GetOffset(blockOffset + subblockOffset)];
        auto layoutSubblockZ = params.layoutZ.GetTileLayout(actualSubblockShape);

        // get the layout on UB
        auto layoutComputeInUb = LayoutComputeInUb::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);


        // get the Y's layout on UB
        auto layoutUbY = LayoutY::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);
        //load Y(A*X) from gm to ub
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        copyGmToUbY(ubY, gmSubblockY, layoutComputeInUb, layoutSubblockY);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // cast Y
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast<ElementCompute, ElementY>(ubYFp32, ubY, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        // get the scale's layout on UB
        auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);

        // load Scale from gm to ub
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        copyGmToUbScale(ubScale, gmTileScale, layoutComputeInUb, layoutGmTileScale);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // cast Scale
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast<ElementCompute, ElementScale>(ubScaleFp32, ubScale, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        // get the bias's layout on UB
        auto layoutUbBias = LayoutBias::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);
        // load Bias from gm to ub
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        copyGmToUbBias(ubBias, gmTileBias, layoutComputeInUb, layoutGmTileBias);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // cast Bias
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast<ElementCompute, ElementBias>(ubBiasFp32, ubBias, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
        AscendC::PipeBarrier<PIPE_V>();

        // broadcastmul
        tileRowBroadcastMul(ubMul, ubYFp32, ubScaleFp32);
        AscendC::PipeBarrier<PIPE_V>();

        // multiply pertoken_scale
        tileEpilogueMul(ubMul, ubMul, params.PerTokenScale);
        AscendC::PipeBarrier<PIPE_V>();

        // add bias
        tileEpilogueAdd(ubZFp32, ubMul, ubBiasFp32);
        AscendC::PipeBarrier<PIPE_V>();

        //cast ubZFp32 to ubZ
        AscendC::Cast<ElementZ, ElementCompute>(ubZ, ubZFp32, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // get the Z's ublayout
        auto layoutUbZ = LayoutZ::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        copyUbToGmZ(gmSubblockZ, ubZ, layoutSubblockZ, layoutUbZ);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    };

private:
    Params params;

    AscendC::LocalTensor<ElementY> ubY;
    AscendC::LocalTensor<ElementScale> ubScale;
    AscendC::LocalTensor<ElementBias> ubBias;
    AscendC::LocalTensor<ElementZ> ubZ;


    AscendC::LocalTensor<float> ubYFp32;
    AscendC::LocalTensor<float> ubScaleFp32;
    AscendC::LocalTensor<float> ubBiasFp32;
    AscendC::LocalTensor<float> ubMul;
    AscendC::LocalTensor<float> ubZFp32;

    TileRowBroadcastMul tileRowBroadcastMul;

    TileElemWiseEpilogueAdd tileEpilogueAdd;
    TileElemWiseEpilogueMul tileEpilogueMul;

    CopyGmToUbY copyGmToUbY;
    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbBias copyGmToUbBias;
    CopyUbToGmZ copyUbToGmZ;
};

}  // namespace Act::Epilogue::Block

#endif  // ACT_EPILOGUE_BLOCK_EPILOGUE_GEMV_DEQUANT_HPP
