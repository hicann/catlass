/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP
#define ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/resource.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/gemv/helper.hpp"
#include "AscendCT/gemv_coord.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/matrix_coord.hpp"

namespace AscendCT::epilogue::block {

template <
    class tempType_,
    class YType_,
    class ZType_,
    class TileElemWiseEpilogueAdd_,
    class TileElemWiseEpilogueMul_,
    class TileCopy_
>
class BlockEpilogue<
    EpilogueAtlasA2ElemWiseOneSource,
    tempType_,
    YType_,
    ZType_,
    TileElemWiseEpilogueAdd_,
    TileElemWiseEpilogueMul_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2ElemWiseOneSource;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementTemp = typename tempType_::Element;
    using LayoutTemp = typename tempType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using ElementZ = typename ZType_::Element;
    using LayoutZ = typename ZType_::Layout;

    using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
    using TileElemWiseEpilogueMul = TileElemWiseEpilogueMul_;

    using CopyGmToUbY = typename TileCopy_::CopyGmToUbY;
    using CopyGmToUbTemp = typename TileCopy_::CopyGmToUbTemp;
    using CopyUbToGmZ = typename TileCopy_::CopyUbToGmZ;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueMul::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;

    static constexpr bool noNeedCast = std::is_same<ElementTemp, ElementY>::value;

    using ElementCompute = typename AscendCT::gemv::helper::ElementAccumulatorSelector<ElementY, ElementZ>::ElementAccumulator;
    using ElementScalar = ElementCompute;
    using TensorCoord = layout::VectorLayout::TensorCoord;

    // Check the layout type of Y, Temp and Z
    // static_assert(std::is_same_v<LayoutY, layout::RowMajor> && std::is_same_v<LayoutTemp, layout::RowMajor> &&
    //                   std::is_same_v<LayoutZ, layout::RowMajor>,
    //               "Layout type of Y, Temp and Z must be RowMajor");

    static_assert(std::is_same_v<LayoutY, layout::VectorLayout> && std::is_same_v<LayoutTemp, layout::VectorLayout> &&
        std::is_same_v<LayoutZ, layout::VectorLayout>,
    "Layout type of Y, Temp and Z must be VectorLayout");

    using LayoutComputeInUb = layout::VectorLayout;

    // Check if ArchTag is matched
    static_assert(std::is_same_v<typename TileElemWiseEpilogueMul::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    // Check if compute length is valid
    static_assert(COMPUTE_LENGTH * OPERANDS_NUM * sizeof(ElementCompute) <= ArchTag::UB_SIZE, "UB out of bounds");

    struct Params {
        ElementScalar alpha;
        ElementScalar beta;
        GM_ADDR ptrY;
        LayoutY layoutY;

        GM_ADDR ptrZ;
        LayoutZ layoutZ;

        // Methods
        ASCENDCT_HOST_DEVICE
        Params() {}

        ASCENDCT_HOST_DEVICE
        Params(ElementScalar alpha_, ElementScalar beta_, GM_ADDR ptrY_, LayoutTemp layoutY_, GM_ADDR ptrZ_, LayoutZ layoutZ_)
            : alpha(alpha_), beta(beta_), ptrY(ptrY_), layoutY(layoutY_), ptrZ(ptrZ_), layoutZ(layoutZ_) {}
    };

    ASCENDCT_DEVICE
    BlockEpilogue(arch::Resource<ArchTag>& resource, Params const& params): params(params) 
    {
        ubTemp = resource.ubBuf.template GetBufferByByte<ElementTemp>(0);

        ubY = resource.ubBuf.template GetBufferByByte<ElementY>(COMPUTE_LENGTH * sizeof(ElementTemp));
        ubYCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(COMPUTE_LENGTH * sizeof(ElementTemp));

        ubZ = resource.ubBuf.template GetBufferByByte<ElementZ>(
            COMPUTE_LENGTH * sizeof(ElementY) + COMPUTE_LENGTH * sizeof(ElementTemp));
        ubZCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(
            COMPUTE_LENGTH * sizeof(ElementY) + COMPUTE_LENGTH * sizeof(ElementTemp));

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    ASCENDCT_DEVICE
    ~BlockEpilogue() 
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    ASCENDCT_DEVICE
    void operator()(
        TensorCoord const& blockOffsetMN,
        TensorCoord const& actualBlockShapeMN,
        AscendC::GlobalTensor<ElementCompute> const& gmBlockTemp,
        LayoutTemp const& layoutBlockTemp) 
    {
        TensorCoord actualBlockShape = actualBlockShapeMN;
        TensorCoord blockOffset = blockOffsetMN;

        TensorCoord subblockShape{
            CeilDiv(actualBlockShape[0], static_cast<uint32_t>(AscendC::GetSubBlockNum()))
        };
        TensorCoord subblockCoord{static_cast<uint32_t>(AscendC::GetSubBlockIdx())};

        TensorCoord actualSubblockShape = TensorCoord::Min(subblockShape, actualBlockShape - subblockCoord * subblockShape);
        TensorCoord subblockOffset = subblockCoord * subblockShape;

        // Get the data and layout of Temp
        auto gmSubblockTemp = gmBlockTemp[layoutBlockTemp.GetOffset(subblockOffset)];
        auto layoutSubblockTemp = layoutBlockTemp.GetTileLayout(actualSubblockShape);

        // Get the data and layout of y
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params.ptrY));
        auto gmSubblockY = gmY[params.layoutY.GetOffset(blockOffset + subblockOffset)];
        auto layoutSubblockY = params.layoutY.GetTileLayout(actualSubblockShape);

        // Get the data and layout of Z
        AscendC::GlobalTensor<ElementZ> gmZ;
        gmZ.SetGlobalBuffer(reinterpret_cast<__gm__ ElementZ*>(params.ptrZ));
        auto gmSubblockZ = gmZ[params.layoutZ.GetOffset(blockOffset + subblockOffset)];
        auto layoutSubblockZ = params.layoutZ.GetTileLayout(actualSubblockShape);

        // get the layout on UB
        auto layoutComputeInUb = LayoutComputeInUb::template MakeLayoutInUb<ElementCompute>(actualSubblockShape);

        // load Temp(A*x) from gm to ub
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        copyGmToUbTemp(ubTemp, gmSubblockTemp, layoutComputeInUb, layoutSubblockTemp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // compute Temp * alpha
        tileEpilogueMul(ubTemp, ubTemp, params.alpha);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        // load Y from gm to ub
        copyGmToUbY(ubY, gmSubblockY, layoutComputeInUb, layoutSubblockY);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // compute Y * beta
        if constexpr (!noNeedCast) {
            AscendC::Cast<ElementCompute, ElementY>(ubYCast, ubY, AscendC::RoundMode::CAST_NONE, COMPUTE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
            tileEpilogueMul(ubYCast, ubYCast, params.beta);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            tileEpilogueMul(ubY, ubY, params.beta);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if constexpr (!noNeedCast) {
            tileEpilogueAdd(ubZCast, ubTemp, ubYCast);
        } else {
            tileEpilogueAdd(ubZ, ubTemp, ubY);
        }

        if constexpr (!noNeedCast) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<ElementZ, ElementCompute>(ubZ, ubZCast, AscendC::RoundMode::CAST_RINT, COMPUTE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        copyUbToGmZ(gmSubblockZ, ubZ, layoutSubblockZ, layoutComputeInUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    };

private:
    Params params;

    AscendC::LocalTensor<ElementY> ubY;
    AscendC::LocalTensor<ElementCompute> ubYCast;
    AscendC::LocalTensor<ElementTemp> ubTemp;
    AscendC::LocalTensor<ElementZ> ubZ;
    AscendC::LocalTensor<ElementCompute> ubZCast;

    TileElemWiseEpilogueAdd tileEpilogueAdd;
    TileElemWiseEpilogueMul tileEpilogueMul;

    CopyGmToUbY copyGmToUbY;
    CopyGmToUbTemp copyGmToUbTemp;
    CopyUbToGmZ copyUbToGmZ;
};

}  // namespace AscendCT::epilogue::block

#endif  // ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMV_HPP