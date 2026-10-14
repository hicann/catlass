/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BATCHED_MATMUL_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BATCHED_MATMUL_SOFTMAX_HPP

#include <cstdint>

#include <tiling/tiling_api.h>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/batched_matmul_softmax_tla_tuning.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag_, class Element_>
struct BlockEpilogueBatchedMatmulSoftmax {
    using ArchTag = ArchTag_;
    using Element = Element_;
    using TensorType = Gemm::GemmType<Element, layout::RowMajor>;
    using CopyGmToUb = Tile::CopyGm2Ub<ArchTag, TensorType>;
    using CopyUbToGm = Tile::CopyUb2Gm<ArchTag, TensorType>;

    CATLASS_DEVICE
    BlockEpilogueBatchedMatmulSoftmax(Arch::Resource<ArchTag>& resource) : resource(resource)
    {}

    CATLASS_DEVICE
    void operator()(
        GM_ADDR ptrC, uint32_t batch, uint32_t m, uint32_t n, uint32_t alignedN, uint32_t rowsPerTile,
        uint32_t useBasicBlock) const
    {
        constexpr uint32_t elementBytes = sizeof(Element);
        constexpr uint32_t elementsPerBlock = BYTE_PER_BLK / elementBytes;
        constexpr uint32_t BASIC_SOFTMAX_ROWS_ALIGNMENT = 8;
        constexpr uint32_t INPUT_EVENT = 0;
        constexpr uint32_t OUTPUT_EVENT = 1;

        if (rowsPerTile == 0) {
            return;
        }

        AscendC::GlobalTensor<Element> gmLogits;
        gmLogits.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrC));

        uint32_t tileElems = rowsPerTile * alignedN;
        constexpr uint32_t srcOffset = 0;
        uint32_t dstOffset = tileElems * elementBytes;
        uint32_t tmpOffset = dstOffset + tileElems * elementBytes;
        uint32_t tmpBytes = ArchTag::UB_SIZE - tmpOffset;
        auto srcLocal = resource.ubBuf.template GetBufferByByte<Element>(srcOffset);
        auto dstLocal = resource.ubBuf.template GetBufferByByte<Element>(dstOffset);
        auto tmpLocal = resource.ubBuf.template GetBufferByByte<uint8_t>(tmpOffset);
        srcLocal.SetSize(tileElems);
        dstLocal.SetSize(tileElems);
        tmpLocal.SetSize(tmpBytes);
        // Host-side tuning supplies the row tile shape; the API fills its execution details here.
        AscendC::tiling::SoftMaxTiling softmaxTiling{};

        uint64_t rows = static_cast<uint64_t>(batch) * m;
        uint64_t tileCount = CeilDiv(rows, static_cast<uint64_t>(rowsPerTile));
        uint32_t vectorCoreIdx = AscendC::GetBlockIdx();
        uint32_t vectorCoreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;
        for (uint64_t tileIdx = vectorCoreIdx; tileIdx < tileCount; tileIdx += vectorCoreNum) {
            uint64_t rowStart = tileIdx * rowsPerTile;
            uint32_t actualRows =
                static_cast<uint32_t>(AscendC::Std::min(static_cast<uint64_t>(rowsPerTile), rows - rowStart));
            uint64_t gmOffset = rowStart * static_cast<uint64_t>(n);
            layout::RowMajor gmLayout(actualRows, n, n);
            layout::RowMajor ubLayout(actualRows, n, alignedN);

            copyGmToUb(srcLocal, gmLogits[gmOffset], ubLayout, gmLayout);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INPUT_EVENT);
            AscendC::SoftMaxShapeInfo shapeInfo{actualRows, alignedN, actualRows, n};
            bool useBasicBlockForTile =
                useBasicBlock != 0 && rowsPerTile % BASIC_SOFTMAX_ROWS_ALIGNMENT == 0 && actualRows == rowsPerTile;
            if (useBasicBlockForTile) {
                AscendC::SoftMax<Element, false, true>(dstLocal, srcLocal, tmpLocal, softmaxTiling, shapeInfo);
            } else {
                AscendC::SoftMax<Element>(dstLocal, srcLocal, tmpLocal, softmaxTiling, shapeInfo);
            }
            // The next DMA must not overwrite input still being read by Vector.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_EVENT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
            copyUbToGm(gmLogits[gmOffset], dstLocal, gmLayout, ubLayout);
            // Preserve output until its GM write completes before reusing UB.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT);
        }
    }

private:
    Arch::Resource<ArchTag>& resource;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BATCHED_MATMUL_SOFTMAX_HPP
