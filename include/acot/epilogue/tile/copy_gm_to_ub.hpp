/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_EPILOGUE_TILE_TILE_COPY_GM_TO_UB_HPP
#define ACOT_EPILOGUE_TILE_TILE_COPY_GM_TO_UB_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul/matmul_type.hpp"
#include "acot/gemm/gemm_type.hpp"

namespace acot::epilogue::tile {

template <
    class ArchTag,
    class GmType
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
};

template <typename Element>
struct CopyGm2Ub<arch::AtlasA2, matmul::MatmulType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    ACOT_DEVICE
    CopyGm2Ub() = default;

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutSrc.shape(0),
            layoutSrc.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) * sizeof(Element),
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_C0,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

// 都是统一的搬运函数
template <typename Element>
struct CopyGm2Ub<arch::AscendC910B3, gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    ACOT_DEVICE
    CopyGm2Ub() = default;

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutSrc layoutDst, LayoutDst layoutSrc)
    {
        // // 进行循环
        uint32_t NAlignment = BYTE_PER_C0 / sizeof(Element); // 对齐32Byte
        uint32_t MActual = layoutSrc.shape(0);
        uint32_t NActual = layoutSrc.shape(1);
        uint32_t stride = layoutSrc.stride(0); // RowMajor
        uint32_t NRound = RoundUp(NActual, NAlignment);
        // 更换搬运参数
        AscendC::DataCopyExtParams params;
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        for(uint32_t MIdx = 0; MIdx < MActual; MIdx++){
            params.blockCount = 1;
            params.blockLen = NRound * sizeof(Element);
            params.srcStride = 0;
            params.dstStride = 0; // 只搬运一个
            params.rsv = 0;
            AscendC::DataCopyPad(dstTensor[MIdx * NRound], srcTensor[MIdx * stride], params, padParams);
        }
    };
};

template <typename Element>
struct CopyGm2Ub<arch::AscendC910B3, gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::ColumnMajor;

    ACOT_DEVICE
    CopyGm2Ub() = default;

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutSrc layoutDst, LayoutDst layoutSrc)
    {
        // // 进行循环
        uint32_t MAlignment = BYTE_PER_C0 / sizeof(Element); // 对齐32Byte
        uint32_t MActual = layoutSrc.shape(1);
        uint32_t NActual = layoutSrc.shape(0);
        uint32_t stride = layoutSrc.stride(1); // RowMajor
        uint32_t MRound = RoundUp(MActual, MAlignment);
        // 更换搬运参数
        AscendC::DataCopyExtParams params;
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        for(uint32_t NIdx = 0; NIdx < NActual; NIdx++){
            params.blockCount = 1;
            params.blockLen = MRound * sizeof(Element);
            params.srcStride = 0;
            params.dstStride = 0; // 只搬运一个
            params.rsv = 0;
            AscendC::DataCopyPad(dstTensor[NIdx * MRound], srcTensor[NIdx * stride], params, padParams);
        }
    };
};

}  // acot::epilogue::tile

#endif
