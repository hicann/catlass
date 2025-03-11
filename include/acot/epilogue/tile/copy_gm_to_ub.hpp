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

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

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
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

template <typename Element>
struct CopyGm2Ub<arch::AtlasA2, matmul::MatmulType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    ACOT_DEVICE
    CopyGm2Ub() = default;

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            1,
            layoutSrc.shape(0) * sizeof(Element),
            0,
            0,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

/// @brief This copy instruction used to copy per token scale from GM to UB.
/// Copy the scale of shape (m,1) on GM to the first column of shape (m,n) on UB,
/// and pad the first block of each row (i.e. pad to shape (m,8) when element type is float).
/// @tparam ArchTag: Architecture tag.
/// @tparam GmType: Type of data on GM.
template <
    class ArchTag,
    class GmType
>
struct CopyPerTokenScale2Ub {
    static_assert(std::is_same_v<typename GmType::Layout, layout::ColumnMajor>,
        "Unsupported layout for CopyPerTokenScale2Ub.");

    using Element = typename GmType::Element;
    using LayoutSrc = typename GmType::Layout;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    ACOT_DEVICE
    CopyPerTokenScale2Ub() = default;

    ACOT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<Element> padParams;

        dataCopyParams.blockCount = layoutSrc.shape(0);
        dataCopyParams.blockLen = layoutSrc.shape(1) * sizeof(Element);  // per token scale has only one column
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;
        // Pad the data to the complete block
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    }
};

}  // acot::epilogue::tile

#endif
