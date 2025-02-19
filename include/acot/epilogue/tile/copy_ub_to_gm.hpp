/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_EPILOGUE_TILE_TILE_COPY_UB_TO_GM_HPP
#define ACOT_EPILOGUE_TILE_TILE_COPY_UB_TO_GM_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemv/gemv_type.hpp"

namespace acot::epilogue::tile
{

    template <
        class ArchTag,
        class GmType>
    struct CopyUb2Gm
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm, can not find the specialization.");
    };

    template <typename Element>
    struct CopyUb2Gm<arch::AtlasA2, gemv::GemvType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::RowMajor;
        using LayoutSrc = layout::RowMajor;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        ACOT_DEVICE
        CopyUb2Gm() = default;

        ACOT_DEVICE
        void operator()(
            AscendC::GlobalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            layout::RowMajor const &layoutDst,
            layout::RowMajor const &layoutSrc)
        {
            AscendC::DataCopyExtParams dataCopyParams(
                layoutDst.shape(0),
                layoutDst.shape(1) * sizeof(Element),
                (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
                (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element),
                0);
            AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
        }
    };

} // acot::epilogue::tile

#endif
