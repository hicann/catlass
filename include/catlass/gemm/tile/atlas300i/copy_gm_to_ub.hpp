/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_COPY_GM_TO_UB_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_COPY_GM_TO_UB_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemm::Tile {

template <class ArchTag, class GmType, class UbType = void>
struct CopyGmToUB {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
};

/// new add 310Pw8a8
template <class ArchTag>
struct CopyGmToUB<
    ArchTag,
    Gemm::GemmType<int8_t, layout::RowMajor, AscendC::TPosition::GM>,
    Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    CopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::GlobalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopyParams intriParams;
        // intriParams.blockCount = (layoutSrc.shape(1) + 31) / 32;
        intriParams.blockCount = CeilDiv(layoutSrc.shape(1), ELE_NUM_PER_C0);
        intriParams.blockLen = 1;
        intriParams.srcGap = 0;
        intriParams.dstGap = C0_NUM_PER_FRACTAL - 1; // 改过
        for (uint32_t i = 0; i < CeilDiv(layoutSrc.shape(0), C0_NUM_PER_FRACTAL); i++) {
            if (i != CeilDiv(layoutSrc.shape(0), C0_NUM_PER_FRACTAL) - 1) {
                for (uint32_t j = 0; j < C0_NUM_PER_FRACTAL; j++) {
                    AscendC::DataCopy(
                        dstTensor[i * layoutDst.stride(1) + j * layoutDst.stride(0)],
                        srcTensor[i * layoutSrc.stride(0) * C0_NUM_PER_FRACTAL + j * layoutSrc.stride(0)], intriParams
                    );
                }
            } else {
                for (uint32_t j = 0; j < (layoutSrc.shape(0) - 1) % C0_NUM_PER_FRACTAL + 1; j++) {
                    AscendC::DataCopy(
                        dstTensor[i * layoutDst.stride(1) + j * layoutDst.stride(0)],
                        srcTensor[i * layoutSrc.stride(0) * C0_NUM_PER_FRACTAL + j * layoutSrc.stride(0)], intriParams
                    );
                }
            }
        }
    }
};
template <class ArchTag>
struct CopyGmToUB<
    ArchTag,
    Gemm::GemmType<int8_t, layout::RowMajor, AscendC::TPosition::GM>,
    Gemm::GemmType<int8_t, layout::RowMajor, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    CopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::GlobalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopyParams datacopyParams;
        datacopyParams.blockCount = 1;
        datacopyParams.blockLen = (layoutSrc.shape(1) + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0;
        datacopyParams.srcGap = 0;
        datacopyParams.dstGap = 0;
        for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
            AscendC::DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], datacopyParams);
        }
    }
};

template <class ArchTag>
struct CopyGmToUB<
    ArchTag,
    Gemm::GemmType<int8_t, layout::ColumnMajor, AscendC::TPosition::GM>,
    Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    CopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::GlobalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = CeilDiv(layoutSrc.shape(0), ELE_NUM_PER_C0);
        intriParams.blockLen = 1;
        intriParams.srcGap = 0;
        intriParams.dstGap = layoutDst.stride(1) / BYTE_PER_C0 - 1;
        for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
            AscendC::DataCopy(dstTensor[i * layoutDst.stride(2)], srcTensor[i * layoutSrc.stride(1)], intriParams);
        }
    }
};

template <class ArchTag>
struct CopyGmToUB<
    ArchTag,
    Gemm::GemmType<int8_t, layout::ColumnMajor, AscendC::TPosition::GM>,
    Gemm::GemmType<int8_t, layout::ColumnMajor, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    CopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::GlobalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopyParams datacopyParams;
        datacopyParams.blockCount = 1;
        datacopyParams.blockLen = (layoutSrc.shape(0) + ELE_NUM_PER_C0 - 1) / ELE_NUM_PER_C0;
        datacopyParams.srcGap = 0;
        datacopyParams.dstGap = 0;
        for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
            AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], datacopyParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToUB<
    ArchTag,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods
    CATLASS_DEVICE
    CopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopyParams datacopyParams;
        datacopyParams.blockCount = 1;
        datacopyParams.blockLen = CeilDiv(layoutSrc.shape(0), ELE_NUM_PER_C0);
        datacopyParams.srcGap = 0;
        datacopyParams.dstGap = 0;
        AscendC::DataCopy(dstTensor, srcTensor, datacopyParams);
    }
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP