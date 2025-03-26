/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_GEMM_TILE_COPY_L1_TO_L0B_HPP
#define ACT_GEMM_TILE_COPY_L1_TO_L0B_HPP

#include "act/act.hpp"
#include "act/layout/layout.hpp"
#include "act/gemm/gemm_type.hpp"
#include "tla/tensor.hpp"

using namespace tla;

namespace Act::Gemm::Tile {

template <
    class ArchTag,
    class L1Type
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

/// Partial specialization for int8_t, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<int8_t, layout::zN>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    ACT_DEVICE
    CopyL1ToL0B() {};

    ACT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

/// Partial specialization for zN in and nZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    ACT_DEVICE
    CopyL1ToL0B() {};

    ACT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for nZ in and nZ out. (Transpose B)
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

    ACT_DEVICE
    CopyL1ToL0B() {};

    ACT_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyL1ToL0B, AtlasA2, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Arch::AtlasA2, Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::A1>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<ElementDst, LayoutDst_>::value &&
                     tla::detail::iszN<ElementSrc, LayoutSrc_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::B2>;
    using TensorSrc = Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    ACT_DEVICE
    TileCopyTla() {};

    ACT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterStrideRow = get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(dstTensor.data()[i * dstOuterStrideRow],
                              srcTensor.data()[i * srcOuterStrideRow],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Arch::AtlasA2, Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::A1>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<ElementDst, LayoutDst_>::value &&
                     tla::detail::isnZ<ElementSrc, LayoutSrc_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::B2>;
    using TensorSrc = Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(ElementSrc);

    // Mehtods

    ACT_DEVICE
    TileCopyTla() {};

    ACT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterStrideRow = get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterShapeCol = get<1, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(dstTensor.data()[i * dstOuterStrideRow],
                              srcTensor.data()[i * srcOuterStrideRow],
                              loadDataParams);
        }
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, int8_t, zN in and nZ out.
template <class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Arch::AtlasA2, Tensor<AscendC::LocalTensor<int8_t>, LayoutSrc_, AscendC::TPosition::A1>,
    Tensor<AscendC::LocalTensor<int8_t>, LayoutDst_, AscendC::TPosition::B2>,
    std::enable_if_t<tla::detail::isnZ<int8_t, LayoutDst_>::value &&
                     tla::detail::iszN<int8_t, LayoutSrc_>::value>> {
    using Element = int8_t;
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<Element>, LayoutDst, AscendC::TPosition::B2>;
    using TensorSrc = Tensor<AscendC::LocalTensor<Element>, LayoutSrc, AscendC::TPosition::A1>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Mehtods

    ACT_DEVICE
    TileCopyTla() {};

    ACT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        const uint32_t srcOuterShapeCol = get<1, 1>(srcTensor.shape());
        const uint32_t srcOuterStrideRow = get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = get<0, 1>(dstTensor.shape());
        const uint32_t dstOuterStrideRow = get<0, 1>(dstTensor.stride());

        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = srcOuterShapeCol;
        loadDataParams.srcStride = srcOuterStrideCol / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadDataWithTranspose(dstTensor.data()[i * dstOuterStrideRow],
                                           srcTensor.data()[i * srcOuterStrideRow * 2],
                                           loadDataParams);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Act::Gemm::Tile

#endif // ACT_GEMM_TILE_COPY_L1_TO_L0B_HPP