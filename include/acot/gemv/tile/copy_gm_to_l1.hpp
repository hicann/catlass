/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_TILE_COPY_GM_TO_L1_HPP
#define ACOT_GEMV_TILE_COPY_GM_TO_L1_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/matmul/matmul_type.hpp"

namespace acot::gemv::tile
{
    template <class ArchTag, class GmType>
    struct CopyGmToL1
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
    };

    // Partial specialization for AtlasA2, RowMajor in and zN out.
    template <class Element>
    struct CopyGmToL1<arch::AtlasA2, matmul::MatmulType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::RowMajor;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

        // default constructor
        ACOT_DEVICE
        CopyGmToL1() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc)
        {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.dValue = layoutSrc.shape(1);
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(0) < STRIDE_LIMIT)
            {
                intriParams.nValue = layoutSrc.shape(0);
                intriParams.srcDValue = layoutSrc.stride(0);
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;

                for (uint32_t i = 0; i < layoutSrc.shape(0); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        }
    };

    /// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
    template <class Element>
    struct CopyGmToL1<arch::AtlasA2, matmul::MatmulType<Element, layout::ColumnMajor>>
    {
        using LayoutDst = layout::nZ;
        using LayoutSrc = layout::ColumnMajor;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyGmToL1() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.dValue = layoutSrc.shape(0);
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(1) < STRIDE_LIMIT)
            {
                intriParams.nValue = layoutSrc.shape(1);
                intriParams.srcDValue = layoutSrc.stride(1);
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(1); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
                }
            }
        }
    };

    /// Partial specialization for zN in and zN out.
    template <
        class ArchTag,
        class Element>
    struct CopyGmToL1<ArchTag, matmul::MatmulType<Element, layout::zN>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::zN;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Mehtods

        ACOT_DEVICE
        CopyGmToL1() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
            uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0));

            AscendC::DataCopyParams repeatParams;

            if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT)
            {
                repeatParams.blockCount = blockCount;
                repeatParams.blockLen = blockLen;
                repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
                repeatParams.dstStride = 0;
                AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
            }
            else
            {
                repeatParams.blockCount = 1;
                repeatParams.blockLen = blockLen;
                repeatParams.srcStride = 0;
                repeatParams.dstStride = 0;
                for (uint32_t i = 0; i < blockCount; i++)
                {
                    uint64_t dstOffset = i * layoutDst.stride(3);
                    uint64_t srcOffset = i * layoutSrc.stride(3);
                    AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
                }
            }
        }
    };

    /// Partial specialization for nZ in and nZ out.
    template <
        class ArchTag,
        class Element>
    struct CopyGmToL1<ArchTag, matmul::MatmulType<Element, layout::nZ>>
    {
        using LayoutDst = layout::nZ;
        using LayoutSrc = layout::nZ;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyGmToL1() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
            uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1));

            AscendC::DataCopyParams repeatParams;

            if (layoutSrc.stride(1) / ELE_NUM_PER_C0 < STRIDE_LIMIT)
            {
                repeatParams.blockCount = blockCount;
                repeatParams.blockLen = blockLen;
                repeatParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen;
                repeatParams.dstStride = 0;
                AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
            }
            else
            {
                repeatParams.blockCount = 1;
                repeatParams.blockLen = blockLen;
                repeatParams.srcStride = 0;
                repeatParams.dstStride = 0;
                for (uint32_t i = 0; i < blockCount; i++)
                {
                    uint64_t dstOffset = i * layoutDst.stride(1);
                    uint64_t srcOffset = i * layoutSrc.stride(1);
                    AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
                }
            }
        }
    };

    template <class ArchTag, class GmType>
    struct CopyGmToL1A
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1A, can not find the specialization.");
    };
    // Partial specialization for AtlasA2, RowMajor in and zN out.
    template <class Element>
    struct CopyGmToL1A<arch::AtlasA2, matmul::MatmulType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::RowMajor;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

        // default constructor
        ACOT_DEVICE
        CopyGmToL1A() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc)
        {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.dValue = layoutSrc.shape(1);
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(0) < STRIDE_LIMIT)
            {
                intriParams.nValue = layoutSrc.shape(0);
                intriParams.srcDValue = layoutSrc.stride(0);
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;

                for (uint32_t i = 0; i < layoutSrc.shape(0); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        }
    };

    template <class ArchTag, class GmType>
    struct CopyGmToL1B
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1B, can not find the specialization.");
    };

    // Partial specialization for AtlasA2, RowMajor in and zN out.
    template <class Element>
    struct CopyGmToL1B<arch::AtlasA2, matmul::MatmulType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::RowMajor;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

        // default constructor
        ACOT_DEVICE
        CopyGmToL1B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc)
        {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.dValue = layoutSrc.shape(1);
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(0) < STRIDE_LIMIT)
            {
                intriParams.nValue = layoutSrc.shape(0);
                intriParams.srcDValue = layoutSrc.stride(0);
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;

                for (uint32_t i = 0; i < layoutSrc.shape(0); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        };
    };

    // Partial specialization for AtlasA2, ColumnMajor in and nN out.
    template <class Element>
    struct CopyGmToL1B<arch::AtlasA2, matmul::MatmulType<Element, layout::ColumnMajor>>
    {
        using LayoutDst = layout::nN;
        using LayoutSrc = layout::ColumnMajor;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

        // default constructor
        ACOT_DEVICE
        CopyGmToL1B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc)
        {
            AscendC::Nd2NzParams intriParams;
            // 按列方向切分
            uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
            uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
            uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;
            if (srcNdStride < STRIDE_LIMIT)
            {
                if (ndNum)
                {
                    intriParams.ndNum = ndNum;
                    intriParams.nValue = C0_NUM_PER_FRACTAL;
                    intriParams.dValue = layoutSrc.shape(0);
                    intriParams.srcNdMatrixStride = srcNdStride;
                    intriParams.srcDValue = layoutSrc.stride(1);

                    intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                    intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                    intriParams.dstNzMatrixStride = layoutDst.stride(3);

                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                }

                if (remains)
                {
                    AscendC::Nd2NzParams tailParams;
                    tailParams.ndNum = 1;
                    tailParams.nValue = remains;
                    tailParams.dValue = layoutSrc.shape(0);
                    tailParams.srcNdMatrixStride = srcNdStride;
                    tailParams.srcDValue = layoutSrc.stride(1);

                    tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                    tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                    tailParams.dstNzMatrixStride = 0; //`

                    AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
                }
            }
            else if (layoutSrc.stride(1) < STRIDE_LIMIT)
            {
                for (uint32_t i = 0; i < ndNum; i++)
                {
                    AscendC::Nd2NzParams intriParams;
                    intriParams.ndNum = 1;
                    intriParams.nValue = C0_NUM_PER_FRACTAL;
                    intriParams.dValue = layoutSrc.shape(0);
                    intriParams.srcNdMatrixStride = 0;
                    intriParams.srcDValue = layoutSrc.stride(1);

                    intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                    intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                    intriParams.dstNzMatrixStride = 0;

                    AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
                }
                if (remains)
                {
                    AscendC::Nd2NzParams tailParams;
                    tailParams.ndNum = 1;
                    tailParams.nValue = remains;
                    tailParams.dValue = layoutSrc.shape(0);
                    tailParams.srcNdMatrixStride = 0;
                    tailParams.srcDValue = layoutSrc.stride(1);

                    tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                    tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                    tailParams.dstNzMatrixStride = 0;

                    AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
                }
            }
            else
            {
                for (uint32_t i = 0; i < layoutSrc.shape(1); i++)
                {
                    uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                    uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                    AscendC::Nd2NzParams intriParams;
                    intriParams.ndNum = 1;
                    intriParams.nValue = 1;
                    intriParams.dValue = layoutSrc.shape(0);
                    intriParams.srcNdMatrixStride = 0;
                    intriParams.srcDValue = 0;

                    intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                    intriParams.dstNzNStride = 0;
                    intriParams.dstNzMatrixStride = 0;

                    uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                    uint32_t offsetSrc = i * layoutSrc.stride(1);
                    AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
                }
            }
        };
    };

    // Partial specialization for AtlasA2, ColumnMajor in and nZ out. int8_t
    template <>
    struct CopyGmToL1B<arch::AtlasA2, matmul::MatmulType<int8_t, layout::ColumnMajor>>
    {
        using LayoutDst = layout::nZ;
        using LayoutSrc = layout::ColumnMajor;
        using Element = int8_t;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        // Methods

        // default constructor
        ACOT_DEVICE
        CopyGmToL1B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::GlobalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::Nd2NzParams intriParams;

            intriParams.ndNum = 1;
            intriParams.dValue = layoutSrc.shape(0);
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(1) < STRIDE_LIMIT)
            {
                intriParams.nValue = layoutSrc.shape(1);
                intriParams.srcDValue = layoutSrc.stride(1);
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(1); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
                }
            }
        };
    };

} // namespace acot::gemv::tile

#endif // ACOT_GEMV_TILE_COPY_GM_TO_L1_HPP