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
#include "acot/gemv/gemv_type.hpp"

constexpr uint32_t STRIDE_LIMIT = 65536;

namespace acot::gemv::tile
{
    template <class ArchTag, class GmType>
    struct CopyGmToL1
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
    };

    // Partial specialization for AtlasA2, RowMajor in and zN out.
    template <class Element>
    struct CopyGmToL1<arch::AtlasA2, gemv::GemvType<Element, layout::RowMajor>>
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

            intriParams.ndNum = 1;                                            // 传输nd矩阵块多少块
            intriParams.dValue = layoutSrc.shape(1);                          // nd矩阵列数
            intriParams.srcNdMatrixStride = 0;                                // 相邻nd矩阵起始地址间的偏移，这里只传输一块，所以为0
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0; // 分形间的列步长，单位是32B，所以要除以32B对应的元素数
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(0) < STRIDE_LIMIT)
            {                                                                    // 一次传完
                intriParams.nValue = layoutSrc.shape(0);                         // nd矩阵行数
                intriParams.srcDValue = layoutSrc.stride(0);                     // 相邻行起始地址间的偏移，其实就是每一行的元素个数
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0; // Z型矩阵相邻行起始地址之间的偏移,就是分形内的行步长，单位是32B，所以要除以32B对应的元素数,实际上就是1
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else // 原矩阵行步长太大，一次无法传完，逐行传
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;    // 相邻行起始地址间的偏移，逐行传不考虑
                intriParams.dstNzNStride = 0; // Z型矩阵相邻行起始地址之间的偏移,逐行传不考虑

                for (uint32_t i = 0; i < layoutSrc.shape(0); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.shape(0)], intriParams);
                }
            }
        }
    };

    // 对向量的传输，重写了一个搬运函数  新增
    //  Partial specialization for AtlasA2, VectorLayout in and zN out.
    template <class Element>
    struct CopyGmToL1<arch::AtlasA2, gemv::GemvType<Element, layout::VectorLayout>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::VectorLayout;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

        // Methods

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
            intriParams.ndNum = 1;                                            // 传输nd矩阵块多少块
            intriParams.dValue = layoutSrc.stride(0);                         // nd矩阵列数
            intriParams.srcNdMatrixStride = 0;                                // 相邻nd矩阵起始地址间的偏移，这里只传输一块，所以为0
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0; // 分形间的列步长，单位是32B，所以要除以32B对应的元素数
            intriParams.dstNzMatrixStride = 0;

            intriParams.nValue = 1;                      // nd矩阵行数
            intriParams.srcDValue = layoutSrc.stride(0); // 相邻行起始地址间的偏移，其实就是每一行的元素个数
            intriParams.dstNzNStride = 0;                // Z型矩阵相邻行起始地址之间的偏移,只传1行不考虑

            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        }
    };

    /// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
    // 对于列优先来说其实就是行列互换，然后还是转成zN
    template <class Element>
    struct CopyGmToL1<arch::AtlasA2, gemv::GemvType<Element, layout::ColumnMajor>>
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

            intriParams.ndNum = 1;                                            // 传输nd矩阵块多少块
            intriParams.dValue = layoutSrc.shape(0);                          // nd矩阵列数(对于列优先矩阵就是行数)
            intriParams.srcNdMatrixStride = 0;                                // 相邻nd矩阵起始地址间的偏移，这里只传输一块，所以为0
            intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0; // 分形间的列步长，单位是32B，所以要除以32B对应的元素数
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(1) < STRIDE_LIMIT) // 一次传完
            {
                intriParams.nValue = layoutSrc.shape(1); // nd矩阵列数
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
    struct CopyGmToL1<ArchTag, gemv::GemvType<Element, layout::zN>>
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
    struct CopyGmToL1<ArchTag, gemv::GemvType<Element, layout::nZ>>
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

    ////////////////////////////////////////////////////////////////
    // 以下是使用到的函数

    ////////////////////////////////////////////////////////////////
    template <class ArchTag, class GmType>
    struct CopyGmToL1A
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1A, can not find the specialization.");
    };
    // Partial specialization for AtlasA2, RowMajor in and zN out.
    template <class Element>
    struct CopyGmToL1A<arch::AtlasA2, gemv::GemvType<Element, layout::RowMajor>>
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

            intriParams.ndNum = 1;                                            // 传输nd矩阵块多少块
            intriParams.dValue = layoutSrc.shape(1);                          // nd矩阵列数
            intriParams.srcNdMatrixStride = 0;                                // 相邻nd矩阵起始地址间的偏移，这里只传输一块，所以为0
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0; // 分形间的列步长，单位是32B，所以要除以32B对应的元素数
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(0) < STRIDE_LIMIT)
            {                                                                    // 一次传完
                intriParams.nValue = layoutSrc.shape(0);                         // nd矩阵行数
                intriParams.srcDValue = layoutSrc.stride(0);                     // 相邻行起始地址间的偏移，其实就是每一行的元素个数
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0; // Z型矩阵相邻行起始地址之间的偏移,就是分形内的行步长，单位是32B，所以要除以32B对应的元素数,实际上就是1
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else // 原矩阵行步长太大，一次无法传完，逐行传
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;    // 相邻行起始地址间的偏移，逐行传不考虑
                intriParams.dstNzNStride = 0; // Z型矩阵相邻行起始地址之间的偏移,逐行传不考虑

                for (uint32_t i = 0; i < layoutSrc.shape(0); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.shape(0)], intriParams);
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
    struct CopyGmToL1B<arch::AtlasA2, gemv::GemvType<Element, layout::RowMajor>>
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

            intriParams.ndNum = 1;                                            // 传输nd矩阵块多少块
            intriParams.dValue = layoutSrc.shape(1);                          // nd矩阵列数
            intriParams.srcNdMatrixStride = 0;                                // 相邻nd矩阵起始地址间的偏移，这里只传输一块，所以为0
            intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0; // 分形间的列步长，单位是32B，所以要除以32B对应的元素数
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(0) < STRIDE_LIMIT)
            {                                                                    // 一次传完
                intriParams.nValue = layoutSrc.shape(0);                         // nd矩阵行数
                intriParams.srcDValue = layoutSrc.stride(0);                     // 相邻行起始地址间的偏移，其实就是每一行的元素个数
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0; // Z型矩阵相邻行起始地址之间的偏移,就是分形内的行步长，单位是32B，所以要除以32B对应的元素数,实际上就是1
                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }
            else // 原矩阵行步长太大，一次无法传完，逐行传
            {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;    // 相邻行起始地址间的偏移，逐行传不考虑
                intriParams.dstNzNStride = 0; // Z型矩阵相邻行起始地址之间的偏移,逐行传不考虑

                for (uint32_t i = 0; i < layoutSrc.shape(0); i++)
                {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.shape(0)], intriParams);
                }
            }
        };
    };

    // Partial specialization for AtlasA2, ColumnMajor in and nN out.
    template <class Element>
    struct CopyGmToL1B<arch::AtlasA2, gemv::GemvType<Element, layout::ColumnMajor>>
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
            uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1); // 源nd矩阵之间的距离
            uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;        // 连续nd矩阵之间的数量，将源nd矩阵分成列为16的连续矩阵，通过repeattime接口一次性传过去
            uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;      // 看源nd矩阵是否存在尾矩阵
            if (srcNdStride < STRIDE_LIMIT)
            { // 情况1：源nd矩阵能1次性搬完的情况，一次性连续搬ndNum个连续小nd矩阵
                if (ndNum)
                {
                    intriParams.ndNum = ndNum;                   // 传输nd矩阵的数目
                    intriParams.nValue = C0_NUM_PER_FRACTAL;     // 矩阵行数
                    intriParams.dValue = layoutSrc.shape(0);     // 矩阵列数，16
                    intriParams.srcNdMatrixStride = srcNdStride; // 源操作数相邻nd矩阵起始地址间的偏移，单位element
                    intriParams.srcDValue = layoutSrc.stride(1); // 源操作数相邻nd矩阵起始地址间的偏移

                    intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0; // 相邻基块之间头和头的距离，就是分形间的行步长，单位是C0_SIZE（32B）
                    intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;  // 目的矩阵中基块内列步长，单位为C0_SIZE（32B）

                    intriParams.dstNzMatrixStride = layoutDst.stride(3); // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移，就是分形间的列步长，单位element

                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                }

                if (remains)
                {
                    AscendC::Nd2NzParams tailParams;
                    tailParams.ndNum = 1;                       // 传输nd矩阵的数目
                    tailParams.nValue = remains;                // 矩阵行数
                    tailParams.dValue = layoutSrc.shape(0);     // 矩阵列数，小于C0_NUM_PER_FRACTAL
                    tailParams.srcNdMatrixStride = srcNdStride; //
                    tailParams.srcDValue = layoutSrc.stride(1); //

                    tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                    tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                    tailParams.dstNzMatrixStride = 0; //`

                    AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
                }
            }
            else if (layoutSrc.stride(1) < STRIDE_LIMIT) // 一次搬不完，但是stride又不是特别大的情况,此时需要循环，一次搬1个小nd矩阵
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
            else // 当stride很大的时候，此时只能一列一列的搬运
            {
                for (uint32_t i = 0; i < layoutSrc.shape(1); i++)
                {
                    uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;   // 第几个小nd矩阵
                    uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL; // 小nd矩阵中的第几列

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
    struct CopyGmToL1B<arch::AtlasA2, gemv::GemvType<int8_t, layout::ColumnMajor>>
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

            intriParams.ndNum = 1;                                            // 传输nd矩阵块多少块
            intriParams.dValue = layoutSrc.shape(0);                          // nd矩阵列数(对于列优先矩阵就是行数)
            intriParams.srcNdMatrixStride = 0;                                // 相邻nd矩阵起始地址间的偏移，这里只传输一块，所以为0
            intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0; // 分形间的列步长，单位是32B，所以要除以32B对应的元素数
            intriParams.dstNzMatrixStride = 0;

            if (layoutSrc.stride(1) < STRIDE_LIMIT) // 一次传完
            {
                intriParams.nValue = layoutSrc.shape(1); // nd矩阵列数
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