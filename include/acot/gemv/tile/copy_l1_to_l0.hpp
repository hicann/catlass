/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_TILE_COPY_L1_TO_L0_HPP
#define ACOT_GEMV_TILE_COPY_L1_TO_L0_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"
#include "acot/gemv/gemv_type.hpp"

namespace acot::gemv::tile
{
    template <
        class ArchTag,
        class GmType>
    struct CopyL1ToL0A
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
    };

    /// Partial specialization for zN in and zZ out.
    // zN -> zZ 可用
    template <class ArchTag, class Element>
    struct CopyL1ToL0A<ArchTag, gemv::GemvType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::zZ;
        using LayoutSrc = layout::zN;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);           // 1个C0的元素数
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element); // 一个基块(分形)的元素数

        // Methods

        ACOT_DEVICE
        CopyL1ToL0A() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst,
            LayoutSrc const &layoutSrc)
        {
            AscendC::LoadData2DParams loadDataParams;
            loadDataParams.startIndex = 0;                                          // 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3)); // 重复次数 = 分形间列方向的分形数
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;   // 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔 = zN矩阵中分形间的列步长，单位：512B。
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1; // 相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，这里实际上就是0，单位：512B。
            loadDataParams.ifTranspose = false;                                    // 是否启用转置功能，对每个分形矩阵进行转置，默认为false
            loadDataParams.addrMode = 0;

            // 在行方向做循环,每次循环处理的行维度是1个分形，总共layoutDst.shape(1)次循环
            for (uint32_t i = 0; i < layoutDst.shape(1); i++)
            {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }
    };

    // 用于矩阵乘列优先的情况，矩阵B由nZ转换成zZ的形式，并传到L0A中,
    // 这种方法应该只适用于方阵，fp16、bf16
    template <class ArchTag, class Element>
    struct CopyL1ToL0A<ArchTag, gemv::GemvType<Element, layout::ColumnMajor>>
    {
        using LayoutDst = layout::zZ;
        using LayoutSrc = layout::nZ;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        ACOT_DEVICE
        CopyL1ToL0A() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {

            AscendC::LoadData2DParams loadDataParams;

            loadDataParams.startIndex = 0;                                                                          // 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）
            loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1))); //
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;                                   // 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔 = zN矩阵中分形间的列步长，单位：512B。
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1; // 相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，这里实际上就是0，单位：512B。
            loadDataParams.ifTranspose = true;                                     // 是否启用转置功能，对每个分形矩阵进行转置，默认为false
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++)
            {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }
    };

    /// Partial specialization for int8_t, nZ in and zZ out. (Transpose A)
    template <class ArchTag>
    struct CopyL1ToL0A<ArchTag, gemv::GemvType<int8_t, layout::ColumnMajor>>
    {
        using Element = int8_t;
        using LayoutDst = layout::zZ;
        using LayoutSrc = layout::nZ;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyL1ToL0A() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::LoadData2dTransposeParams loadDataParams;

            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
            loadDataParams.srcStride = 1;
            loadDataParams.dstGap = 0;
            loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

            for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++)
            {
                AscendC::LoadDataWithTranspose(
                    dstTensor[i * layoutDst.stride(1) * 2],
                    srcTensor[i * layoutSrc.stride(1)],
                    loadDataParams);
            }
        }
    };

    template <
        class ArchTag,
        class GmType>
    struct CopyL1ToL0B
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
    };

    /// Partial specialization for int8_t, zN in and nZ out.
    template <class ArchTag>
    struct CopyL1ToL0B<ArchTag, gemv::GemvType<int8_t, layout::RowMajor>>
    {
        using Element = int8_t;
        using LayoutDst = layout::nZ;
        using LayoutSrc = layout::zN;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyL1ToL0B() {};

        ACOT_DEVICE
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

            for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++)
            {
                AscendC::LoadDataWithTranspose(
                    dstTensor[i * layoutDst.stride(1)],
                    srcTensor[i * layoutSrc.stride(1) * 2],
                    loadDataParams);
            }
        }
    };

    /// Partial specialization for zN in and nZ out.
    template <class ArchTag, class Element>
    struct CopyL1ToL0B<ArchTag, gemv::GemvType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::nZ;
        using LayoutSrc = layout::zN;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyL1ToL0B() {};

        ACOT_DEVICE
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

            for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++)
            {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }
    };

    /// Partial specialization for nZ in and nZ out. (Transpose B)
    template <class ArchTag, class Element>
    struct CopyL1ToL0B<ArchTag, gemv::GemvType<Element, layout::ColumnMajor>>
    {
        using LayoutDst = layout::nZ;
        using LayoutSrc = layout::nZ;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyL1ToL0B() {};

        ACOT_DEVICE
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

            for (uint32_t i = 0; i < layoutDst.shape(1); i++)
            {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }
    };

    template <
        class ArchTag,
        class GmType>
    struct CopyL1BToL0B
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1B to l0B, can not find the specialization.");
    };

    // 模仿上面那段代码造一个L1B -> L0B rowMajor, zN-> zN
    template <class ArchTag, class Element>
    struct CopyL1BToL0B<ArchTag, gemv::GemvType<Element, layout::RowMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::zN;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyL1BToL0B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::LoadData2DParams loadDataParams;

            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1)); // 分形间的行个数
            loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;   // 分形间的行步长
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < layoutDst.shape(3); i++)
            {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(3)], srcTensor[i * layoutSrc.stride(3)], loadDataParams);
            }
        }
    };

    // L1B -> L0B colummMajor, nN -> zN
    // fp16或bf16使用
    template <class ArchTag, class Element>
    struct CopyL1BToL0B<ArchTag, gemv::GemvType<Element, layout::ColumnMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::nN;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

        // Methods

        ACOT_DEVICE
        CopyL1BToL0B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::LoadData2DParams loadDataParams;

            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = layoutSrc.shape(1) * layoutSrc.shape(3);
            loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL; // 分形间的行步长
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = true;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        };
    };

    // L1B -> L0B colummMajor, nN -> zN
    // fp32使用
    template <class ArchTag>
    struct CopyL1BToL0B<ArchTag, gemv::GemvType<float, layout::ColumnMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::nN;
        using Element = float;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);           // 32 / 4 = 8
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element); // 512 / 4 = 128

        // Methods

        ACOT_DEVICE
        CopyL1BToL0B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::LoadData2dTransposeParams loadDataParams;

            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0))); // 迭代次数，行方向分形数/2，因为一次迭代是对2个分形操作
            loadDataParams.srcStride = 1;                                                                           // 相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔，单位是(16*16*4B))
            loadDataParams.dstGap = 0;                                                                              // 相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）
            loadDataParams.dstFracGap = CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)) - 1;                     // 每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为（单位：512B）。

            for (uint32_t i = 0; i < CeilDiv<2 * ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
            {
                AscendC::LoadDataWithTranspose(
                    dstTensor[i * layoutDst.stride(3) * 2], // i * 分形间的列步长
                    srcTensor[i * layoutSrc.stride(3)],     // i * 分形间的列步长
                    loadDataParams);
            }
        };
    };

    // L1B -> L0B colummMajor, nZ -> zN   int8_t
    // int8使用
    template <class ArchTag>
    struct CopyL1BToL0B<ArchTag, gemv::GemvType<int8_t, layout::ColumnMajor>>
    {
        using LayoutDst = layout::zN;
        using LayoutSrc = layout::nZ;
        using Element = int8_t;

        static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);           // 32 / 1 = 32
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element); // 512 / 1 = 512

        // Methods

        ACOT_DEVICE
        CopyL1BToL0B() {};

        ACOT_DEVICE
        void operator()(
            AscendC::LocalTensor<Element> const &dstTensor,
            AscendC::LocalTensor<Element> const &srcTensor,
            LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
        {
            AscendC::LoadData2dTransposeParams loadDataParams;

            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0))); // 迭代次数，行方向分形数/2，因为一次迭代是对2个分形操作
            loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL / 2;                           // 相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔，单位是(32*32*1B))
            loadDataParams.dstGap = 1;                                                                          // 相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）
            loadDataParams.dstFracGap = 0;                                                                      // 每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为（单位：512B）。

            for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
            {
                AscendC::LoadDataWithTranspose(
                    dstTensor[i * layoutDst.stride(3)],     // i * 分形间的列步长
                    srcTensor[i * layoutSrc.stride(3) * 2], // i * 分形间的列步长
                    loadDataParams);
            }
        }
    };

} // namespace acot::gemv::tile

#endif // ACOT_GEMV_TILE_COPY_L1_TO_L0_HPP