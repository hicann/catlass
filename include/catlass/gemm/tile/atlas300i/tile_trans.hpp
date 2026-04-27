/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_TILE_TRANS_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_TILE_TRANS_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
namespace Catlass::Gemm::Tile {

template <class ArchTag, class GmType, class UbType>
struct TileTrans {};

///////////////////////////////////////////////////////////
// w8a8 UbTranspose
///////////////////////////////////////////////////////////

template <class ArchTag>
struct TileTrans<
    ArchTag,
    Gemm::GemmType<int8_t, layout::RowMajor, AscendC::TPosition::VECCALC>,
    Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    TileTrans() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::LocalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = layoutSrc.stride(0) / ELE_NUM_PER_C0;
        transDataParams.dstRepStride = (layoutSrc.stride(0) / ELE_NUM_PER_C0 == 1 ? 0 : ELE_NUM_PER_C0);
        transDataParams.srcRepStride = (layoutSrc.stride(0) / ELE_NUM_PER_C0 == 1 ? 0 : 1);
        for (uint32_t i = 0; i < CeilDiv(layoutSrc.shape(0), ELE_NUM_PER_C0); i++) {
            AscendC::LocalTensor<int8_t> srcTensorList0[16];
            AscendC::LocalTensor<int8_t> dstTensorList0[16];
            AscendC::LocalTensor<int8_t> srcTensorList1[16];
            AscendC::LocalTensor<int8_t> dstTensorList1[16];
            for (uint32_t j = 0; j < 16; j++) {
                srcTensorList0[j] = srcTensor[i * ELE_NUM_PER_C0 * layoutSrc.stride(0) + j * layoutSrc.stride(0)];
                srcTensorList1[j] = srcTensor
                    [i * ELE_NUM_PER_C0 * layoutSrc.stride(0) + j * layoutSrc.stride(0)
                     + C0_NUM_PER_FRACTAL * layoutSrc.stride(0)];
                dstTensorList0[j] = dstTensor[i * layoutDst.stride(1) + j * ELE_NUM_PER_C0];
                dstTensorList1[j] =
                    dstTensor[i * layoutDst.stride(1) + j * ELE_NUM_PER_C0 + C0_NUM_PER_FRACTAL * ELE_NUM_PER_C0];
            }
            transDataParams.dstHighHalf = false;
            transDataParams.srcHighHalf = false;
            TransDataTo5HD(dstTensorList0, srcTensorList0, transDataParams);

            transDataParams.dstHighHalf = true;
            TransDataTo5HD(dstTensorList0, srcTensorList1, transDataParams);

            transDataParams.srcHighHalf = true;
            TransDataTo5HD(dstTensorList1, srcTensorList1, transDataParams);

            transDataParams.dstHighHalf = false;
            TransDataTo5HD(dstTensorList1, srcTensorList0, transDataParams);
        }
    }
};

template <class ArchTag>
struct TileTrans<
    ArchTag,
    Gemm::GemmType<int8_t, layout::ColumnMajor, AscendC::TPosition::VECCALC>,
    Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    TileTrans() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::LocalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = layoutSrc.stride(1) / ELE_NUM_PER_C0;
        transDataParams.dstRepStride =
            (layoutSrc.stride(1) / ELE_NUM_PER_C0 == 1 ? 0 : (layoutDst.stride(1) * 2 / ELE_NUM_PER_C0));
        transDataParams.srcRepStride = (layoutSrc.stride(1) / ELE_NUM_PER_C0 == 1 ? 0 : 1);
        for (uint32_t i = 0; i < CeilDiv(layoutSrc.shape(1), ELE_NUM_PER_C0); i++) {
            AscendC::LocalTensor<int8_t> srcTensorList0[16];
            AscendC::LocalTensor<int8_t> dstTensorList0[16];
            AscendC::LocalTensor<int8_t> srcTensorList1[16];
            AscendC::LocalTensor<int8_t> dstTensorList1[16];
            for (uint32_t j = 0; j < 16; j++) {
                srcTensorList0[j] = srcTensor[i * ELE_NUM_PER_C0 * layoutSrc.stride(1) + j * layoutSrc.stride(1)];
                srcTensorList1[j] = srcTensor
                    [i * ELE_NUM_PER_C0 * layoutSrc.stride(1) + j * layoutSrc.stride(1)
                     + C0_NUM_PER_FRACTAL * layoutSrc.stride(1)];
                dstTensorList0[j] = dstTensor[i * layoutDst.stride(3) + j * ELE_NUM_PER_C0];
                dstTensorList1[j] = dstTensor[i * layoutDst.stride(3) + j * ELE_NUM_PER_C0 + layoutDst.stride(1)];
            }
            transDataParams.dstHighHalf = false;
            transDataParams.srcHighHalf = false;
            TransDataTo5HD(dstTensorList0, srcTensorList0, transDataParams);

            transDataParams.dstHighHalf = true;
            TransDataTo5HD(dstTensorList0, srcTensorList1, transDataParams);

            transDataParams.srcHighHalf = true;
            TransDataTo5HD(dstTensorList1, srcTensorList1, transDataParams);

            transDataParams.dstHighHalf = false;
            TransDataTo5HD(dstTensorList1, srcTensorList0, transDataParams);
        }
    }
};

template <class ArchTag>
struct TileTrans<
    ArchTag,
    Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::VECCALC>,
    Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    TileTrans() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::LocalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.orgShape(0) * layoutSrc.orgShape(1));
    }
};

template <class ArchTag>
struct TileTrans<
    ArchTag,
    Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::VECCALC>,
    Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::VECCALC>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    TileTrans() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::LocalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.orgShape(0) * layoutSrc.orgShape(1));
    }
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_MMAD_HPP